from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import mediapipe as mp
import numpy as np
import cv2
import base64
from datetime import datetime
import uuid
import itertools
import os
import sys

# Add the server directory to Python path
server_dir = os.path.dirname(os.path.abspath(__file__))
if server_dir not in sys.path:
    sys.path.append(server_dir)

from model import KeyPointClassifier, PointHistoryClassifier
from database import db

app = Flask(__name__)
# Enable CORS for all routes with proper configuration
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type", "Authorization"], "methods": ["GET", "POST", "OPTIONS"]}})
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   ping_timeout=60, 
                   ping_interval=25, 
                   async_mode='threading',
                   logger=True,  # Enable detailed logging
                   engineio_logger=True)  # Enable Engine.IO logging

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize classifiers
print(f"Current working directory: {os.getcwd()}")
print(f"Server directory: {server_dir}")
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# Load the keypoint classifier labels
keypoint_classifier_labels = []
point_history_classifier_labels = []

# Load the label files
def load_labels():
    global keypoint_classifier_labels, point_history_classifier_labels
    
    # Path to keypoint classifier labels
    keypoint_label_path = os.path.join(server_dir, 'model', 'keypoint_classifier_label.csv')
    # If file doesn't exist in server/model, try to use the one from outside server
    if not os.path.exists(keypoint_label_path):
        keypoint_label_path = os.path.join(os.path.dirname(server_dir), 'model', 'keypoint_classifier', 'keypoint_classifier_label.csv')
    
    # Path to point history classifier labels
    point_history_label_path = os.path.join(server_dir, 'model', 'point_history_classifier_label.csv')
    # If file doesn't exist in server/model, try to use the one from outside server
    if not os.path.exists(point_history_label_path):
        point_history_label_path = os.path.join(os.path.dirname(server_dir), 'model', 'point_history_classifier', 'point_history_classifier_label.csv')
    
    # Load keypoint classifier labels
    if os.path.exists(keypoint_label_path):
        with open(keypoint_label_path, 'r', encoding='utf-8') as f:
            keypoint_classifier_labels = [line.strip() for line in f]
        print(f"Loaded {len(keypoint_classifier_labels)} keypoint classifier labels from {keypoint_label_path}")
    else:
        print(f"Warning: Keypoint classifier label file not found at {keypoint_label_path}")
        keypoint_classifier_labels = ["Unknown"]
    
    # Load point history classifier labels
    if os.path.exists(point_history_label_path):
        with open(point_history_label_path, 'r', encoding='utf-8') as f:
            point_history_classifier_labels = [line.strip() for line in f]
        print(f"Loaded {len(point_history_classifier_labels)} point history classifier labels from {point_history_label_path}")
    else:
        print(f"Warning: Point history classifier label file not found at {point_history_label_path}")
        point_history_classifier_labels = ["Unknown"]

# Call the function to load labels
load_labels()

@app.route('/')
def index():
    return "Gesture Recognition Server Running"

@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)
    # Room cleanup will be handled by the database

@socketio.on('create-room')
def handle_create_room(data=None):
    print(f"[Room Creation] Received create-room event from client {request.sid}")
    
    try:
        # Generate a unique room ID
        room_id = str(uuid.uuid4())[:8]
        print(f"[Room Creation] Generated room ID: {room_id}")
        
        # Join the room
        join_room(room_id)
        print(f"[Room Creation] Client {request.sid} joined room {room_id}")
        
        # Save room to database
        try:
            db.create_room(room_id, request.sid)
            print(f"[Room Creation] Room {room_id} saved to database")
        except Exception as db_error:
            print(f"[Room Creation] Database error: {str(db_error)}")
            # Continue even if database save fails
        
        # Create response data
        response_data = {'roomId': room_id, 'success': True}
        
        # Emit to the client who requested the room
        print(f"[Room Creation] Emitting room-created event with data: {response_data}")
        emit('room-created', response_data, to=request.sid)
        
        print(f"[Room Creation] Room creation completed successfully")
        return response_data
        
    except Exception as e:
        error_msg = f"Error creating room: {str(e)}"
        print(f"[Room Creation Error] {error_msg}")
        import traceback
        print(f"[Room Creation Error] Traceback: {traceback.format_exc()}")
        
        error_response = {'error': error_msg, 'success': False}
        emit('error', {'message': error_msg}, to=request.sid)
        return error_response

@socketio.on('join-room')
def handle_join_room(data):
    room_id = data['roomId']
    room = db.get_active_room(room_id)
    
    if room:
        join_room(room_id)
        db.join_room(room_id, request.sid)
        emit('user-joined', {'userId': request.sid}, room=room_id)
        
        # Send gesture history to the new user
        gestures = db.get_room_gestures(room_id)
        emit('gesture-history', {'gestures': gestures})
    else:
        emit('error', {'message': 'Room not found or inactive'})

@socketio.on('leave-room')
def handle_leave_room(data):
    room_id = data['roomId']
    leave_room(room_id)
    db.leave_room(room_id, request.sid)
    emit('user-left', {'userId': request.sid}, room=room_id)

@socketio.on('gesture-frame')
def handle_gesture_frame(data):
    room_id = data['roomId']
    frame_data = data['frame']
    
    # Decode base64 image
    img_bytes = base64.b64decode(frame_data.split(',')[1])
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    
    # Process frame for hand detection
    results = process_frame(frame)
    
    if results:
        # Store gesture in database
        db.add_gesture(room_id, request.sid, results)
        
        # Emit to all users in the room
        emit('gesture-detected', {
            'userId': request.sid,
            'gesture': results
        }, room=room_id)

@app.route('/gestures/<room_id>')
def get_room_gestures(room_id):
    gestures = db.get_room_gestures(room_id)
    return jsonify({
        'room_id': room_id,
        'gesture_count': len(gestures),
        'gestures': gestures
    })

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    status = {
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'environment': os.environ.get('FLASK_ENV', 'production'),
        'database_connected': hasattr(db, 'db') and db.db is not None,
        'models_loaded': {
            'keypoint_classifier': keypoint_classifier is not None,
            'point_history_classifier': point_history_classifier is not None
        }
    }
    return jsonify(status)

def process_frame(frame):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert landmarks to list
            landmark_list = calc_landmark_list(frame, hand_landmarks)
            
            # Pre-process landmarks
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            
            # Get hand sign classification
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            
            return {
                'gesture_id': int(hand_sign_id),
                'landmarks': landmark_list,
                'timestamp': datetime.now().isoformat()
            }
    
    return None

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = landmark_list.copy()
    
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    
    # Convert to one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    
    return temp_landmark_list

@socketio.on('ping')
def handle_ping():
    print(f"Received ping from client {request.sid}")
    emit('pong')

@socketio.on('test-event')
def handle_test_event(data):
    print(f"Received test event from client {request.sid}: {data}")
    emit('test-response', {'received': True, 'message': 'Server received your test event'})

@app.route('/create-room', methods=['POST', 'OPTIONS'])
def create_room_http():
    """HTTP endpoint for room creation as a fallback"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    try:
        # Generate a unique room ID
        room_id = str(uuid.uuid4())[:8]
        print(f"[HTTP Room Creation] Generated room ID: {room_id}")
        
        # Get client ID from request
        client_id = request.json.get('clientId', 'unknown')
        
        # Save room to database
        try:
            db.create_room(room_id, client_id)
            print(f"[HTTP Room Creation] Room {room_id} saved to database")
        except Exception as db_error:
            print(f"[HTTP Room Creation] Database error: {str(db_error)}")
        
        # Create response
        response = jsonify({
            'success': True,
            'roomId': room_id
        })
        
        # Add CORS headers
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        error_msg = f"Error creating room: {str(e)}"
        print(f"[HTTP Room Creation Error] {error_msg}")
        
        # Create error response
        response = jsonify({
            'success': False,
            'error': error_msg
        })
        
        # Add CORS headers
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500

@socketio.on('simple-create-room')
def handle_simple_create_room():
    """Simplified room creation event"""
    print(f"[Simple Room Creation] Received simple-create-room event from client {request.sid}")
    
    try:
        # Generate a unique room ID
        room_id = str(uuid.uuid4())[:8]
        print(f"[Simple Room Creation] Generated room ID: {room_id}")
        
        # Join the room
        join_room(room_id)
        print(f"[Simple Room Creation] Client {request.sid} joined room {room_id}")
        
        # Create response data
        response_data = {'roomId': room_id, 'success': True}
        
        # Emit to the client who requested the room
        print(f"[Simple Room Creation] Emitting simple-room-created event with data: {response_data}")
        emit('simple-room-created', response_data, to=request.sid)
        
        print(f"[Simple Room Creation] Room creation completed successfully")
        
    except Exception as e:
        error_msg = f"Error creating room: {str(e)}"
        print(f"[Simple Room Creation Error] {error_msg}")
        emit('error', {'message': error_msg}, to=request.sid)

@socketio.on('recognize-gesture')
def recognize_gesture(data):
    # Extract the frame from the data
    frame_data = data.get('frame')
    room_id = data.get('roomId')
    
    if not frame_data or not room_id:
        return
    
    try:
        # Convert the base64 image to a numpy array
        image_data = frame_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Process the image for hand landmarks
        image = cv2.flip(image, 1)  # Mirror image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks
                landmark_list = calc_landmark_list(image, hand_landmarks)
                
                # Pre-process landmarks
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                # Classify hand gesture
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                # Get the gesture name
                if 0 <= hand_sign_id < len(keypoint_classifier_labels):
                    gesture_name = keypoint_classifier_labels[hand_sign_id]
                else:
                    gesture_name = "Unknown"
                
                print(f"Recognized gesture: {gesture_name} (ID: {hand_sign_id})")
                
                # Send the result back to the client
                emit('gesture-recognition-result', {
                    'gesture': gesture_name,
                    'gestureId': int(hand_sign_id)
                }, to=request.sid)
                
                # Also store in gesture history
                store_gesture(room_id, request.sid, gesture_name)
                
                # Only process the first hand for simplicity
                break
    except Exception as e:
        print(f"Error in gesture recognition: {e}")
        import traceback
        print(traceback.format_exc())

def store_gesture(room_id, user_id, gesture_name):
    """Store gesture in database"""
    try:
        gesture_data = {
            'userId': user_id,
            'gesture': {'translation': gesture_name},
            'timestamp': datetime.now().isoformat()
        }
        db.add_gesture(room_id, user_id, gesture_data)
    except Exception as e:
        print(f"Error storing gesture: {e}")

if __name__ == '__main__':
    print("Starting Gesture Recognition Server...")
    print("Make sure MongoDB is running and .env is configured correctly")
    port = int(os.getenv('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False) 
