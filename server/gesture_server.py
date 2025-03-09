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
from model import KeyPointClassifier, PointHistoryClassifier
from database import db

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize classifiers
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

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
def handle_create_room():
    room_id = str(uuid.uuid4())
    db.create_room(room_id, request.sid)
    join_room(room_id)
    emit('room-created', {'roomId': room_id})
    print(f'Room created: {room_id}')

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

if __name__ == '__main__':
    print("Starting Gesture Recognition Server...")
    print("Make sure MongoDB is running and .env is configured correctly")
    port = int(os.getenv('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False) 