#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import copy
import csv
import itertools
from collections import Counter, deque
import os
import threading
import queue
import math

import cv2 as cv
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
import pyttsx3
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from model import KeyPointClassifier, PointHistoryClassifier
from utils import CvFpsCalc

# Initialize the pyttsx3 TTS engine globally
tts_engine = pyttsx3.init()

# Create a queue for TTS requests
tts_queue = queue.Queue()

# Global variables for volume control
volume_controller = None
min_vol = None
max_vol = None
volume_mode = False  # Track if volume control is active
static_mode = False  # Track if static mode is active
dynamic_mode = False  # Track if dynamic mode is active

def tts_worker():
    """Worker thread that processes TTS requests one by one."""
    while True:
        gesture_label = tts_queue.get()  # Block until an item is available.
        if gesture_label is None:
            break  # Use None as a signal to stop the worker.
        try:
            tts_engine.say(gesture_label)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            tts_queue.task_done()

# Launch the dedicated TTS worker thread.
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)

    args = parser.parse_args()

    return args


def load_point_history_labels(filepath):
    labels = []
    # Use utf-8-sig encoding to automatically handle BOM characters, if any
    with open(filepath, 'r', encoding='utf-8-sig', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Check that the row is not empty and the first element is not just whitespace,
            # and strip any unwanted characters.
            if row and row[0].strip():
                labels.append(row[0].strip())
    return labels


def init_volume_control():
    """Initialize audio device for volume control"""
    global volume_controller, min_vol, max_vol
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_controller = cast(interface, POINTER(IAudioEndpointVolume))
    vol_range = volume_controller.GetVolumeRange()
    min_vol, max_vol = vol_range[0], vol_range[1]


def process_volume_control(image, landmarks):
    """Process hand landmarks for volume control"""
    global volume_controller, min_vol, max_vol
    
    if not landmarks:
        return image
    
    # Get coordinates for thumb and index finger
    h, w, c = image.shape
    thumb_x = int(landmarks[4].x * w)
    thumb_y = int(landmarks[4].y * h)
    index_x = int(landmarks[8].x * w)
    index_y = int(landmarks[8].y * h)
    
    # Draw circles on thumb and index finger
    cv.circle(image, (thumb_x, thumb_y), 15, (255, 255, 255), -1)
    cv.circle(image, (index_x, index_y), 15, (255, 255, 255), -1)
    
    # Draw line between fingers
    length = math.hypot(index_x - thumb_x, index_y - thumb_y)
    line_color = (0, 255, 0) if length >= 50 else (0, 0, 255)
    cv.line(image, (thumb_x, thumb_y), (index_x, index_y), line_color, 3)
    
    # Calculate and set volume
    vol = np.interp(length, [50, 220], [min_vol, max_vol])
    volume_controller.SetMasterVolumeLevel(vol, None)
    
    # Draw volume bar
    vol_bar = np.interp(length, [50, 220], [400, 150])
    vol_percentage = np.interp(length, [50, 220], [0, 100])
    
    cv.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
    cv.rectangle(image, (50, int(vol_bar)), (85, 400), (0, 0, 0), cv.FILLED)
    cv.putText(image, f'{int(vol_percentage)} %', (40, 450),
               cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
    
    return image


def main():
    global volume_mode, static_mode, dynamic_mode  # Add to globals
    
    # Initialize volume control
    init_volume_control()

    # Argument parsing
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    point_history_labels = load_point_history_labels("model/point_history_classifier/point_history_classifier_label.csv")

    # FPS Measurement
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history for dynamic gestures
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    # Mode toggles (initially both are off)
    static_mode = 0
    dynamic_mode = 0

    # Variable to track the last static gesture spoken via TTS
    last_static_gesture = ""

    # Training mode variables
    train_mode = 0
    train_number = -1

    while True:
        fps = cvFpsCalc.get()

        # Process key events for toggling modes and training mode keys
        key = cv.waitKey(10)
        if key == 27:  # ESC to exit
            break

        if key != -1:
            if key == ord('s'):
                static_mode = 0 if static_mode == 1 else 1
                print(f"Static mode: {'on' if static_mode == 1 else 'off'}")
                if not static_mode:
                    train_mode = 0
                    train_number = -1
                # Reset last spoken gesture when toggling static mode.
                last_static_gesture = ""
            if key == ord('d'):
                dynamic_mode = 0 if dynamic_mode == 2 else 2
                print(f"Dynamic mode: {'on' if dynamic_mode == 2 else 'off'}")
                if not dynamic_mode:
                    train_mode = 0
                    train_number = -1
            # Process training keys (digit keys, 'k' for keypoint training and 'h' for point history training)
            if 48 <= key <= 57:  # 0 ~ 9
                train_number = key - 48
            if key == ord('k'):
                if static_mode:
                    train_mode = 1
                    print("Entered Keypoint Training Mode, label:", train_number)
            if key == ord('h'):
                if dynamic_mode:
                    train_mode = 2
                    print("Entered Point History Training Mode, label:", train_number)
            if key == ord('n'):
                train_mode = 0
                train_number = -1
                print("Exited Training Mode")
            # New volume control toggle
            elif key == ord('v'):  # Toggle volume control
                volume_mode = not volume_mode
                print(f"Volume control: {'on' if volume_mode else 'off'}")

        # Camera capture and flip
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Process frame for hand detection
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # Initialize gesture labels for display info
        static_gesture_label = ""
        dynamic_gesture_label = ""

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Only process volume control if mode is active
                if volume_mode:
                    debug_image = process_volume_control(debug_image, hand_landmarks.landmark)
                
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Logging for training mode (if applicable)
                logging_csv(train_number, train_mode,
                            pre_process_landmark(landmark_list),
                            pre_process_point_history(debug_image, list(point_history)))

                # -------------------------------
                # Static mode processing (Keypoint gestures with TTS)
                if static_mode:
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    static_pred = keypoint_classifier(pre_processed_landmark_list)
                    static_gesture_label = keypoint_classifier_labels[static_pred]
                    # Speak the static gesture only if it has changed
                    if static_gesture_label != last_static_gesture:
                        speak_gesture_async(static_gesture_label)
                        last_static_gesture = static_gesture_label
                # -------------------------------
                # Dynamic mode processing (Point History gestures with green bubbles)
                if dynamic_mode:
                    # Append the index fingertip coordinate (landmark_list[8]) to the history
                    point_history.append(landmark_list[8])
                    pre_processed_point_history_list = pre_process_point_history(debug_image, list(point_history))
                    if len(pre_processed_point_history_list) == (history_length * 2):
                        dynamic_pred = point_history_classifier(pre_processed_point_history_list)
                        finger_gesture_history.append(dynamic_pred)
                        most_common_fg = Counter(finger_gesture_history).most_common(1)[0][0]
                        dynamic_gesture_label = point_history_labels[most_common_fg]
                        # Control peripherals based on recognized dynamic gesture
                        if dynamic_gesture_label.lower() in ["letter s", "s", "letter v", "v"]:
                            control_bluetooth(dynamic_gesture_label)
                        if dynamic_gesture_label.lower() in ["circle", "triangle"]:
                            adjust_brightness(dynamic_gesture_label)
                    else:
                        dynamic_gesture_label = ""
                # -------------------------------
                # Draw results on screen (bounding box, landmarks, and info text)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    static_gesture_label,
                    dynamic_gesture_label,
                )
        else:
            # When no hand is detected, clear history buffers and reset spoken static gesture
            point_history.clear()
            finger_gesture_history.clear()
            last_static_gesture = ""

        # Draw green bubbles if dynamic mode is active
        if dynamic_mode:
            debug_image = draw_point_history(debug_image, list(point_history))

        # Add volume mode status to the display
        if volume_mode:
            cv.putText(debug_image, "Volume Mode", (10, 150),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                      cv.LINE_AA)

        # Add static mode status to the display
        if static_mode:
            cv.putText(debug_image, "Static Mode", (10, 180),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                      cv.LINE_AA)

        # Add dynamic mode status to the display
        if dynamic_mode:
            cv.putText(debug_image, "Dynamic Mode", (10, 210),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                      cv.LINE_AA)

        # Display FPS and mode status on the screen
        mode_info = f"Static: {'ON' if static_mode else 'OFF'}, Dynamic: {'ON' if dynamic_mode else 'OFF'}"
        debug_image = draw_info(debug_image, fps, train_mode, train_number)
        cv.putText(debug_image, mode_info, (10, 140),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv.LINE_AA)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, train_mode, train_number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    # Display training mode indicator
    if train_mode in [1, 2]:
        mode_str = "Logging Key Point" if train_mode == 1 else "Logging Point History"
        cv.putText(image, "TRAIN MODE: " + mode_str, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
        if train_number != -1:
            cv.putText(image, "NUM: " + str(train_number), (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
    return image


def control_bluetooth(gesture_label):
    # Map dynamic gesture to controlling local machine Bluetooth peripheral.
    # For example, if the dynamic gesture is "Letter S"/"S", then turn Bluetooth ON;
    # if it is "Letter V"/"V", then turn Bluetooth OFF.
    if gesture_label.lower() in ["letter s", "s"]:
        # Turn Bluetooth ON (example for Linux)
        os.system("rfkill unblock bluetooth")
    elif gesture_label.lower() in ["letter v", "v"]:
        # Turn Bluetooth OFF (example for Linux)
        os.system("rfkill block bluetooth")


def adjust_brightness(gesture_label):
    # Get the current brightness (returns a list, so take the first element)
    current_brightness = sbc.get_brightness()[0]
    if gesture_label.lower() == "circle":
        # Increase brightness by 10, ensuring it does not exceed 100%
        new_brightness = min(current_brightness + 10, 100)
        sbc.set_brightness(new_brightness)
    elif gesture_label.lower() == "triangle":
        # Decrease brightness by 10, ensuring it does not drop below 0%
        new_brightness = max(current_brightness - 10, 0)
        sbc.set_brightness(new_brightness)


def speak_gesture(gesture_label):
    tts_engine.say(gesture_label)
    tts_engine.runAndWait()


def speak_gesture_async(gesture_label):
    """Add the gesture label to the TTS queue to be spoken asynchronously."""
    tts_queue.put(gesture_label)


def stop_tts_worker():
    """Stops the TTS worker gracefully."""
    tts_queue.put(None)
    tts_thread.join()


if __name__ == '__main__':
    main()
