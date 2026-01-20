import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
import time

POSE_MODEL = "pose_landmarker_full.task"
HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "face_landmarker.task"

if not os.path.exists(POSE_MODEL):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        POSE_MODEL)

if not os.path.exists(HAND_MODEL):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        HAND_MODEL)

if not os.path.exists(FACE_MODEL):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        FACE_MODEL)

# connections 
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arm 
    (11, 23), (12, 24), (23, 24),  # body 
    (23, 25), (25, 27), (24, 26), (26, 28),  # leg
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]

pose_result = None
hand_result = None
face_result = None

def pose_callback(result, output_image, timestamp_ms):
    global pose_result
    pose_result = result

def hand_callback(result, output_image, timestamp_ms):
    global hand_result
    hand_result = result

def face_callback(result, output_image, timestamp_ms):
    global face_result
    face_result = result


current_mode = 2
countdown_active = False
countdown_start = 0
captured_number = None
capture_time = 0

def count_fingers(hand_landmarks, handedness):
    if not hand_landmarks:
        return 0

    fingers = []

    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]

    if handedness == "Right":
        fingers.append(thumb_tip.x < thumb_ip.x)
    else:
        fingers.append(thumb_tip.x > thumb_ip.x)

    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]

    for tip_id, pip_id in zip(tip_ids, pip_ids):
        fingers.append(hand_landmarks[tip_id].y < hand_landmarks[pip_id].y)

    return sum(fingers)

def get_total_fingers():
    total = 0
    if hand_result and hand_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
            handedness = "Right"
            if hand_result.handedness and idx < len(hand_result.handedness):
                handedness = hand_result.handedness[idx][0].category_name
            total += count_fingers(hand_landmarks, handedness)
    return total

def draw_face(canvas, h, w):
    if face_result and face_result.face_landmarks:
        for face_landmarks in face_result.face_landmarks:
            for lm in face_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(canvas, (x, y), 1, (80, 80, 80), -1)
            def draw_contour(indices, color):
                for i in range(len(indices) - 1):
                    if indices[i] < len(face_landmarks) and indices[i+1] < len(face_landmarks):
                        pt1 = face_landmarks[indices[i]]
                        pt2 = face_landmarks[indices[i+1]]
                        cv2.line(canvas, (int(pt1.x*w), int(pt1.y*h)), (int(pt2.x*w), int(pt2.y*h)), color, 1)
            draw_contour(FACE_OVAL, (0, 255, 255))
            draw_contour(LEFT_EYE, (255, 255, 0))
            draw_contour(RIGHT_EYE, (255, 255, 0))
            draw_contour(LIPS_OUTER, (100, 100, 255))

def draw_body_hands(canvas, h, w):
    # Body 
    if pose_result and pose_result.pose_landmarks:
        for pose_landmarks in pose_result.pose_landmarks:
            for i, lm in enumerate(pose_landmarks):
                if i >= 11:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)
            for conn in POSE_CONNECTIONS:
                if conn[0] < len(pose_landmarks) and conn[1] < len(pose_landmarks):
                    pt1 = pose_landmarks[conn[0]]
                    pt2 = pose_landmarks[conn[1]]
                    cv2.line(canvas, (int(pt1.x*w), int(pt1.y*h)), (int(pt2.x*w), int(pt2.y*h)), (0, 200, 0), 2)
    # Hands
    if hand_result and hand_result.hand_landmarks:
        colors = [(255, 100, 100), (100, 100, 255)]
        for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
            color = colors[idx % 2]
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(canvas, (x, y), 3, color, -1)
            for conn in HAND_CONNECTIONS:
                if conn[0] < len(hand_landmarks) and conn[1] < len(hand_landmarks):
                    pt1 = hand_landmarks[conn[0]]
                    pt2 = hand_landmarks[conn[1]]
                    cv2.line(canvas, (int(pt1.x*w), int(pt1.y*h)), (int(pt2.x*w), int(pt2.y*h)), color, 2)

def draw_all(h, w):
    global current_mode, countdown_active, countdown_start, captured_number, capture_time
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if current_mode == 1:
        draw_face(canvas, h, w)
        label = "Mode 1: Face"
    else:
        draw_body_hands(canvas, h, w)
        label = "Mode 2: Body + Hands"

    cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "Press 1/2: switch | SPACE: capture", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # finger 
    if current_mode == 2:
        fingers = get_total_fingers()
        cv2.putText(canvas, f"Fingers: {fingers}", (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return canvas

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=1, min_pose_detection_confidence=0.5, min_tracking_confidence=0.5,
    result_callback=pose_callback)

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5,
    result_callback=hand_callback)

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_faces=1, min_face_detection_confidence=0.5, min_tracking_confidence=0.5,
    result_callback=face_callback)

# webcam 
cap = cv2.VideoCapture(0)
timestamp = 0

with vision.PoseLandmarker.create_from_options(pose_options) as pose_lm, \
     vision.HandLandmarker.create_from_options(hand_options) as hand_lm, \
     vision.FaceLandmarker.create_from_options(face_options) as face_lm:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp += 1
        pose_lm.detect_async(mp_image, timestamp)
        hand_lm.detect_async(mp_image, timestamp)
        face_lm.detect_async(mp_image, timestamp)

        skeleton = draw_all(h, w)
        cv2.imshow('Skeleton', skeleton)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_mode = 1
        elif key == ord('2'):
            current_mode = 2
        elif key == ord(' ') and not countdown_active:
            countdown_active = True
            countdown_start = time.time()
            captured_number = None

cap.release()
cv2.destroyAllWindows()

