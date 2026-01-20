import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os

MODEL_PATH = "pose_landmarker_full.task"
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # 얼굴 오른쪽
    (0, 4), (4, 5), (5, 6), (6, 8),  # 얼굴 왼쪽
    (9, 10),  # 입
    (11, 12),  # 어깨
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # 왼팔
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # 오른팔
    (11, 23), (12, 24), (23, 24),  # 몸통
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # 왼다리
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),  # 오른다리
]

def draw_landmarks_on_image(rgb_image, detection_result):
    if detection_result.pose_landmarks:
        h, w, _ = rgb_image.shape
        # black screen 
        annotated_image = np.zeros((h, w, 3), dtype=np.uint8)

        for pose_landmarks in detection_result.pose_landmarks:

            # dot  
            for landmark in pose_landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

            # line 
            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                    start = pose_landmarks[start_idx]
                    end = pose_landmarks[end_idx]
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(annotated_image, start_point, end_point, (255, 0, 0), 2)

            # 좌표 표시
            nose = pose_landmarks[0]
            l_shoulder = pose_landmarks[11]
            r_shoulder = pose_landmarks[12]

            cv2.putText(annotated_image, f"Nose: ({int(nose.x*w)}, {int(nose.y*h)})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_image, f"L_Shoulder: ({l_shoulder.x:.2f}, {l_shoulder.y:.2f})",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_image, f"R_Shoulder: ({r_shoulder.x:.2f}, {r_shoulder.y:.2f})",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated_image
    return rgb_image

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# PoseLandmarker 
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback
)

# webcam 
cap = cv2.VideoCapture(0)
timestamp = 0

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("cannot read webcame")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp += 1
        landmarker.detect_async(mp_image, timestamp)

        # draw 
        if latest_result is not None:
            frame_rgb = draw_landmarks_on_image(frame_rgb, latest_result)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow('MediaPipe Pose', frame_bgr)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
