import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
from PIL import ImageFont, ImageDraw, Image


WIDTH, HEIGHT = 1920, 1080
BOX_WIDTH = 200
STABLE_TIME = 0.5

# map
CONSONANTS = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

# right box
NUM_RIGHT = 6
R_BOX_H = HEIGHT // NUM_RIGHT 
right_persistent = [0] * NUM_RIGHT
right_candidate = [0] * NUM_RIGHT
right_start_times = [0.0] * NUM_RIGHT
right_processed = [False] * NUM_RIGHT

# left box
NUM_LEFT = 2
L_BOX_H = 135 
left_persistent = [0] * NUM_LEFT
left_candidate = [0] * NUM_LEFT
left_start_times = [0.0] * NUM_LEFT

# 모델 경로
POSE_MODEL = "pose_landmarker_full.task"
HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "face_landmarker.task"

# === 2. 인덱스 및 연결선 정의 (요청 스타일 반영) ===
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]

POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# === 3. 헬퍼 함수 ===
def put_korean_text(img, text, pos, font_size, color):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    fonts = ["/System/Library/Fonts/Supplemental/AppleGothic.ttf", "C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]
    font = None
    for f_path in fonts:
        try:
            font = ImageFont.truetype(f_path, font_size)
            break
        except: continue
    if font is None: font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return np.array(img_pil)

def draw_contour(canvas, landmarks, indices, color, thickness, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    for i in range(len(points) - 1):
        cv2.line(canvas, points[i], points[i+1], color, thickness)

# === 4. 전역 결과 및 콜백 ===
latest_pose, latest_hand, latest_face = None, None, None

def pose_callback(result, output_image, timestamp_ms): global latest_pose; latest_pose = result
def hand_callback(result, output_image, timestamp_ms): global latest_hand; latest_hand = result
def face_callback(result, output_image, timestamp_ms): global latest_face; latest_face = result

# === 5. Task Landmarker 옵션 설정 ===
base_options = lambda path: python.BaseOptions(model_asset_path=path)
pose_opts = vision.PoseLandmarkerOptions(base_options=base_options(POSE_MODEL), running_mode=vision.RunningMode.LIVE_STREAM, num_poses=2, result_callback=pose_callback)
hand_opts = vision.HandLandmarkerOptions(base_options=base_options(HAND_MODEL), running_mode=vision.RunningMode.LIVE_STREAM, num_hands=4, result_callback=hand_callback)
face_opts = vision.FaceLandmarkerOptions(base_options=base_options(FACE_MODEL), running_mode=vision.RunningMode.LIVE_STREAM, num_faces=2, min_face_detection_confidence=0.2, result_callback=face_callback)

# === 6. 메인 루프 ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

with vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm, \
     vision.HandLandmarker.create_from_options(hand_opts) as hand_lm, \
     vision.FaceLandmarker.create_from_options(face_opts) as face_lm:

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        curr_time = time.time()
        timestamp_ms = int(curr_time * 1000)

        pose_lm.detect_async(mp_image, timestamp_ms)
        hand_lm.detect_async(mp_image, timestamp_ms)
        face_lm.detect_async(mp_image, timestamp_ms)

        black_board = np.zeros((h, w, 3), dtype=np.uint8)

        # --- [로직] 박스 카운팅 ---
        curr_r_counts = [0] * NUM_RIGHT
        curr_l_counts = [0] * NUM_LEFT
        if latest_hand and latest_hand.hand_landmarks:
            for hand_lms in latest_hand.hand_landmarks:
                for tip_idx in [4, 8, 12, 16, 20]:
                    tip = hand_lms[tip_idx]
                    tx, ty = int(tip.x * w), int(tip.y * h)
                    if tx > (w - BOX_WIDTH):
                        idx = ty // R_BOX_H
                        if 0 <= idx < NUM_RIGHT: curr_r_counts[idx] += 1
                    elif tx < BOX_WIDTH:
                        idx = ty // L_BOX_H
                        if 0 <= idx < NUM_LEFT: curr_l_counts[idx] += 1

        # 우측 박스 (누적 합산)
        for i in range(NUM_RIGHT):
            det = curr_r_counts[i]
            if det > 0:
                if det == right_candidate[i]:
                    if not right_processed[i] and (curr_time - right_start_times[i] >= STABLE_TIME):
                        right_persistent[i] = ((right_persistent[i] + det) - 1) % 14 + 1
                        right_processed[i] = True
                else:
                    right_candidate[i], right_start_times[i], right_processed[i] = det, curr_time, False
            else:
                right_candidate[i], right_start_times[i], right_processed[i] = 0, 0, False

        # 좌측 박스 (교체)
        for i in range(NUM_LEFT):
            det = curr_l_counts[i]
            if det > 0:
                if det == left_candidate[i]:
                    if curr_time - left_start_times[i] >= STABLE_TIME: left_persistent[i] = det
                else:
                    left_candidate[i], left_start_times[i] = det, curr_time
            else:
                left_candidate[i], left_start_times[i] = 0, 0

        # --- [시각화 1] 스켈레톤 (요청 스타일 반영) ---
        
        # 1. 얼굴 (Face)
        if latest_face and latest_face.face_landmarks:
            for landmarks in latest_face.face_landmarks:
                draw_contour(black_board, landmarks, FACE_OVAL, (100, 100, 100), 1, w, h)   # 윤곽: 회색
                draw_contour(black_board, landmarks, LEFT_EYE, (0, 255, 255), 2, w, h)      # 눈: 노란색
                draw_contour(black_board, landmarks, RIGHT_EYE, (0, 255, 255), 2, w, h)
                draw_contour(black_board, landmarks, LIPS_OUTER, (0, 0, 255), 2, w, h)     # 입술 밖: 빨간색
                draw_contour(black_board, landmarks, LIPS_INNER, (0, 0, 255), 1, w, h)     # 입술 안: 빨간색
                draw_contour(black_board, landmarks, LEFT_EYEBROW, (255, 255, 255), 2, w, h) # 눈썹: 흰색
                draw_contour(black_board, landmarks, RIGHT_EYEBROW, (255, 255, 255), 2, w, h)

        # 2. 몸통 (Pose) - 얼굴 제외
        if latest_pose and latest_pose.pose_landmarks:
            for landmarks in latest_pose.pose_landmarks:
                for conn in POSE_CONNECTIONS:
                    if conn[0] >= 11 and conn[1] >= 11: # 얼굴 제외
                        p1, p2 = landmarks[conn[0]], landmarks[conn[1]]
                        if p1.presence > 0.5 and p2.presence > 0.5:
                            cv2.line(black_board, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 200, 0), 2) # 초록색 선
                for i, lm in enumerate(landmarks):
                    if i >= 11 and lm.presence > 0.5:
                        cv2.circle(black_board, (int(lm.x*w), int(lm.y*h)), 4, (0, 255, 0), -1) # 초록색 점

        # 3. 손 (Hands)
        if latest_hand and latest_hand.hand_landmarks:
            for i, landmarks in enumerate(latest_hand.hand_landmarks):
                lbl = latest_hand.handedness[i][0].category_name
                color = (100, 100, 255) if lbl == "Right" else (255, 100, 100) # 오른손 파랑, 왼손 빨강
                for conn in HAND_CONNECTIONS:
                    p1, p2 = landmarks[conn[0]], landmarks[conn[1]]
                    cv2.line(black_board, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), color, 2)
                for lm in landmarks:
                    cv2.circle(black_board, (int(lm.x*w), int(lm.y*h)), 3, color, -1)

        # --- [시각화 2] 박스 UI ---
        for i in range(NUM_RIGHT):
            tl, br = (w - BOX_WIDTH, i * R_BOX_H), (w, (i + 1) * R_BOX_H)
            is_wait = (right_candidate[i] > 0 and not right_processed[i])
            box_clr = (0, 255, 255) if is_wait else (0, 0, 255)
            cv2.rectangle(black_board, tl, br, box_clr, 2)
            if right_persistent[i] > 0:
                black_board = put_korean_text(black_board, CONSONANTS[right_persistent[i]-1], (tl[0]+60, tl[1]+40), 100, (255, 255, 255))

        for i in range(NUM_LEFT):
            tl, br = (0, i * L_BOX_H), (BOX_WIDTH, (i + 1) * L_BOX_H)
            is_wait = (left_candidate[i] > 0 and left_candidate[i] != left_persistent[i])
            box_clr = (0, 255, 255) if is_wait else (0, 0, 255)
            cv2.rectangle(black_board, tl, br, box_clr, 2)
            cv2.putText(black_board, str(left_persistent[i]), (tl[0]+70, tl[1]+90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        cv2.imshow('Retina Skeleton & Control System', black_board)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()