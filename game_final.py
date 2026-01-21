import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
import time
import requests  # 서버 통신용


# === 서버 설정 ===
BRAIN_SERVER_URL = "http://127.0.0.1:8000/predict"

# === 모델 파일 준비 ===
POSE_MODEL = "pose_landmarker_full.task"
HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "face_landmarker.task"

def download_model(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

download_model("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task", POSE_MODEL)
download_model("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", HAND_MODEL)
download_model("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", FACE_MODEL)

# === 연결 선 정의 ===
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # 팔
    (11, 23), (12, 24), (23, 24),  # 몸통
    (23, 25), (25, 27), (24, 26), (26, 28),  # 다리
    (28, 30), (27, 29), (30, 32), (29, 31)   # 발
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# 얼굴 주요 부위 인덱스 (MediaPipe Face Mesh 기준)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
EYEBROWS = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 336, 296, 334, 293, 300, 276, 283, 282, 295, 285] # 눈썹 추가 (감정 표현에 중요)

# === 전역 변수 ===
pose_result = None
hand_result = None
face_result = None

# === 콜백 함수 ===
def pose_callback(result, output_image, timestamp_ms):
    global pose_result
    pose_result = result

def hand_callback(result, output_image, timestamp_ms):
    global hand_result
    hand_result = result

def face_callback(result, output_image, timestamp_ms):
    global face_result
    face_result = result

# === 유틸 함수 (그리기 & 손가락 계산) ===

def count_fingers(hand_landmarks, handedness):
    if not hand_landmarks: return 0
    fingers = []
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    if handedness == "Right": fingers.append(thumb_tip.x > thumb_ip.x)
    else: fingers.append(thumb_tip.x < thumb_ip.x)
    for tip_id, pip_id in zip([8, 12, 16, 20], [6, 10, 14, 18]):
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

def draw_face_landmarks(canvas, h, w):
    """얼굴 주요 특징(눈, 코, 입, 눈썹) 그리기"""
    if face_result and face_result.face_landmarks:
        for face_landmarks in face_result.face_landmarks:
            # 윤곽선 그리기 함수
            def draw_contour(indices, color, thickness=1):
                for i in range(len(indices) - 1):
                    idx1, idx2 = indices[i], indices[i+1]
                    if idx1 < len(face_landmarks) and idx2 < len(face_landmarks):
                        pt1 = face_landmarks[idx1]
                        pt2 = face_landmarks[idx2]
                        x1, y1 = int(pt1.x * w), int(pt1.y * h)
                        x2, y2 = int(pt2.x * w), int(pt2.y * h)
                        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)
            
            # 1. 주요 부위 그리기 (선명하게)
            draw_contour(FACE_OVAL, (100, 100, 100), 1) # 얼굴 윤곽 (회색)
            draw_contour(LEFT_EYE, (0, 255, 255), 2)    # 눈 (노란색, 두껍게)
            draw_contour(RIGHT_EYE, (0, 255, 255), 2)
            draw_contour(LIPS_OUTER, (0, 0, 255), 2)    # 입 (빨간색, 두껍게)
            draw_contour(EYEBROWS, (255, 255, 255), 2)  # 눈썹 (흰색, 두껍게 - 감정 표현 핵심)

def draw_body_and_hands(canvas, h, w):
    """몸과 손 그리기"""
    # Body
    if pose_result and pose_result.pose_landmarks:
        for landmarks in pose_result.pose_landmarks:
            for conn in POSE_CONNECTIONS:
                if conn[0] < len(landmarks) and conn[1] < len(landmarks):
                    pt1, pt2 = landmarks[conn[0]], landmarks[conn[1]]
                    if pt1.visibility > 0.5 and pt2.visibility > 0.5:
                        x1, y1 = int(pt1.x * w), int(pt1.y * h)
                        x2, y2 = int(pt2.x * w), int(pt2.y * h)
                        cv2.line(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2) # 초록색
            # 주요 관절 점
            for i, lm in enumerate(landmarks):
                if i >= 11 and lm.visibility > 0.5: # 얼굴 제외한 몸통부터
                    cv2.circle(canvas, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

    # Hands
    if hand_result and hand_result.hand_landmarks:
        colors = [(255, 100, 100), (100, 100, 255)]
        for idx, landmarks in enumerate(hand_result.hand_landmarks):
            color = colors[idx % 2]
            for conn in HAND_CONNECTIONS:
                pt1, pt2 = landmarks[conn[0]], landmarks[conn[1]]
                cv2.line(canvas, (int(pt1.x*w), int(pt1.y*h)), (int(pt2.x*w), int(pt2.y*h)), color, 2)
            for lm in landmarks:
                cv2.circle(canvas, (int(lm.x*w), int(lm.y*h)), 3, color, -1)

def draw_all_features(h, w):
    """얼굴 + 몸 + 손 모두 그리기 (Retina View용)"""
    canvas = np.zeros((h, w, 3), dtype=np.uint8) # 검은 배경
    
    # 3가지 모두 그림
    draw_face_landmarks(canvas, h, w)
    draw_body_and_hands(canvas, h, w)
    
    return cv2.flip(canvas, 1)

# === 그리드 생성 및 서버 전송 ===

def create_grid_image(frames, grid_size=(3, 3)):
    """9장 프레임을 하나의 그리드 이미지로 병합"""
    if not frames: return None
    rows, cols = grid_size
    target_count = rows * cols
    
    # 프레임 부족 시 마지막 프레임 복제
    if len(frames) < target_count:
        frames += [frames[-1]] * (target_count - len(frames))
    
    # 균등 샘플링
    indices = np.linspace(0, len(frames) - 1, target_count, dtype=int)
    selected_frames = [frames[i] for i in indices]
    
    # 리사이즈 및 병합
    h, w, _ = selected_frames[0].shape
    small_h, small_w = h // 2, w // 2 
    resized_frames = [cv2.resize(f, (small_w, small_h)) for f in selected_frames]
    
    grid_rows = []
    for r in range(rows):
        row_imgs = resized_frames[r*cols : (r+1)*cols]
        grid_rows.append(np.hstack(row_imgs))
    return np.vstack(grid_rows)

def send_to_brain(skeleton_image, category, candidates):
    """서버 전송"""
    print(">>> Sending Grid to Brain (A6000)...")
    _, img_encoded = cv2.imencode('.jpg', skeleton_image)
    files = {'file': ('skeleton.jpg', img_encoded.tobytes(), 'image/jpeg')}
    data = {'category': category, 'candidates': ", ".join(candidates)}
    
    try:
        response = requests.post(BRAIN_SERVER_URL, files=files, data=data, timeout=30)
        if response.status_code == 200:
            result = response.json()['result']
            print(f">>> Brain Answer: {result}")
            return result
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        print(f">>> Fail: {e}")
        return "Conn Fail"

# === MediaPipe 설정 ===
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=3, min_pose_detection_confidence=0.5, min_tracking_confidence=0.5,
    result_callback=pose_callback)

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=6, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5,
    result_callback=hand_callback)

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_faces=3, min_face_detection_confidence=0.5, min_tracking_confidence=0.5,
    result_callback=face_callback)

# === 메인 실행 함수 1: 카테고리 선택 ===
def run_webcam():
    cap = cv2.VideoCapture(0)
    returned_finger_count = None

    with vision.PoseLandmarker.create_from_options(pose_options) as pose_lm, \
         vision.HandLandmarker.create_from_options(hand_options) as hand_lm, \
         vision.FaceLandmarker.create_from_options(face_options) as face_lm:

        while cap.isOpened():
            success, frame = cap.read()
            if not success: continue

            h, w, _ = frame.shape
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = int(time.time() * 1000)
            
            pose_lm.detect_async(mp_image, ts)
            hand_lm.detect_async(mp_image, ts)
            face_lm.detect_async(mp_image, ts) # 얼굴 인식도 항상 실행

            # 선택 단계에서도 얼굴까지 다 보여줌 (draw_all_features 사용)
            skeleton = draw_all_features(h, w)
            
            fingers = get_total_fingers()
            cv2.putText(skeleton, "Select Category (Fingers)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.putText(skeleton, f"Count: {fingers}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
            cv2.putText(skeleton, "Press SPACE to Confirm", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            cv2.imshow('Retina View', skeleton)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'): break
            elif key == ord(' '):
                # Config 저장
                config_path = os.path.join(os.path.dirname(__file__), 'config.py')
                try:
                    with open(config_path, 'r', encoding='utf-8') as f: lines = f.readlines()
                    for i, line in enumerate(lines):
                        if line.startswith('LATEST_FINGER_COUNT'):
                            lines[i] = f'LATEST_FINGER_COUNT = {fingers}\n'
                            break
                    with open(config_path, 'w', encoding='utf-8') as f: f.writelines(lines)
                except: pass
                returned_finger_count = fingers
                break

    cap.release()
    cv2.destroyAllWindows()
    return returned_finger_count

# === 메인 실행 함수 2: 제스처 인식 (그리드 + 얼굴 포함) ===
def run_gesture_recognition(category, candidates):
    cap = cv2.VideoCapture(0)
    
    recorded_frames = []
    is_recording = False
    
    with vision.PoseLandmarker.create_from_options(pose_options) as pose_lm, \
         vision.HandLandmarker.create_from_options(hand_options) as hand_lm, \
         vision.FaceLandmarker.create_from_options(face_options) as face_lm:
        
        predicted_text = "Ready (SPACE to Record)"
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            h, w, _ = frame.shape
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = int(time.time() * 1000)
            
            pose_lm.detect_async(mp_image, ts)
            hand_lm.detect_async(mp_image, ts)
            face_lm.detect_async(mp_image, ts)
            
            # 여기서 얼굴+몸+손이 모두 그려진 프레임을 가져옴
            current_skeleton = draw_all_features(h, w)
            
            # 녹화 로직
            if is_recording:
                recorded_frames.append(current_skeleton.copy())
                cv2.rectangle(current_skeleton, (0,0), (w,h), (0,0,255), 5)
                cv2.putText(current_skeleton, f"REC: {len(recorded_frames)}", (w//2-50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            # UI 표시
            display_img = current_skeleton.copy()
            cv2.putText(display_img, f"Category: {category}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(display_img, f"AI: {predicted_text}", (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            
            cv2.imshow('Retina View (Brain Input)', display_img)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord(' '):
                if not is_recording:
                    is_recording = True
                    recorded_frames = []
                    predicted_text = "Recording..."
                else:
                    is_recording = False
                    if len(recorded_frames) > 5:
                        cv2.putText(display_img, "Processing...", (w//2-50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.imshow('Retina View (Brain Input)', display_img)
                        cv2.waitKey(1)
                        
                        # 얼굴까지 포함된 그리드 생성!
                        grid_image = create_grid_image(recorded_frames)
                        predicted_text = send_to_brain(grid_image, category, candidates)
                    else:
                        predicted_text = "Too short!"
            elif key == ord('q'):
                break

 # === 메인 실행 로직: 무한 반복 구조 ===
if __name__ == "__main__":
    # 게임에 사용할 후보군 예시 (필요에 따라 수정)
    GAMES = {
        "emotion": ["happy", "sad", "angry", "surprised"],
        "action": ["running", "jumping", "sleeping", "dancing"]
    }
    
    print("=== Speed Game Start! (Press 'q' to Exit) ===")
    
    try:
        while True:
            # 1. 숫자 인식 (카테고리/난이도/모드 선택 단계)
            print("\n[Step 1] Select Category with your fingers...")
            selected_count = run_webcam()
            
            # 'q'를 눌러 종료한 경우 루프 탈출
            if selected_count is None:
                break
                
            # 2. 제스처 인식 (동작 수행 단계)
            # 여기서는 예시로 손가락 개수에 따라 카테고리를 동적으로 정하거나
            # 특정 후보군을 전달할 수 있습니다.
            category_name = f"Mode_{selected_count}"
            candidates_list = GAMES.get("action") # 혹은 selected_count에 따라 분기
            
            print(f"\n[Step 2] Recording Gesture for Category: {category_name}")
            run_gesture_recognition(category_name, candidates_list)
            
            # run_gesture_recognition 안에서 'q'를 누르면 다음 루프로 넘어가거나 
            # 프로그램 전체를 종료하고 싶다면 별도의 상태 체크를 추가할 수 있습니다.
            
    except KeyboardInterrupt:
        print("\nGame Terminated.")
    finally:
        cv2.destroyAllWindows()
