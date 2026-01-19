import cv2
import numpy as np
import os
import time
import math
from datetime import datetime
import pickle

from collections import deque
import sys
import torch
import pandas as pd
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def extract_features_legacy(gray, prev_gray, prev_pts):
    feat = {}
    
    # 1. Mean Intensity
    feat['meanIntensity'] = np.mean(gray)

    # 2. Edge Density (Sobel)
    # MATLAB의 edge(gray, 'sobel')은 이진화된 에지 맵을 반환하므로 유사하게 구현
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    edge_map = (mag > 50).astype(np.uint8) # 임계값 적용
    feat['edgeDensity'] = np.mean(edge_map)

    # 3. Frame Difference
    if prev_gray is None:
        feat['frameDiff'] = 0.0
    else:
        diff = cv2.absdiff(gray, prev_gray)
        feat['frameDiff'] = np.mean(diff)

    # 4. Optical Flow (Simple Dense Flow for Magnitude)
    if prev_gray is None:
        feat['flowMag'] = np.nan
    else:
        # 간단한 Dense Optical Flow (Farneback) 사용하여 크기 계산
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        feat['flowMag'] = np.mean(mag)

    return feat


def webcam_02hz_gray(output_dir='run_webcam', duration_sec=30, device_index=0):
    # 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 웹캠 연결
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print(f"Error: 웹캠(Index {device_index})을 열 수 없습니다.")
        return

    interval_sec = 5.0  # 0.2Hz = 1 frame per 5 seconds
    total_frames = max(1, int(math.ceil(duration_sec / interval_sec)))

    print(f"Recording started for {duration_sec}s at 0.2Hz...")

    # 데이터 저장용 리스트
    history = {
        'meanIntensity': [],
        'edgeDensity': [],
        'frameDiff': [],
        'flowMag': [],
        'timestamps': [],
        'fileNames': []
    }

    prev_gray = None
    t0 = time.time()

    try:
        for k in range(1, total_frames + 1):
            start_loop = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            feat = extract_features_legacy(gray, prev_gray, None)
            
            ts_now = datetime.now()
            ts_str = ts_now.strftime('%H:%M:%S')
            ts_full = ts_now.strftime('%Y%m%d_%H%M%S_%f')[:-3]
            
            fname = f"gray_{k:04d}_{ts_str.replace(':', '')}.png"
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, gray)

            history['meanIntensity'].append(feat['meanIntensity'])
            history['edgeDensity'].append(feat['edgeDensity'])
            history['frameDiff'].append(feat['frameDiff'])
            history['flowMag'].append(feat['flowMag'])
            history['timestamps'].append(ts_full)
            history['fileNames'].append(fpath)

            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            combined_img = np.hstack((frame, gray_3ch))
            
            # 정보 텍스트 삽입
            info_text = f"Time: {ts_str} | Frame: {k}/{total_frames}"
            feat_text = f"Int: {feat['meanIntensity']:.1f}, Edge: {feat['edgeDensity']:.3f}, Diff: {feat['frameDiff']:.2f}"
            cv2.putText(combined_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined_img, feat_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('STEP3: Webcam(0.2Hz) -> Gray -> Feature', combined_img)

            prev_gray = gray.copy()

            elapsed = time.time() - t0
            target_time = k * interval_sec
            sleep_time = max(0.01, target_time - elapsed)
            
            if cv2.waitKey(int(sleep_time * 1000)) & 0xFF == ord('q'):
                break

    finally:
        # 종료 및 메타데이터 저장
        cap.release()
        cv2.destroyAllWindows()
        
        meta = {
            'outputDir': output_dir,
            'durationSec': duration_sec,
            'deviceIndex': device_index,
            'history': history
        }
        
        with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(meta, f)
        
        print(f"Saved metadata: {os.path.join(output_dir, 'metadata.pkl')}")


def realtime_digit_recognition(
    model_path=None,
    hand_model_path=None,
    device_index=0,
    output_dir='captures_1hz',
):
    # 모델 모듈 경로 추가
    model_dir = os.path.join(
        os.path.dirname(__file__),
        "Real-Time-Sign-Language-Recognition",
        "Sign Language Recognition",
    )
    if model_dir not in sys.path:
        sys.path.append(model_dir)
    from CNNModel import CNNModel

    # 손 랜드마크 이름(학습 데이터 컬럼과 동일)
    hand_landmark_names = [
        "WRIST",
        "THUMB_CMC",
        "THUMB_MCP",
        "THUMB_IP",
        "THUMB_TIP",
        "INDEX_FINGER_MCP",
        "INDEX_FINGER_PIP",
        "INDEX_FINGER_DIP",
        "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP",
        "MIDDLE_FINGER_PIP",
        "MIDDLE_FINGER_DIP",
        "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP",
        "RING_FINGER_PIP",
        "RING_FINGER_DIP",
        "RING_FINGER_TIP",
        "PINKY_MCP",
        "PINKY_PIP",
        "PINKY_DIP",
        "PINKY_TIP",
    ]

    # 모델 로드 (숫자 1-9: 9 클래스)
    if model_path is None:
        model_path = os.path.join(model_dir, "CNN_model_number_custom.pth")
    if hand_model_path is None:
        hand_model_path = os.path.join(model_dir, "hand_landmarker.task")
    model = CNNModel(num_classes=9)
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "CNN_model_number_SIBI.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    classes = {str(i + 1): i for i in range(9)}
    pred_history = deque(maxlen=7)
    min_confidence = 0.6

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print(f"Error: 웹캠(Index {device_index})을 열 수 없습니다.")
        return

    base_options = python.BaseOptions(model_asset_path=hand_model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    hand_detector = vision.HandLandmarker.create_from_options(options)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    last_save_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=frame_rgb)
            result = hand_detector.detect_for_video(mp_image, int(time.time() * 1000))

            display_text = "DIGIT: -"
            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                x_coords, y_coords, z_coords = [], [], []
                for lm in hand_landmarks:
                    x_coords.append(lm.x)
                    y_coords.append(lm.y)
                    z_coords.append(lm.z)
                x_min, y_min, z_min = min(x_coords), min(y_coords), min(z_coords)

                data = {}
                for i, lm in enumerate(hand_landmarks):
                    name = hand_landmark_names[i] if i < len(hand_landmark_names) else f"LM_{i}"
                    data[f"{name}_x"] = lm.x - x_min
                    data[f"{name}_y"] = lm.y - y_min
                    data[f"{name}_z"] = lm.z - z_min

                coordinates = pd.DataFrame([data])
                coordinates = np.reshape(coordinates.values, (coordinates.shape[0], 63, 1))
                coordinates = torch.from_numpy(coordinates).float()

                with torch.no_grad():
                    outputs = model(coordinates)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    pred = int(np.argmax(probs))
                    conf = float(probs[pred])

                pred_history.append(pred if conf >= min_confidence else None)
                valid_preds = [p for p in pred_history if p is not None]
                if valid_preds:
                    pred = max(set(valid_preds), key=valid_preds.count)
                    predicted_character = next((k for k, v in classes.items() if v == pred), None)
                    if predicted_character is not None:
                        display_text = f"DIGIT: {predicted_character}"

            # 1초마다 이미지 저장
            now = time.time()
            if now - last_save_time >= 1.0:
                ts = time.strftime("%Y%m%d_%H%M%S")
                fpath = os.path.join(output_dir, f"frame_{ts}.png")
                cv2.imwrite(fpath, frame)
                last_save_time = now

            # 원본 영상 위에 텍스트 표시 (요청에 따라 비활성화)
            # cv2.putText(frame, display_text, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
            # cv2.imshow('digit', frame)

            # 텍스트만 표시하는 빈 캔버스
            blank_canvas = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(blank_canvas, display_text, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
            cv2.imshow('digit', blank_canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 실행 예시: 'test' 폴더에 10초 동안 촬영

    # 기존 촬영 함수
    # webcam_02hz_gray('test', 30, 0) # 파이썬은 대개 첫번째 캠 인덱스가 0입니다.

    # 숫자 인식 실행
    realtime_digit_recognition()
