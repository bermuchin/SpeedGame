import cv2
import numpy as np
import os
import time
from datetime import datetime
import pickle

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

def webcam_1hz_gray(output_dir='run_webcam', duration_sec=30, device_index=0):
    # 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 웹캠 연결
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print(f"Error: 웹캠(Index {device_index})을 열 수 없습니다.")
        return

    print(f"Recording started for {duration_sec}s at 1Hz...")

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
        for k in range(1, duration_sec + 1):
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
            info_text = f"Time: {ts_str} | Frame: {k}/{duration_sec}"
            feat_text = f"Int: {feat['meanIntensity']:.1f}, Edge: {feat['edgeDensity']:.3f}, Diff: {feat['frameDiff']:.2f}"
            cv2.putText(combined_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined_img, feat_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('STEP3: Webcam(1Hz) -> Gray -> Feature', combined_img)

            prev_gray = gray.copy()

            elapsed = time.time() - t0
            target_time = k * 1.0
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

if __name__ == "__main__":
    # 실행 예시: 'test' 폴더에 10초 동안 촬영
    webcam_1hz_gray('test', 10, 0) # 파이썬은 대개 첫번째 캠 인덱스가 0입니다.