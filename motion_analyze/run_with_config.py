import sys
import os
import importlib.util

# pose_webcam 모듈을 import하여 실행
def run_pose_webcam():
    """pose_webcam.py의 run_webcam 함수를 실행하고 finger count를 반환"""
    # pose_webcam 모듈을 동적으로 import
    pose_webcam_path = os.path.join(os.path.dirname(__file__), 'pose_webcam.py')
    spec = importlib.util.spec_from_file_location("pose_webcam", pose_webcam_path)
    pose_webcam = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pose_webcam)

    print("=" * 50)
    print("Starting pose_webcam.py...")
    print("Press SPACE to capture finger count and exit")
    print("Press 'q' to quit without capturing")
    print("=" * 50)

    # pose_webcam의 run_webcam 함수 실행
    finger_count = pose_webcam.run_webcam()

    return finger_count

# config.py에서 저장된 파라미터 출력
def print_config(finger_count):
    """config.py에서 LATEST_FINGER_COUNT를 읽어서 출력"""
    # config 모듈을 동적으로 다시 로드하여 최신 값을 가져옴
    config_path = os.path.join(os.path.dirname(__file__), 'config.py')

    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    print("\n" + "=" * 50)
    print("Webcam Result:")
    print("=" * 50)
    print(f"Returned finger count: {finger_count}")

    # finger count에 해당하는 WORD_DICT 카테고리 찾기
    if finger_count is not None:
        # WORD_DICT의 키를 리스트로 변환 (순서 보장)
        categories = list(config.WORD_DICT.keys())
        # finger count를 인덱스로 사용 (1~5 -> 0~4 인덱스)
        category_index = finger_count - 1

        if 0 <= category_index < len(categories):
            selected_category = categories[category_index]
            selected_words = config.WORD_DICT[selected_category]

            print(f"Selected category: {selected_category}")
            print(f"Candidate words: {selected_words}")

            # finger count에 해당하는 System Prompt Template 출력
            if finger_count in config.SYSTEM_PROMPT_TEMPLATES:
                # candidates를 포맷팅하여 템플릿에 삽입
                prompt_template = config.SYSTEM_PROMPT_TEMPLATES[finger_count].format(
                    candidates=", ".join(selected_words)
                )
                print("\n" + "=" * 50)
                print("Selected System Prompt Template:")
                print("=" * 50)
                print(prompt_template)
                print("=" * 50)

                # config.py에 LATEST_SYSTEM_PROMPT 저장
                with open(config_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # LATEST_SYSTEM_PROMPT 값 업데이트
                for i, line in enumerate(lines):
                    if line.startswith('LATEST_SYSTEM_PROMPT'):
                        # 멀티라인 문자열을 저장하기 위해 repr 사용
                        lines[i] = f'LATEST_SYSTEM_PROMPT = {repr(prompt_template)}\n'
                        break

                # 파일에 다시 쓰기
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

                print(f"System prompt saved to config.py")

                return prompt_template, selected_category, selected_words
            else:
                print(f"Warning: No system prompt template for finger count {finger_count}")
        else:
            print(f"Warning: finger count {finger_count} is out of range (1-{len(categories)})")

    return None, None, None

if __name__ == "__main__":
    
    while True:
        # 1. pose_webcam 실행하고 finger count 받기
        finger_count = run_pose_webcam()

        # 2. pose_webcam 종료 후 결과 및 config.py의 파라미터 출력
        print_config(finger_count)

        # 3. config 받아서 prompt로 LLM에 전달
        
        # 4. LLM의 결과를 Print

        # 5. 비장의 무기
        # run_pose_webcam()
