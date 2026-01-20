# agent.py
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import collections
import config
from prompts import PROMPT_TEMPLATES, DEFAULT_PROMPT_TEMPLATE

CATEGORY_SWITCH = 5 # 카테고리 !!! (수정)


class MotionAgent:
    def __init__(self):
        print(">>> [Agent] A6000 Brain 초기화 중...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(config.MODEL_ID)
        self.check_health()

    def check_health(self):
        """시스템 상태 진단"""
        print("\n" + "="*30)
        print("🩺 Agent Health Check")
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {torch.cuda.get_device_name(0)} (VRAM: {vram:.2f} GB)")
        print(f"✅ Model Device: {self.model.device}")
        print("="*30 + "\n")

    def preprocess_image(self, image_np):
        """중앙 크롭 및 리사이징 (배경 제거 효과)"""
        h, w, _ = image_np.shape
        min_dim = min(h, w)
        # Center Crop
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        cropped = image_np[start_y:start_y+min_dim, start_x:start_x+min_dim]
        # Resize to 480x480 (Speed optimization)
        return Image.fromarray(cropped).resize((480, 480))

    def infer_stream(self, image, selected_categories, frame_buffer):
        """스트리밍 추론 메인 로직"""
        # 1. 예외 처리
        if image is None:
            return "카메라 대기 중...", "...", frame_buffer

        # 2. 버퍼 관리 (State handling)
        if frame_buffer is None:
            frame_buffer = collections.deque(maxlen=6)
        
        # 전처리(Crop) 후 버퍼에 추가
        processed_img = self.preprocess_image(image)
        frame_buffer.append(processed_img)

        # 버퍼가 충분히 차지 않았으면 대기 (Latency 조절)
        if len(frame_buffer) < 3:
            return "정보 수집 중...", "...", frame_buffer

        # 3. 카테고리 준비 (숫자 스위치 사용)
        switch_value = CATEGORY_SWITCH
        if switch_value == 1:
            selected_category = "감정/상태"
        elif switch_value == 2:
            selected_category = "동물"
        elif switch_value == 3:
            selected_category = "행동"
        elif switch_value == 4:
            selected_category = "스포츠"
        elif switch_value == 5:
            selected_category = "자연"
        else:
            selected_category = "동물"

        candidates = config.WORD_DICT.get(selected_category, [])

        # 4. 프롬프트 구성
        prompt_template = PROMPT_TEMPLATES.get(selected_category, DEFAULT_PROMPT_TEMPLATE)
        prompt_text = prompt_template.format(candidates=", ".join(candidates))
        
        # 5. 입력 메시지 구성 (Multi-frame)
        content_list = []
        # 버퍼의 모든 이미지를 입력으로 사용
        for img in frame_buffer:
            content_list.append({"type": "image", "image": img})
        content_list.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content_list}]

        # 6. 추론 실행
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        full_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        # 7. 결과 파싱
        try:
            if "[동작 분석]:" in full_text and "정답:" in full_text:
                perception = full_text.split("[동작 분석]:")[1].split("정답:")[0].strip()
                decision = full_text.split("정답:")[1].strip()
            elif "관찰:" in full_text and "정답:" in full_text:
                perception = full_text.split("관찰:")[1].split("정답:")[0].strip()
                decision = full_text.split("정답:")[1].strip()
            else:
                perception = full_text
                decision = "..."
        except Exception:
            perception = "해석 중..."
            decision = "..."

        return perception, decision, frame_buffer
