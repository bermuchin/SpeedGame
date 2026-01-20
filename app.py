# app.py
import gradio as gr
from agent import MotionAgent # agent.py에서 클래스 가져오기
import config # 카테고리 리스트 가져오기

# 1. 에이전트 인스턴스 생성 (한 번만 로드)
# 전역 변수로 생성하여 메모리에 상주
brain_agent = MotionAgent()

# 2. UI 래퍼 함수
def run_inference(image, categories, buffer):
    return brain_agent.infer_stream(image, categories, buffer)

# 3. Gradio UI 구성
with gr.Blocks(title="A6000 Dual-Stream Brain") as demo:
    gr.Markdown("## Real-time Agent ")
    gr.Markdown("Center Crop & Multi-Frame Logic Applied")
    
    # [State] 사용자별 버퍼 메모리
    state_buffer = gr.State(None)

    with gr.Row():
        # Input
        input_cam = gr.Image(sources=["webcam"], label="Dorsal Stream (Live)", streaming=True)
        
        # Controls & Output
        with gr.Column():
            category_selector = gr.Radio(
                choices=list(config.WORD_DICT.keys()),
                label="활성 카테고리",
                value="동물"
            )
            
            clear_btn = gr.Button("뇌 리셋 (다음 문제)", variant="primary")
            gr.Markdown("---") 

            output_perception = gr.Textbox(label="동작 분석", lines=2)
            output_decision = gr.Textbox(label="최종 판단", elem_classes="answer-box")
        
        # Reset Logic
        clear_btn.click(
            fn=lambda: (None, "", ""),         
            inputs=None, 
            outputs=[state_buffer, output_perception, output_decision] 
        )

    # Styling
    demo.css = ".answer-box {font-size: 50px; font-weight: bold; color: #2c3e50; text-align: center; background-color: #f0f0f0;}"
    
    # Streaming Logic
    input_cam.stream(
        fn=run_inference, 
        inputs=[input_cam, category_selector, state_buffer], 
        outputs=[output_perception, output_decision, state_buffer],
        time_limit=10,
        stream_every=0.2 
    )

if __name__ == "__main__":
    print(">>> [System] UI Launching...")
    demo.launch(share=True)
