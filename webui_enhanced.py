"""
IndexTTS2 增强版 WebUI
支持完整情感控制 + 推理参数调优
可作为独立服务运行，也可被 api_server.py 挂载
"""
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import argparse


def create_webui(get_tts):
    """创建增强版 WebUI，接受 get_tts 可调用对象，延迟获取模型实例"""

    def synthesize(text, spk_audio, emo_audio, emo_alpha,
                   happy, angry, sad, afraid, disgusted, melancholic, surprised, calm,
                   use_emo_text, emo_text, use_random,
                   num_beams, do_sample, top_k, top_p, temperature, max_mel_tokens,
                   length_penalty, repetition_penalty):
        tts = get_tts()
        if tts is None:
            raise gr.Error("模型未初始化")
        try:
            os.makedirs("outputs", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"outputs/tts_{timestamp}.wav"

            emo_vector_val = [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            if not any(emo_vector_val):
                emo_vector_val = None

            tts.infer(
                spk_audio_prompt=spk_audio,
                text=text,
                output_path=output_path,
                emo_audio_prompt=emo_audio if emo_audio else None,
                emo_alpha=emo_alpha,
                emo_vector=emo_vector_val,
                use_emo_text=use_emo_text,
                emo_text=emo_text if emo_text else None,
                use_random=use_random,
                verbose=True,
                num_beams=int(num_beams),
                do_sample=do_sample,
                top_k=int(top_k),
                top_p=top_p,
                temperature=temperature,
                max_mel_tokens=int(max_mel_tokens),
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
            )
            return output_path
        except Exception as e:
            error_msg = f"生成失败: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            raise gr.Error(f"生成失败: {str(e)}")

    with gr.Blocks(title="IndexTTS2", fill_height=True) as demo:
        gr.Markdown("""
        # IndexTTS2 - 情感可控的零样本语音合成
        上传参考音频（3-10秒）→ 输入文本 → 调节情感和参数 → 生成语音
        """)

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="合成文本",
                    placeholder="请输入要合成的文字...\n支持中英文混合，支持拼音标注（如：之前你做DE5很好）",
                    lines=4,
                )
                spk_audio = gr.Audio(label="音色参考音频（必需）", type="filepath", sources=["upload", "microphone"])
                emo_audio = gr.Audio(label="情感参考音频（可选）", type="filepath", sources=["upload", "microphone"])

                with gr.Accordion("情感控制", open=False):
                    emo_alpha = gr.Slider(0, 1, value=1.0, label="情感强度 (emo_alpha)")
                    gr.Markdown("8 维情感向量：[开心, 愤怒, 悲伤, 恐惧, 厌恶, 忧郁, 惊讶, 平静]")
                    with gr.Row():
                        happy = gr.Slider(0, 1, value=0, label="开心")
                        angry = gr.Slider(0, 1, value=0, label="愤怒")
                        sad = gr.Slider(0, 1, value=0, label="悲伤")
                        afraid = gr.Slider(0, 1, value=0, label="恐惧")
                    with gr.Row():
                        disgusted = gr.Slider(0, 1, value=0, label="厌恶")
                        melancholic = gr.Slider(0, 1, value=0, label="忧郁")
                        surprised = gr.Slider(0, 1, value=0, label="惊讶")
                        calm = gr.Slider(0, 1, value=0, label="平静")

                    use_emo_text = gr.Checkbox(label="文本情感识别", value=False)
                    emo_text = gr.Textbox(label="独立情感文本（可选）", placeholder="如：你吓死我了！", lines=2)
                    use_random = gr.Checkbox(label="随机采样（降低克隆保真度）", value=False)

                with gr.Accordion("推理参数调优", open=False):
                    gr.Markdown("调整这些参数可以优化速度或质量")
                    num_beams = gr.Slider(1, 10, value=3, step=1, label="num_beams（beam search 宽度，1=贪心）")
                    do_sample = gr.Checkbox(label="do_sample（采样）", value=True)
                    top_k = gr.Slider(1, 100, value=30, step=1, label="top_k")
                    top_p = gr.Slider(0, 1, value=0.8, label="top_p")
                    temperature = gr.Slider(0.1, 2.0, value=0.8, label="temperature")
                    max_mel_tokens = gr.Slider(100, 3000, value=1500, step=50, label="max_mel_tokens")
                    length_penalty = gr.Slider(0.0, 2.0, value=0.0, label="length_penalty（长度惩罚）")
                    repetition_penalty = gr.Slider(0.0, 20.0, value=10.0, label="repetition_penalty（重复惩罚）")
                    gr.Markdown("**速度优化推荐**：num_beams=1, do_sample=False, top_k=10")

            with gr.Column():
                submit_btn = gr.Button("生成语音", variant="primary", size="lg")
                output_audio = gr.Audio(label="合成结果", type="filepath")
                gr.Markdown("""
                ---
                **使用场景**：
                - 简单克隆：只上传音色参考音频
                - 情感克隆：音色参考音频 + 情感音频
                - 精确控制：音色参考音频 + 手动调情感向量
                - 智能识别：音色参考音频 + 勾选文本情感（alpha 建议 0.6）
                """)

        submit_btn.click(
            synthesize,
            inputs=[text_input, spk_audio, emo_audio, emo_alpha,
                    happy, angry, sad, afraid, disgusted, melancholic, surprised, calm,
                    use_emo_text, emo_text, use_random,
                    num_beams, do_sample, top_k, top_p, temperature, max_mel_tokens,
                    length_penalty, repetition_penalty],
            outputs=output_audio,
        )

    return demo


# 独立运行
if __name__ == "__main__":
    from indextts.infer_v2 import IndexTTS2

    parser = argparse.ArgumentParser(description="IndexTTS2 Enhanced WebUI")
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--model_dir", default="checkpoints/IndexTTS-2")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_cuda_kernel", action="store_true")
    parser.add_argument("--use_deepspeed", action="store_true")
    args = parser.parse_args()

    tts = IndexTTS2(
        cfg_path=os.path.join(args.model_dir, "config.yaml"),
        model_dir=args.model_dir,
        use_fp16=args.use_fp16,
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed,
    )
    demo = create_webui(lambda: tts)
    demo.launch(server_name=args.server_name, server_port=args.server_port)
