import random
import gradio as gr
from sample import (arg_parse, 
                    sampling,
                    load_fontdiffuer_pipeline)

def run_fontdiffuer(handwriting_image, sampling_step, guidance_scale, batch_size):
    # 設定 100 個要生成的中文字
    characters_to_generate = "你好世界繁體字體生成風格字庫測試手寫字型藝術設計創作書法漢字"[:100]
    args.character_input = True  # 🆕 讓系統知道「只用風格圖片」，不需要 content_image
    args.sampling_step = sampling_step
    args.guidance_scale = guidance_scale
    args.batch_size = batch_size
    args.seed = random.randint(0, 10000)
    
    output_images = []
    for char in characters_to_generate:
        args.content_character = char  # 設定當前要生成的字
        out_image = sampling(
            args=args,
            pipe=pipe,
            content_image=None,  # 🆕 這裡明確設定為 None，避免報錯
            style_image=handwriting_image  # 風格圖片
        )
        output_images.append(out_image)
    
    return output_images  # 回傳 100 張字型圖片


if __name__ == '__main__':
    args = arg_parse()
    args.demo = True
    args.ckpt_dir = 'ckpt'
    args.ttf_path = 'ttf/KaiXinSongA.ttf'

    # 載入 FontDiffuser 預訓練模型
    pipe = load_fontdiffuer_pipeline(args=args)

    with gr.Blocks() as demo:
        gr.HTML("""
            <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
            <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">FontDiffuser</h1>
            <h2 style="text-align: left; font-weight: 600; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
            🖋️ 上傳一張手寫字圖片，生成一整套字型！
            </h2>
            </div>
        """)
        handwriting_image = gr.Image(label="Upload Your Handwritten Style Image", image_mode="RGB", type="pil")

        sampling_step = gr.Slider(20, 50, value=20, step=10, 
                                  label="Sampling Step", info="The sampling step by FontDiffuser.")
        guidance_scale = gr.Slider(1, 12, value=7.5, step=0.5, 
                                   label="Scale of Classifier-free Guidance", 
                                   info="The scale used for classifier-free guidance sampling")
        batch_size = gr.Slider(1, 4, value=1, step=1, 
                               label="Batch Size", info="The number of images to be sampled.")
        
        FontDiffuser = gr.Button('Generate 100 Characters')
        output_gallery = gr.Gallery(label="Generated Font Samples", show_label=True, columns=10, rows=10)
        
        FontDiffuser.click(
            fn=run_fontdiffuer,
            inputs=[handwriting_image, sampling_step, guidance_scale, batch_size],
            outputs=output_gallery
        )
    
    demo.launch(debug=True)