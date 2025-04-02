import random
import gradio as gr
import sys
sys.path.append("C:/Program Files/FontForge/python")
import fontforge
from sample import (arg_parse, 
                    sampling,
                    load_fontdiffuer_pipeline)

def generate_full_font(handwriting_image, sampling_step, guidance_scale, batch_size):
    # 讀取完整繁體字型字集
    with open("big5_4808.txt", "r", encoding="utf-8") as f:
        characters_to_generate = f.read().strip()

    args.character_input = True
    args.sampling_step = sampling_step
    args.guidance_scale = guidance_scale
    args.batch_size = batch_size
    args.seed = random.randint(0, 10000)
    
    output_images = {}
    for char in characters_to_generate:
        args.content_character = char
        out_image = sampling(
            args=args,
            pipe=pipe,
            content_image=None,
            style_image=handwriting_image
        )
        output_images[char] = out_image  # 存成字典 {字: 圖片}
    
    return output_images  # 回傳所有字的圖片

def generate_ttf_font(output_images):
    font = fontforge.font()
    font.encoding = "UnicodeFull"  # 設定 Unicode 繁體字型

    for char, image in output_images.items():
        glyph = font.createChar(ord(char))
        glyph.importOutlines(image)  # 匯入字體圖片
        glyph.autoTrace()  # 自動轉向量

    # 儲存 .ttf 檔案
    ttf_path = "output_font.ttf"
    font.generate(ttf_path)
    
    return ttf_path

def run_and_download_ttf(handwriting_image, sampling_step, guidance_scale, batch_size):
    output_images = generate_full_font(handwriting_image, sampling_step, guidance_scale, batch_size)
    ttf_path = generate_ttf_font(output_images)
    return ttf_path

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
            🖋️ 上傳一張手寫字圖片，生成完整繁體字型！
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
        
        FontDiffuser = gr.Button('Generate Full Traditional Chinese Font')
        ttf_download = gr.File(label="Download Generated TTF Font")
        
        FontDiffuser.click(
            fn=run_and_download_ttf,
            inputs=[handwriting_image, sampling_step, guidance_scale, batch_size],
            outputs=ttf_download
        )
    
    demo.launch(debug=True)
