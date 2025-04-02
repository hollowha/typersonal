import random
import gradio as gr
import os
from sample import (arg_parse, 
                    sampling,
                    load_fontdiffuer_pipeline)
from PIL import Image
import svgwrite
import shutil
from fontTools.ttLib import TTFont
import fontTools.ttLib.tables._c_m_a_p

def read_big5_characters(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [char.strip() for line in f for char in line]  # 確保是單一字元列表

def run_fontdiffuer(handwriting_image, sampling_step, guidance_scale, batch_size):
    # 讀取 Big5 繁體字全集
    characters_to_generate = read_big5_characters("big5_4808.txt")
    args.character_input = True  # 讓系統知道「只用風格圖片」，不需要 content_image
    args.sampling_step = sampling_step
    args.guidance_scale = guidance_scale
    args.batch_size = batch_size
    args.seed = random.randint(0, 10000)
    
    output_images = []
    output_folder = "generated_images"
    os.makedirs(output_folder, exist_ok=True)
    
    for char in characters_to_generate:
        if len(char) != 1:
            continue  # 避免意外讀取到多個字元的錯誤
        args.content_character = char  # 設定當前要生成的字
        out_image = sampling(
            args=args,
            pipe=pipe,
            content_image=None,  # 這裡明確設定為 None，避免報錯
            style_image=handwriting_image  # 風格圖片
        )
        
        image_path = os.path.join(output_folder, f"{ord(char)}.png")
        out_image.save(image_path)
        output_images.append(image_path)
    
    return output_images  # 回傳所有字型圖片

def create_ttf_from_images(image_folder, output_ttf):
    font = TTFont()
    cmap_table = fontTools.ttLib.tables._c_m_a_p.table_cmap()
    cmap_table.cmap = {}
    
    for image_file in os.listdir(image_folder):
        try:
            char_code = int(os.path.splitext(image_file)[0])  # 取得 Unicode 編碼
            cmap_table.cmap[char_code] = chr(char_code)
        except ValueError:
            continue  # 忽略非字元檔案
    
    font["cmap"] = cmap_table
    font.save(output_ttf)
    return output_ttf

def download_ttf():
    ttf_path = "generated_font.ttf"
    create_ttf_from_images("generated_images", ttf_path)
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
            🖋️ 上傳一張手寫字圖片，生成完整 Big5 繁體字型！
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
        
        FontDiffuser = gr.Button('Generate Full Big5 Character Set')
        output_gallery = gr.Gallery(label="Generated Font Samples", show_label=True, columns=10, rows=10)
        download_button = gr.Button("Download TTF Font")
        ttf_output = gr.File(label="Download Generated Font")
        
        FontDiffuser.click(
            fn=run_fontdiffuer,
            inputs=[handwriting_image, sampling_step, guidance_scale, batch_size],
            outputs=output_gallery
        )
        
        download_button.click(
            fn=download_ttf,
            inputs=[],
            outputs=ttf_output
        )
    
    demo.launch(debug=True)
