import random
import os
from PIL import Image
import gradio as gr
import numpy as np
import torch
import torchvision.transforms as T
import cv2
from sample import (
    arg_parse,
    sampling,
    load_fontdiffuer_pipeline
)

from src.dpm_solver.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

PRIMARY = "#d95f20"  # 深橘
SECONDARY = "#1f3c38"  # 深綠
ACCENT = "#e6a23c"  # 金黃
LIGHT_FILL = "#7fb77e"  # 淺綠

STYLE_DIRS = {
    "書法風": "font_ref_imgs",
    "簡約現代": "modern_minimal",
    "潑墨風": "ink_style",
    "潮流街頭": "street_trend",
    "可愛手繪": "cute_handdrawn"
}

cached_image = {}
FIXED_GUIDANCE_SCALE = 7.5
FIXED_BATCH_SIZE = 1

def get_style_b_image(character, folder):
    unicode_str = str(ord(character))
    path = os.path.join(folder, f"{unicode_str}.png")
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    return None

def sampling_with_latent(args, pipe, content_image, style_latent, thickness=0.0):
    with torch.no_grad():
        content_image = content_image.to(pipe.model.device)
        style_latent = style_latent.to(pipe.model.device)

        def forward_with_latent(x, t):
            content_feat, content_res = pipe.model.content_encoder(content_image)
            content_res.append(content_feat)
            style_feat = style_latent
            style_hidden = style_feat.permute(0, 2, 3, 1).reshape(style_feat.shape[0], -1, style_feat.shape[1])
            style_content_feat, style_content_res = pipe.model.content_encoder(content_image)
            style_content_res.append(style_content_feat)
            output = pipe.model.unet(
                x, t,
                [style_feat, content_res, style_hidden, style_content_res],
                args.content_encoder_downsample_size
            )[0]
            return output

        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=pipe.train_scheduler_betas)
        model_fn = model_wrapper(
            model=forward_with_latent,
            noise_schedule=noise_schedule,
            model_type=pipe.model_type,
            guidance_type=pipe.guidance_type,
            condition=None,
            unconditional_condition=None,
            guidance_scale=pipe.guidance_scale
        )
        dpm_solver = DPM_Solver(
            model_fn=model_fn,
            noise_schedule=noise_schedule,
            algorithm_type=args.algorithm_type,
            correcting_x0_fn=args.correcting_x0_fn
        )
        x_T = torch.randn((1, 3, args.content_image_size[0], args.content_image_size[1])).to(pipe.model.device)
        x_sample = dpm_solver.sample(
            x=x_T,
            steps=args.num_inference_steps,
            order=args.order,
            skip_type=args.skip_type,
            method=args.method,
        )
        x_sample = (x_sample / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        image = (x_sample[0] * 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        if thickness < 0:
            gray = cv2.dilate(gray, kernel, iterations=int(-thickness))
        elif thickness > 0:
            gray = cv2.erode(gray, kernel, iterations=int(thickness))
        return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

def blend_styles_latent(character, image_a, style_option, alpha, thickness):
    from utils import ttf2im, load_ttf, is_char_in_font
    if not isinstance(image_a, Image.Image):
        print(f"[錯誤] image_a 不是 PIL 圖像: {type(image_a)}")
        return None

    if not isinstance(character, str) or len(character) != 1:
        print(f"[錯誤] character 不是單一中文字元: {character}")
        return None
    style_folder = STYLE_DIRS.get(style_option)
    if not style_folder:
        return None
    image_b = get_style_b_image(character, style_folder)
    if not image_a or not image_b:
        return None

    cache_key = (character, style_option, round(alpha, 2))
    if cache_key in cached_image:
        image = np.array(cached_image[cache_key])
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        if thickness > 0:
            gray = cv2.erode(gray, kernel, iterations=int(thickness))
        elif thickness < 0:
            gray = cv2.dilate(gray, kernel, iterations=int(-thickness))
        return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

    tf = T.Compose([
        T.Resize((128, 128)), T.ToTensor(), T.Normalize([0.5], [0.5])
    ])
    image_a_tensor = tf(image_a).unsqueeze(0).to(pipe.model.device)
    image_b_tensor = tf(image_b).unsqueeze(0).to(pipe.model.device)

    with torch.no_grad():
        latent_a, _, _ = pipe.model.style_encoder(image_a_tensor)
        latent_b, _, _ = pipe.model.style_encoder(image_b_tensor)
    fused_latent = (1 - alpha) * latent_a + alpha * latent_b

    if not is_char_in_font(font_path=args.ttf_path, char=character):
        return None

    font = load_ttf(ttf_path=args.ttf_path)
    content_image = ttf2im(font=font, char=character)
    content_tf = T.Compose([
        T.Resize(args.content_image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    content_tensor = content_tf(content_image)[None, :].to(pipe.model.device)

    image = sampling_with_latent(args, pipe, content_tensor, fused_latent, thickness=0)
    cached_image[cache_key] = image
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    if thickness > 0:
        gray = cv2.erode(gray, kernel, iterations=int(thickness))
    elif thickness < 0:
        gray = cv2.dilate(gray, kernel, iterations=int(-thickness))
    return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

def generate_images(character, reference_image, sampling_step):
    args.character_input = True
    args.content_character = character
    args.num_inference_steps = sampling_step
    args.guidance_scale = FIXED_GUIDANCE_SCALE
    args.batch_size = FIXED_BATCH_SIZE
    args.seed = random.randint(0, 10000)
    args.version = "V3"

    style_a_image = sampling(
        args=args,
        pipe=pipe,
        content_image=None,
        style_image=reference_image
    )

    return style_a_image, character, style_a_image

def update_blend(style_a_image, style_option, alpha, character, thickness):
    return blend_styles_latent(character, style_a_image, style_option, alpha, thickness)



if __name__ == '__main__':
    args = arg_parse()
    args.demo = True
    args.ckpt_dir = 'ckpt'
    args.ttf_path = 'ttf/KaiXinSongA.ttf'
    pipe = load_fontdiffuer_pipeline(args=args)

    theme = gr.themes.Default(
        primary_hue="orange",
        secondary_hue="green",
        neutral_hue="gray",
        radius_size="lg",
        text_size="md"
    ).set(
        slider_color="*primary_500",
        slider_color_dark="*primary_600"
    )

    with gr.Blocks(theme=theme, css="""
    #app-container {
        max-width: 960px;
        margin: 0 auto;
        padding: 1rem;
    }
    .gr-block.gr-column {
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    }
    .example-section {
        margin-top: 2rem;
    }
    .main-button button {
        background: var(--primary);
        color: white;
        font-size: 1.1rem;
        padding: 0.8rem 2rem;
        border-radius: 0.8rem;
        font-weight: bold;
    }
    #enable-blend-highlight {
        border: 2px solid #d95f20;
        padding: 0.5rem;
        border-radius: 0.5rem;
        background-color: #fff7f0;
        font-weight: bold;
        font-size: 1.05rem;
        color: #d95f20;
    }
    #style-option-highlight {
        border: 2px dashed #e6a23c;
        padding: 0.5rem;
        border-radius: 0.5rem;
        background-color: #fffbe6;
        font-weight: bold;
        font-size: 1.05rem;
        color: #d95f20;
    }
    """) as demo:
        with gr.Column(elem_id="app-container"):
            gr.HTML("""
<div class='header'>
    <img src='file/logo.png' alt='Logo' style='width:12rem; height:auto; display:block; margin:auto;'>
</div>
""")

            with gr.Row():
                with gr.Column(scale=1):
                    character = gr.Textbox(label="要生成的字", value="體")
                    sampling_step = gr.Slider(10, 30, value=15, step=5, label="精準度 (Sampling Step)")
                    reference_image = gr.Image(label="上傳你的手寫字", type="pil")
                    with gr.Accordion("▼ 範例圖示 (點擊套用)", open=False):
                        gr.Examples(
                            examples=[
                                ['龍', 'figures/ref_imgs/ref_鷢.jpg'],
                                ['轉', 'figures/ref_imgs/ref_鲸.jpg'],
                                ['懭', 'figures/ref_imgs/ref_籍_1.jpg'],
                                ['識', 'figures/ref_imgs/ref_鞣.jpg']
                            ],
                            inputs=[character, reference_image]
                        )

                with gr.Column(scale=1):
                    enable_blend = gr.Checkbox(label="啟用風格融合", elem_id="enable-blend-highlight")
                    style_a_image = gr.Image(label="AI 生成")
                    style_option = gr.Dropdown(
                        list(STYLE_DIRS.keys()),
                        value="書法風",
                        label="選擇融合風格",
                        visible=False,
                        elem_id="style-option-highlight"
                    )
                    alpha = gr.Slider(0, 1, value=0.5, step=0.05, label="融合比例 (0=A,1=B)", visible=False)
                    thickness = gr.Slider(-1.5, 1.5, value=0, step=0.1, label="後處理粗細調整", visible=False)
                    blend_output = gr.Image(label="融合輸出 (Latent + CV2)", visible=False)

            with gr.Row():
                generate_btn = gr.Button("AI擴展生成字型", elem_classes="main-button")

           
                
                

            

        state_a = gr.State()
        state_character = gr.State()

        generate_btn.click(
            fn=generate_images,
            inputs=[character, reference_image, sampling_step],
            outputs=[style_a_image, state_character, state_a]
        )



        def toggle_blend_components(enable):
            return [gr.update(visible=enable)] * 4  # alpha, thickness, blend_output, style_option

        enable_blend.change(
            fn=toggle_blend_components,
            inputs=enable_blend,
            outputs=[alpha, thickness, blend_output, style_option]
        )

        alpha.release(
            fn=update_blend,
            inputs=[state_a, style_option, alpha, state_character, thickness],
            outputs=blend_output
        )
        thickness.release(
            fn=update_blend,
            inputs=[state_a, style_option, alpha, state_character, thickness],
            outputs=blend_output
        )

    demo.launch(debug=True, share=True)