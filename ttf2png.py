from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
import os
import sys

# === 可修改參數 ===
FONT_PATH = "ttf/JasonHandwriting6.ttf"
OUTPUT_DIR = "cute_handdrawn"
IMG_SIZE = 96
FONT_SIZE = 72

os.makedirs(OUTPUT_DIR, exist_ok=True)
ttf = TTFont(FONT_PATH)

# 取得所有 Unicode 字碼
unicode_chars = set()
for table in ttf['cmap'].tables:
    if table.isUnicode():
        unicode_chars.update(table.cmap.keys())

# 排序處理
unicode_chars = sorted(unicode_chars)
total = len(unicode_chars)

font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

print(f"🔢 共找到 {total} 個支援字元，開始轉圖...\n")

# 建立進度列函式
def print_progress(current, total, bar_length=40):
    percent = current / total
    arrow = '█' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f"\r進度: |{arrow}{spaces}| {int(percent * 100)}% ({current}/{total})")
    sys.stdout.flush()

count = 0
for i, codepoint in enumerate(unicode_chars):
    char = chr(codepoint)
    file_path = os.path.join(OUTPUT_DIR, f"{codepoint}.png")

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color="white")
    draw = ImageDraw.Draw(img)

    # 取得文字範圍並置中
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = ((IMG_SIZE - w) / 2, (IMG_SIZE - h) / 2)

    draw.text(pos, char, font=font, fill="black")
    img.save(file_path)
    count += 1

    # 每 10 字更新一次進度條
    if i % 10 == 0 or i == total - 1:
        print_progress(i + 1, total)

# 最後補上換行
print("\n✅ 字圖轉換完成！共輸出：", count, "張圖片")
