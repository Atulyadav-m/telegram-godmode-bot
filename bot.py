import os
import base64
import requests
import logging
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ======================
# LOAD ENV
# ======================
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

# ======================
# LOGGING
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GODMODE-BOT")

# ======================
# HF MODELS
# ======================
BLIP2_URL = "https://api-inference.huggingface.co/models/Salesforce/blip2-flan-t5-xl"
DEEPSEEK_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# ======================
# HF CALLS
# ======================
def call_blip2(image_bytes: bytes) -> str:
    """Image -> detailed description"""
    response = requests.post(
        BLIP2_URL,
        headers=HEADERS,
        files={"file": image_bytes},
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    return result[0]["generated_text"]

def call_deepseek(description: str) -> str:
    """Description -> GOD MODE prompt"""
    prompt = f"""
You are Gemini Pro Vision Prompt Engineer.

From the image description below, generate an EXTREMELY DETAILED AI IMAGE PROMPT.

Image description:
{description}

Return output in this format:

Scene Description:
...

Positive Prompt:
...

Negative Prompt:
...

Midjourney Prompt:
...

Stable Diffusion Prompt:
...

DALL·E Prompt:
...
"""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 700,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }

    response = requests.post(
        DEEPSEEK_URL,
        headers={**HEADERS, "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()[0]["generated_text"]

# ======================
# TELEGRAM HANDLERS
# ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔥 *GOD MODE PROMPT BOT*\n\n"
        "📸 Send me an image and I will generate:\n"
        "• Gemini-style prompt\n"
        "• Midjourney\n"
        "• Stable Diffusion\n"
        "• DALL·E\n\n"
        "⚡ Powered by BLIP-2 + DeepSeek",
        parse_mode="Markdown"
    )

async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("🧠 Analyzing image... Please wait")

        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()

        # Aspect ratio detection
        img = Image.open(BytesIO(image_bytes))
        aspect_ratio = f"{img.width}:{img.height}"

        # BLIP-2
        description = call_blip2(image_bytes)

        # DeepSeek
        final_prompt = call_deepseek(description)

        await update.message.reply_text(
            f"🔥 *PROMPTS READY*\n"
            f"📐 Aspect Ratio: `{aspect_ratio}`\n\n"
            f"{final_prompt}",
            parse_mode="Markdown"
        )

    except Exception as e:
        logger.exception(e)
        await update.message.reply_text("❌ Error processing image. Try again.")

# ======================
# MAIN
# ======================
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, image_handler))

    logger.info("🚀 GOD MODE BOT RUNNING...")
    app.run_polling()

if __name__ == "__main__":
    main()
