import os
from dotenv import load_dotenv
from telegram import Update, Document
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from chatur_core.chatur_core import ChaturAI

# Load env vars
load_dotenv(override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Create ChaturAI instance per session (basic version)
chatbot = ChaturAI(groq_api_key=GROQ_API_KEY)

# Store sessions per user
user_sessions = {}
def get_user_chatbot(chat_id: int):
    if chat_id not in user_sessions:
        user_sessions[chat_id] = ChaturAI(groq_api_key=GROQ_API_KEY)
    return user_sessions[chat_id]


# Handle /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    get_user_chatbot(chat_id)
    await update.message.reply_text("👋 Hi! I’m Chatur AI.\n\nSend me a PDF file, then ask questions about it, or anything general too!")

# Handle PDF upload with feedback
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    chatbot = get_user_chatbot(chat_id)
    
    document: Document = update.message.document
    filename = document.file_name or "uploaded.pdf"

    # Reject non-PDF files
    if document.mime_type != "application/pdf" or not filename.lower().endswith(".pdf"):
        await update.message.reply_text("Please upload PDF files only.")
        return

    # Acknowledge the start of analysis
    await update.message.reply_text(f"📄 Analyzing: {filename} ...")

    # Download and process the PDF
    try:
        file = await document.get_file()
        file_bytes = await file.download_as_bytearray()
        chatbot.add_pdf(file_bytes, filename)
        await update.message.reply_text("Successfully analyzed your PDF.\nYou can now ask questions from it.")
    except Exception as e:
        await update.message.reply_text(f"Failed to process the PDF.\nError: {e}")

# Handle messages (questions)
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    chatbot = get_user_chatbot(chat_id)

    user_input = update.message.text
    await update.message.chat.send_action(action="typing")

    response = chatbot.answer(user_input)
    await update.message.reply_text(response)

# Main app runner
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Chatur AI Telegram bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
