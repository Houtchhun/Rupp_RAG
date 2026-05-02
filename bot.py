import logging

from telegram import Update
from telegram.error import Conflict
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from qa import ask_question

TOKEN = "8695125884:AAFavd4nVYZbcbsavTvOqdb07gUWSfcBNPY"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rupp_bot")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I am RUPP AI Assistant")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    try:
        response = ask_question(user_message)
        await update.message.reply_text(response)
    except Exception as e:
        logger.exception("Failed to process message: %s", user_message)
        await update.message.reply_text(f"Sorry, I encountered an error: {str(e)}")


def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Starting RUPP Telegram bot polling...")
    try:
        app.run_polling()
    except Conflict:
        logger.error(
            "Telegram polling conflict: another bot instance is already running with the same token."
        )
    except Exception:
        logger.exception("Bot stopped unexpectedly.")


if __name__ == "__main__":
    main()