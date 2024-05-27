import os
import subprocess
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, CallbackContext
import concurrent.futures
import json


# /start
async def start(update: Update, context: CallbackContext) -> None:
    keyboard = [
        [InlineKeyboardButton("Category", callback_data='category')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Hi! I am your investment assistant. Choose the stock category you're interested in.",
        reply_markup=reply_markup
    )

async def button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()

    if query.data == 'category':
        keyboard = [
            [InlineKeyboardButton("Consumer goods and services", callback_data='consumer_goods_and_services')],
            [InlineKeyboardButton("Electric power", callback_data='electric_power')],
            [InlineKeyboardButton("Energy", callback_data='energy')],
            [InlineKeyboardButton("Financial sector", callback_data='financial_sector')],
            [InlineKeyboardButton("Healthcare", callback_data='healthcare')],
            [InlineKeyboardButton("Information technology", callback_data='info_tech')],
            [InlineKeyboardButton("Real Estate", callback_data='real_estate')],
            [InlineKeyboardButton("Engineering and transport", callback_data='mech_engineering_and_transport')],
            [InlineKeyboardButton("Raw materials", callback_data='raw_materials')],
            [InlineKeyboardButton("Telecommunications", callback_data='telecommunications')],
            [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data='back_to_main')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            'Choose the category:',
            reply_markup=reply_markup
        )
    elif query.data == 'back_to_main':
        keyboard = [
            [InlineKeyboardButton("Category", callback_data='category')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            'Hi! I am your investment assistant. You can choose a category of stocks.',
            reply_markup=reply_markup
        )
    elif query.data in [
            'consumer_goods_and_services', 'electric_power', 'energy', 
            'financial_sector', 'healthcare', 'info_tech', 'real_estate', 
            'mech_engineering_and_transport', 'raw_materials', 'telecommunications']:
        context.user_data['last_category'] = query.data
        await show_company_buttons(update, context, query.data)
    elif query.data.startswith('company_'):
        ticker = query.data.split('_', 1)[1]
        await show_company_options(update, context, ticker)
    elif query.data == 'back_to_categories':
        await show_company_buttons(update, context, context.user_data['last_category'])
    elif query.data.startswith('show_'):
        category = query.data.split('_', 1)[1]
        await show_company_buttons(update, context, category)
    else:
        await show_company_buttons(update, context, query.data)

async def show_company_buttons(update: Update, context: CallbackContext, category: str) -> None:
    company_file_path = f"/Users/leo/Desktop/Invest_hub/data/data_for_tikers_by_spheres/{category}.txt"

    if not os.path.exists(company_file_path):
        await update.callback_query.message.reply_text("No companies found for this category.")
        return

    keyboard = []
    with open(company_file_path, 'r') as file:
        for line in file:
            ticker, _, company_name = line.strip().split(',')
            keyboard.append([InlineKeyboardButton(f"{company_name} ({ticker})", callback_data=f'company_{ticker}')])
    
    keyboard.append([InlineKeyboardButton("⬅️ Back to Categories", callback_data='back_to_categories')])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text("Choose a company:", reply_markup=reply_markup)

    for line in open(company_file_path, 'r'):
        ticker, _, _ = line.strip().split(',')
        if f'prediction_{ticker}' in context.user_data:
            await update.callback_query.message.reply_text(context.user_data[f'prediction_{ticker}'])

async def show_company_options(update: Update, context: CallbackContext, ticker: str) -> None:
    category = context.user_data.get('last_category')
    company_name = ticker

    company_file_path = f"/Users/leo/Desktop/Invest_hub/data/data_for_tikers_by_spheres/{category}.txt"
    if os.path.exists(company_file_path):
        with open(company_file_path, 'r') as file:
            for line in file:
                t, _, name = line.strip().split(',')
                if t == ticker:
                    company_name = name
                    break

    analysis_result = run_analysis_script(ticker)
    await update.callback_query.message.reply_text(f"{company_name}\n\n{analysis_result}")

    waiting_message = await update.callback_query.message.reply_text("Sorry, but it takes a bit of time to analyse stocks and predict prices, so you have to wait a little while\n\n Waiting for predictions...")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_prediction = executor.submit(run_prediction_script, ticker)
        predictions = future_prediction.result()

    await waiting_message.delete()

    if "error" not in predictions:
        prediction_message = (
            f"Predicted prices:\n"
            f"- Next day: {predictions['day']['price']} ({predictions['day']['change']})\n"
            f"- Next week: {predictions['week']['price']} ({predictions['week']['change']})\n"
            f"- Next month: {predictions['month']['price']} ({predictions['month']['change']})"
        )
    else:
        prediction_message = f"Failed to retrieve predictions: {predictions['error']}"

    max_message_length = 4096
    if len(prediction_message) > max_message_length:
        prediction_message = prediction_message[:max_message_length] + "\n... (message truncated)"

    print(f"Final prediction message length: {len(prediction_message)}")
    
    context.user_data['last_prediction_message'] = prediction_message

    await update.callback_query.message.reply_text(prediction_message)

    # /back
    keyboard = [
        [InlineKeyboardButton("Buy", url=f'https://www.tinkoff.ru/invest/stocks/{ticker}')],
        [InlineKeyboardButton("⬅️ Back", callback_data='back_to_categories')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.message.reply_text("Choose an action:", reply_markup=reply_markup)


def run_analysis_script(ticker: str) -> str:
    try:
        result = subprocess.run(['python3', '/Users/leo/Desktop/Invest_hub/decision.py', ticker], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return "An error occurred while running the analysis script."
    except Exception as e:
        return str(e)

def run_prediction_script(ticker: str) -> dict:
    try:
        command = ['python3', '/Users/leo/Desktop/Invest_hub/predictions/lstm_pred_main.py', ticker]
        print(f"Running command: {' '.join(command)}")
        
        with open(os.devnull, 'w') as fnull:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=fnull, text=True)
        
        print(f"Command output: {result.stdout}")
        if result.returncode == 0:
            predictions = json.loads(result.stdout)
            return predictions
        else:
            print(f"Command error: {result.stderr}")
            return {"error": "An error occurred while running the prediction script."}
    except Exception as e:
        print(f"Exception: {e}")
        return {"error": str(e)}


def main():
    application = Application.builder().token("7196406688:AAHUKWbp-u4tAwsWnIuMYq2OWq4r3mf5p8A").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))

    # Запуск бота
    application.run_polling()

if __name__ == '__main__':
    main()