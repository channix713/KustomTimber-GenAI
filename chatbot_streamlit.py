# ======================================================================
# STREAMLIT CHATBOT USING STOCK OR SUMMARY SHEET
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import gspread
import os
from pathlib import Path
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from openai import OpenAI


# ======================================================================
#  PAGE SETUP
# ======================================================================
st.set_page_config(page_title="Google Sheets Inventory Chatbot", layout="wide")
st.title("📦 Inventory Chatbot (Stock + Summary Sheets)")


# ======================================================================
#  LOAD OPENAI KEY (Your required method)
# ======================================================================
try:
    ENV_PATH = Path(__file__).parent / "OpenAIKey.env"
except NameError:
    ENV_PATH = Path(os.getcwd()) / "OpenAIKey.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found in secrets or .env")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4.1-mini"


# ======================================================================
#  GOOGLE SHEETS AUTH
# ======================================================================
SERVICE_ACCOUNT_FILE = "credentials.json"

try:
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.readonly"
        ]
    )
except Exception as e:
    st.error("❌ Google credentials file missing or invalid.")
    st.stop()

gc = gspread.authorize(creds)
drive_service = build("drive", "v3", credentials=creds)

SPREADSHEET_ID = "1UG_N-zkgwCpObWTgmg8EPS7-N08aqintu8h3kN8yRmM"
WORKSHEET_NAME = "Stock"
WORKSHEET_NAME2 = "Summary"


# ======================================================================
#  LOAD SHEETS
# ======================================================================
@st.cache_data(show_spinner=True)
def load_sheets():
    stock_ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    summary_ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME2)

    # STOCK
    stock_df = pd.DataFrame(stock_ws.get_all_values())
    stock_df.columns = stock_df.iloc[0].str.strip()
    stock_df = stock_df[1:].reset_index(drop=True)
    stock_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    stock_df.dropna(how='all', inplace=True)

    # SUMMARY
    summary_df = pd.DataFrame(summary_ws.get_all_values())
    summary_df.columns = summary_df.iloc[0].str.strip()
    summary_df = summary_df[1:].reset_index(drop=True)
    summary_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    summary_df.dropna(how='all', inplace=True)

    # CLEAN
    stock_df = stock_df.replace([np.inf, -np.inf, pd.NA], np.nan)
    summary_df = summary_df.replace([np.inf, -np.inf, pd.NA], np.nan)

    # STOCK numeric
    for col in ["Product Code", "PACKS"]:
        if col in stock_df.columns:
            stock_df[col] = pd.to_numeric(stock_df[col], errors="coerce")

    # SUMMARY numeric
    summary_numeric_cols = [
        'COST','PACK SIZE','ORDERED','LANDED','Shipped','SOH (DC)',
        'Packs (DC)','Invoiced','AVAILABLE','SOH + SOO','SOO COST','SOH COST'
    ]
    for col in summary_numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    return stock_df, summary_df


stock_df, summary_df = load_sheets()


# ======================================================================
#  AI ENGINE
# ======================================================================
def ask_sheet(question, df_name):

    df = stock_df if df_name == "stock" else summary_df
    cols = list(df.columns)

    prompt = f"""
You are an expert pandas code generator.

DATAFRAME: {df_name}_df
COLUMNS: {cols}

STRICT RULES:
1. Only use EXACT column names shown above.
2. If question mentions months/dates, ALWAYS use column 'Month'
3. If question mentions packs, ALWAYS use 'PACKS'
4. Your final line MUST set variable: result
5. Return ONLY Python code.

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    ai_code = response.choices[0].message.content.strip()

    # FIX incorrect Month Required references
    ai_code = ai_code.replace("['Month Required']", "['Month']")

    if "Month Required" in ai_code:
        return f"❌ AI hallucinated 'Month Required'. BLOCKED.\n\nCODE:\n{ai_code}"

    # Execute code
    try:
        local_vars = {
            "stock_df": stock_df,
            "summary_df": summary_df,
            df_name + "_df": df
        }
        exec(ai_code, {}, local_vars)

        if "result" not in local_vars:
            return f"❌ AI did not set variable 'result'. CODE:\n{ai_code}"

        result = local_vars["result"]

    except Exception as e:
        return f"❌ Error running AI code: {e}\nCODE:\n{ai_code}"

    # Convert result for display
    result_text = result.to_string() if isinstance(result, (pd.DataFrame, pd.Series)) else str(result)

    # AI explains result
    explanation_prompt = f"""
Explain this result clearly.

QUESTION:
{question}

RESULT:
{result_text}
"""
    explanation = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": explanation_prompt}]
    ).choices[0].message.content

    return explanation


# ======================================================================
#  STREAMLIT UI
# ======================================================================
st.subheader("Choose which Google Sheet to query")
df_choice = st.selectbox("Sheet:", ["Stock Sheet (stock_df)", "Summary Sheet (summary_df)"])

df_name = "stock" if df_choice.startswith("Stock") else "summary"
df_selected = stock_df if df_name == "stock" else summary_df

if st.checkbox("Show dataframe"):
    st.dataframe(df_selected)

st.subheader("Ask a question about the selected sheet:")
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        response = ask_sheet(question, df_name)
        st.write("### Chatbot Answer:")
        st.write(response)
