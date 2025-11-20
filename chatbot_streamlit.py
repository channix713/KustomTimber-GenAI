# ======================================================================
# STREAMLIT BASED GOOGLE SHEETS CHATBOT (Stock + Summary)
# With get_google_creds(), OpenAIKey.env loading, and strict AI control
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import gspread
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from openai import OpenAI

# ======================================================================
#  STREAMLIT PAGE SETUP
# ======================================================================
st.set_page_config(page_title="Inventory Chatbot", layout="wide")
st.title("📦 Inventory Chatbot (Stock + Summary Sheets)")


# ======================================================================
#  LOAD OPENAI API KEY (Your required method)
# ======================================================================
try:
    ENV_PATH = Path(__file__).parent / "OpenAIKey.env"
except NameError:
    ENV_PATH = Path(os.getcwd()) / "OpenAIKey.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY missing in secrets or .env")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4.1-mini"


# ======================================================================
#  GOOGLE SHEETS AUTH — Using get_google_creds()
# ======================================================================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

def get_google_creds():
    """Load GCP service account from Streamlit secrets."""
    if "GOOGLE_SERVICE_ACCOUNT_JSON" not in st.secrets:
        st.error("❌ Missing GOOGLE_SERVICE_ACCOUNT_JSON in secrets.")
        st.stop()
    try:
        info = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
        return Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception as e:
        st.error(f"❌ Invalid Google credentials: {e}")
        st.stop()

creds = get_google_creds()
gc = gspread.authorize(creds)
drive_service = build("drive", "v3", credentials=creds)


# ======================================================================
#  SHEET CONFIG
# ======================================================================
SPREADSHEET_ID = "1UG_N-zkgwCpObWTgmg8EPS7-N08aqintu8h3kN8yRmM"
WORKSHEET_NAME = "Stock"
WORKSHEET_NAME2 = "Summary"


# ======================================================================
#  LOAD SHEETS (with caching)
# ======================================================================
@st.cache_data(show_spinner=True)
def load_sheets():
    # --- LOAD STOCK ---
    ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    stock_df = pd.DataFrame(ws.get_all_values())
    stock_df.columns = stock_df.iloc[0].str.strip()
    stock_df = stock_df[1:].reset_index(drop=True)
    stock_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    stock_df.dropna(how="all", inplace=True)

    # Convert numeric columns
    for col in ["Product Code", "PACKS"]:
        if col in stock_df.columns:
            stock_df[col] = pd.to_numeric(stock_df[col], errors="coerce")

    # --- LOAD SUMMARY ---
    ws2 = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME2)
    summary_df = pd.DataFrame(ws2.get_all_values())
    summary_df.columns = summary_df.iloc[0].str.strip()
    summary_df = summary_df[1:].reset_index(drop=True)
    summary_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    summary_df.dropna(how="all", inplace=True)

    summary_numeric_cols = [
        "COST", "PACK SIZE", "ORDERED", "LANDED", "Shipped",
        "SOH (DC)", "Packs (DC)", "Invoiced", "AVAILABLE",
        "SOH + SOO", "SOO COST", "SOH COST"
    ]

    for col in summary_numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    return stock_df, summary_df


stock_df, summary_df = load_sheets()


# ======================================================================
#  AI ENGINE (strict, safe, multi-line execution)
# ======================================================================
def ask_sheet(question, df_name):

    df = stock_df if df_name == "stock" else summary_df
    cols = list(df.columns)

    prompt = f"""
You are an expert pandas code generator.

DATAFRAME: {df_name}_df
COLUMNS = {cols}

STRICT RULES:
1. Only use EXACT column names listed above.
2. If question mentions date/month, use column 'Month' ONLY.
3. Never use 'Month Required'.
4. If question mentions packs → use 'PACKS'.
5. Final line of code MUST set variable: result
6. Output ONLY Python code.

QUESTION:
{question}
"""

    # ---- Generate Python code ----
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    ai_code = resp.choices[0].message.content.strip()

    # Fix accidental hallucinations
    ai_code = ai_code.replace("['Month Required']", "['Month']")

    if "Month Required" in ai_code:
        return f"❌ Invalid column 'Month Required'. BLOCKED.\n\nCODE:\n{ai_code}"

    # ---- EXECUTE PYTHON CODE SAFELY ----
    try:
        local_vars = {
            "stock_df": stock_df,
            "summary_df": summary_df,
            df_name + "_df": df
        }

        exec(ai_code, {}, local_vars)

        if "result" not in local_vars:
            return f"❌ AI did not create variable 'result'.\n\nCODE:\n{ai_code}"

        result = local_vars["result"]

    except Exception as e:
        return f"❌ Error running AI code: {e}\n\nCODE:\n{ai_code}"

    # Format result
    result_text = (
        result.to_string() if isinstance(result, (pd.DataFrame, pd.Series)) else str(result)
    )

    # ---- Ask AI to explain the result ----
    explain_prompt = f"""
Explain the result clearly.

QUESTION:
{question}

RESULT:
{result_text}
"""

    explanation = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": explain_prompt}]
    ).choices[0].message.content

    return explanation


# ======================================================================
#  STREAMLIT UI
# ======================================================================
st.subheader("Select which sheet to query")
sheet_choice = st.selectbox(
    "Choose sheet:",
    ["Stock Sheet (stock_df)", "Summary Sheet (summary_df)"]
)

df_name = "stock" if sheet_choice.startswith("Stock") else "summary"
df_selected = stock_df if df_name == "stock" else summary_df

if st.checkbox("Show dataframe"):
    st.dataframe(df_selected, use_container_width=True)

st.subheader("Ask a question:")
question = st.text_input("Enter your question")

if st.button("Ask"):
    if not question.strip():
        st.warning("Enter a question before clicking Ask.")
    else:
        response = ask_sheet(question, df_name)
        st.write("### Chatbot Answer:")
        st.write(response)
