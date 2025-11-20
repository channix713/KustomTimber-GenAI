# ======================================================================
# STREAMLIT GOOGLE SHEETS CHATBOT (Stock + Summary)
# With isolated dataframes, strict validator, auto-refresh, case-insensitive
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import gspread
import os
import json
import re
import ast
from pathlib import Path
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from openai import OpenAI

# ======================================================================
# STREAMLIT SETUP
# ======================================================================
st.set_page_config(page_title="Inventory Chatbot", layout="wide")
st.title("📦 Inventory Chatbot — Stock & Summary Sheets")


# ======================================================================
# LOAD OPENAI KEY (your required method)
# ======================================================================
try:
    ENV_PATH = Path(__file__).parent / "OpenAIKey.env"
except NameError:
    ENV_PATH = Path(os.getcwd()) / "OpenAIKey.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY in secrets or .env")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4.1-mini"


# ======================================================================
# GOOGLE CREDS
# ======================================================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

def get_google_creds():
    """Load credentials from Streamlit Secrets."""
    if "GOOGLE_SERVICE_ACCOUNT_JSON" not in st.secrets:
        st.error("❌ GOOGLE_SERVICE_ACCOUNT_JSON missing in secrets.")
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
# SHEET CONFIG
# ======================================================================
SPREADSHEET_ID = "1UG_N-zkgwCpObWTgmg8EPS7-N08aqintu8h3kN8yRmM"
WS_STOCK = "Stock"
WS_SUMMARY = "Summary"


# ======================================================================
# AUTO-REFRESH / MANUAL REFRESH
# ======================================================================
def refresh_sheets():
    st.cache_data.clear()
    st.success("🔄 Google Sheets refreshed successfully!")
    st.stop()

if st.button("🔄 Refresh Sheets Now"):
    refresh_sheets()


# ======================================================================
# LOAD SHEETS
# ======================================================================
@st.cache_data(show_spinner=True)
def load_sheets():

    # =======================
    # STOCK SHEET
    # =======================
    ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WS_STOCK)
    stock_df = pd.DataFrame(ws.get_all_values())
    stock_df.columns = stock_df.iloc[0].str.strip()
    stock_df = stock_df[1:].reset_index(drop=True)
    stock_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    for col in ["Product Code", "PACKS"]:
        if col in stock_df:
            stock_df[col] = pd.to_numeric(stock_df[col], errors="coerce")

    # =======================
    # SUMMARY SHEET
    # =======================
    ws2 = gc.open_by_key(SPREADSHEET_ID).worksheet(WS_SUMMARY)
    summary_df = pd.DataFrame(ws2.get_all_values())
    summary_df.columns = summary_df.iloc[0].str.strip()
    summary_df = summary_df[1:].reset_index(drop=True)
    summary_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    numeric_cols = [
        "COST","PACK SIZE","ORDERED","LANDED","Shipped","SOH (DC)",
        "Packs (DC)","Invoiced","AVAILABLE","SOH + SOO","SOO COST","SOH COST"
    ]
    for col in numeric_cols:
        if col in summary_df:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    # Create AVAILABLE_num for consistency
    if "AVAILABLE" in summary_df:
        summary_df["AVAILABLE_num"] = pd.to_numeric(summary_df["AVAILABLE"], errors="coerce")

    return stock_df, summary_df


stock_df, summary_df = load_sheets()


# ======================================================================
# STRICT VALIDATOR
# ======================================================================
FORBIDDEN = ["import", "open(", "os.", "sys.", "subprocess", "__", "eval", "exec"]

def extract_columns(code_text):
    pattern = r'\[\s*[\'"]([^\'"]+)[\'"]\s*\]'
    return re.findall(pattern, code_text)

def validate_ai_code(ai_code: str, df_columns):
    # Block forbidden keywords
    for bad in FORBIDDEN:
        if bad in ai_code:
            return False, f"Forbidden keyword detected: {bad}"

    # Must define result
    if "result =" not in ai_code:
        return False, "Missing: result ="

    # Check syntax
    try:
        ast.parse(ai_code)
    except Exception as e:
        return False, f"Syntax error: {e}"

    # Validate columns (case-insensitive)
    lower_cols = [c.lower() for c in df_columns]
    for col in extract_columns(ai_code):
        if col.lower() not in lower_cols:
            return False, f"Invalid column (not in dataframe): {col}"

    return True, "OK"


# ======================================================================
# AI ENGINE WITH DATAFRAME ISOLATION
# ======================================================================
def ask_sheet(question, df_name):

    # CASE INSENSITIVE
    q = question.lower().strip()

    # Pick dataframe
    if df_name == "stock":
        df = stock_df
        df_label = "stock_df"
        id_col = "Product Code"
        value_col = "PACKS"
        forbidden_cols = ["ITEM #", "AVAILABLE", "AVAILABLE_num"]
    else:
        df = summary_df
        df_label = "summary_df"
        id_col = "ITEM #"
        value_col = "AVAILABLE_num"
        forbidden_cols = ["Product Code", "PACKS", "Month"]

    cols = list(df.columns)

    # ---- AI PROMPT ----
    prompt = f"""
You are an expert pandas code generator.

You are working ONLY with this dataframe:

DATAFRAME NAME: {df_label}
COLUMNS: {cols}

You MUST follow these rules:

1. ONLY use the dataframe {df_label}.
2. Product ID column = "{id_col}"
3. Main numeric column = "{value_col}"
4. Forbidden columns (NEVER use):
   {forbidden_cols}
5. Use EXACT column names as shown above.
6. Final line MUST be:
       result = <value>
7. Output ONLY python code (no explanations).

QUESTION:
{q}
"""

    # --- AI GENERATES PYTHON ---
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}]
    )
    ai_code = resp.choices[0].message.content.strip()

    # CLEAN CODE
    ai_code = (
        ai_code.replace("```python", "")
               .replace("```", "")
               .replace("“", '"')
               .replace("”", '"')
               .replace("’", "'")
               .strip()
    )

    # VALIDATION
    ok, msg = validate_ai_code(ai_code, df.columns)
    if not ok:
        return f"❌ AI Code Validation Failed:\n{msg}\n\nCODE:\n{ai_code}"

    # EXECUTE
    try:
        local_vars = {df_label: df}
        exec(ai_code, {}, local_vars)
        result = local_vars["result"]
    except Exception as e:
        return f"❌ Error executing AI code: {e}\n\nCODE:\n{ai_code}"

    # FORMAT
    result_text = result.to_string() if isinstance(result, (pd.DataFrame, pd.Series)) else str(result)

    # EXPLAIN
    explain_prompt = f"""
Explain this result clearly:

QUESTION:
{question}

RESULT:
{result_text}
"""
    explanation = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":explain_prompt}]
    ).choices[0].message.content

    return explanation


# ======================================================================
# UI
# ======================================================================
st.subheader("Choose Sheet to Query")
sheet_choice = st.selectbox("Sheet:", ["Stock Sheet", "Summary Sheet"])

df_name = "stock" if sheet_choice == "Stock Sheet" else "summary"
df_selected = stock_df if df_name == "stock" else summary_df

if st.checkbox("Show DataFrame Preview"):
    st.dataframe(df_selected, use_container_width=True)

st.subheader("Ask a question:")
question = st.text_input("Ask anything about the selected sheet (case-insensitive):")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        st.markdown("### Chatbot Answer:")
        st.write(ask_sheet(question, df_name))
