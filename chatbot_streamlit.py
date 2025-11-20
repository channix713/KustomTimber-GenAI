# ======================================================================
# STREAMLIT GOOGLE SHEETS CHATBOT (Stock + Summary)
# FINAL VERSION — with python-prefix fix & safe indexing
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
    st.error("❌ Missing OPENAI_API_KEY in secrets or env")
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
# AUTO-REFRESH BUTTON
# ======================================================================
def refresh_sheets():
    st.cache_data.clear()
    st.success("🔄 Sheets refreshed!")
    st.experimental_rerun()

if st.button("🔄 Refresh Sheets Now"):
    refresh_sheets()


# ======================================================================
# LOAD SHEETS
# ======================================================================
@st.cache_data(show_spinner=True)
def load_sheets():

    # ---------- STOCK ----------
    ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WS_STOCK)
    stock_df = pd.DataFrame(ws.get_all_values())
    stock_df.columns = stock_df.iloc[0].str.strip()
    stock_df = stock_df[1:].reset_index(drop=True)

    for col in ["Product Code", "PACKS"]:
        if col in stock_df:
            stock_df[col] = pd.to_numeric(stock_df[col], errors="coerce")

    # ---------- SUMMARY ----------
    ws2 = gc.open_by_key(SPREADSHEET_ID).worksheet(WS_SUMMARY)
    summary_df = pd.DataFrame(ws2.get_all_values())
    summary_df.columns = summary_df.iloc[0].str.strip()
    summary_df = summary_df[1:].reset_index(drop=True)

    numeric_cols = [
        "COST","PACK SIZE","ORDERED","LANDED","Shipped","SOH (DC)",
        "Packs (DC)","Invoiced","AVAILABLE","SOH + SOO","SOO COST","SOH COST","ITEM #"
    ]
    for col in numeric_cols:
        if col in summary_df:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    summary_df["AVAILABLE_num"] = pd.to_numeric(summary_df.get("AVAILABLE", np.nan), errors="coerce")

    return stock_df, summary_df


stock_df, summary_df = load_sheets()


# ======================================================================
# STRICT VALIDATOR
# ======================================================================
FORBIDDEN = [
    "import", "open(", "os.", "sys.", "subprocess", "__",
    "eval", "exec", "globals", "locals",
    ".iloc", "values[", "to_numpy", "[0]"
]

def extract_columns(code_text):
    return re.findall(r'\[\s*[\'"]([^\'"]+)[\'"]\s*\]', code_text)

def validate_ai_code(ai_code: str, df_columns):

    for bad in FORBIDDEN:
        if bad in ai_code.lower():
            return False, f"Forbidden pattern detected: {bad}"

    if "result =" not in ai_code:
        return False, "Missing required: result ="

    try:
        ast.parse(ai_code)
    except Exception as e:
        return False, f"Syntax error: {e}"

    lower_cols = [c.lower() for c in df_columns]
    for col in extract_columns(ai_code):
        if col.lower() not in lower_cols:
            return False, f"Invalid column used: {col}"

    return True, "OK"


# ======================================================================
# AI ENGINE (SAFE)
# ======================================================================
def ask_sheet(question, df_name):

    q = question.lower().strip()

    # Pick dataframe + rules
    if df_name == "stock":
        df = stock_df
        df_label = "stock_df"
        id_col = "Product Code"
        numeric_col = "PACKS"
        forbidden_cols = ["ITEM #", "AVAILABLE", "AVAILABLE_num"]
    else:
        df = summary_df
        df_label = "summary_df"
        id_col = "ITEM #"
        numeric_col = "AVAILABLE_num"
        forbidden_cols = ["Product Code", "PACKS", "Month"]

    cols = list(df.columns)

    # ------------------ AI PROMPT ------------------
    prompt = f"""
You generate SAFE pandas code.

DATAFRAME: {df_label}
COLUMNS: {cols}

RULES:
- Only use {df_label}.
- Product ID column: "{id_col}"
- Numeric column: "{numeric_col}"
- Forbidden columns: {forbidden_cols}
- NEVER use .iloc, .values, .to_numpy, or [0]
- To get a single value ALWAYS use: .sum(), .max(), .min(), or .head(1)
- Final line MUST be: result = <value>
- Output ONLY pure python code.

QUESTION:
{q}
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}]
    )

    ai_code = resp.choices[0].message.content.strip()

    # ---------------- CLEAN CODE ----------------
    ai_code = ai_code.replace("```python","").replace("```","").strip()
    ai_code = ai_code.replace("“", '"').replace("”", '"').replace("’", "'")

    # Remove python-prefixes
    ai_code = re.sub(r"^python\s+", "", ai_code)
    ai_code = re.sub(r"^python:", "", ai_code)
    ai_code = re.sub(r"^py\s+", "", ai_code)
    ai_code = ai_code.strip()

    # BLOCK unwanted indexing
    ai_code = ai_code.replace(".iloc[0]", "")
    ai_code = re.sub(r"\.values *\[[^\]]+\]", "", ai_code)
    ai_code = re.sub(r"\.to_numpy *\([^\)]*\)", "", ai_code)

    # VALIDATE
    ok, msg = validate_ai_code(ai_code, df.columns)
    if not ok:
        return f"❌ AI Code Validation Failed:\n{msg}\n\nCODE:\n{ai_code}"

    # ---------------- EXECUTE ----------------
    try:
        exec_locals = {df_label: df}
        exec(ai_code, {}, exec_locals)
        result = exec_locals["result"]

        if isinstance(result, (pd.DataFrame, pd.Series)) and result.empty:
            return f"⚠ No matching rows.\n\nCODE:\n{ai_code}"

    except Exception as e:
        return f"❌ Error executing AI code: {e}\n\nCODE:\n{ai_code}"

    # ---------------- FORMAT RESULT ----------------
    result_text = (
        result.to_string() if isinstance(result,(pd.DataFrame,pd.Series))
        else str(result)
    )

    # ---------------- EXPLAIN ----------------
    explain_prompt = f"""
Explain clearly:

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

question = st.text_input("Ask your question (case-insensitive):")

if st.button("Ask"):
    if question.strip():
        st.write(ask_sheet(question, df_name))
    else:
        st.warning("Enter a question first.")
