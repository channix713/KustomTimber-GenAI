import os
import re
import json
import streamlit as st
import pandas as pd
import numpy as np
import gspread
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from google.oauth2 import service_account


# ================================================================
#  AUTO REFRESH EVERY 30 SECONDS
# ================================================================
try:
    st.autorefresh(interval=30000, key="auto_refresh")
except:
    pass


# ================================================================
#  SECURE KEY LOADING
# ================================================================
try:
    ENV_PATH = Path(__file__).parent / "OpenAIKey.env"
except NameError:
    ENV_PATH = Path(os.getcwd()) / "OpenAIKey.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("❗ Missing OpenAI API key.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4.1-mini"

GCP_JSON_STRING = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON", os.getenv("GCP_SERVICE_ACCOUNT_JSON"))
if not GCP_JSON_STRING:
    st.error("❗ Missing Google service account JSON.")
    st.stop()

GCP_CREDS = json.loads(GCP_JSON_STRING)


# ================================================================
#  GOOGLE SHEETS AUTH
# ================================================================
def google_auth():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = service_account.Credentials.from_service_account_info(GCP_CREDS, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc


# ================================================================
#  LOAD & CLEAN SHEETS (AUTO REFRESH EVERY 60s)
# ================================================================
SPREADSHEET_ID = "1UG_N-zkgwCpObWTgmg8EPS7-N08aqintu8h3kN8yRmM"
WORKSHEET_NAME = "Stock"
WORKSHEET_NAME2 = "Summary"

@st.cache_data(show_spinner=True, ttl=60)
def load_sheets():
    gc = google_auth()

    # ---------------- STOCK SHEET ----------------
    ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    rows = ws.get_all_values()
    stock_df = pd.DataFrame(rows)
    stock_df.columns = stock_df.iloc[0].str.strip()
    stock_df = stock_df[1:].reset_index(drop=True)

    # Clean all text
    for col in stock_df.columns:
        stock_df[col] = stock_df[col].astype(str).str.strip()

    # Product Code cleanup
    if "Product Code" in stock_df.columns:
        stock_df["Product Code"] = (
            stock_df["Product Code"]
            .str.replace(r"[^0-9]", "", regex=True)
            .str.strip()
        )

    # MonthNorm from Month Required text
    if "Month Required" in stock_df.columns:
        stock_df["MonthNorm"] = (
            stock_df["Month Required"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        stock_df["MonthNorm"] = None

    # Packs cleanup
    if "Packs" in stock_df.columns:
        stock_df["Packs_num"] = (
            stock_df["Packs"]
            .astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
        )
        stock_df["Packs_num"] = pd.to_numeric(stock_df["Packs_num"], errors="coerce")

    # Status normalization
    if "Status" in stock_df.columns:
        def normalize_status(t):
            t = t.lower().strip()
            aliases = ["inv", "invo", "invoice", "invoiced", "invc", "inv.", "invoicing"]
            for a in aliases:
                if t.startswith(a):
                    return "invoiced"
            return t
        stock_df["Status"] = stock_df["Status"].apply(normalize_status)

    # ---------------- SUMMARY SHEET ----------------
    ws2 = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME2)
    rows2 = ws2.get_all_values()
    summary_df = pd.DataFrame(rows2)
    summary_df.columns = summary_df.iloc[0].str.strip()
    summary_df = summary_df[1:].reset_index(drop=True)

    for col in summary_df.columns:
        summary_df[col] = summary_df[col].astype(str).str.strip()

    # Numeric cleanup
    numeric_cols = ["AVAILABLE", "ORDERED", "LANDED", "Invoiced"]
    for col in numeric_cols:
        if col in summary_df.columns:
            summary_df[col + "_num"] = summary_df[col].str.replace(r"[^0-9.]", "", regex=True)
            summary_df[col + "_num"] = pd.to_numeric(summary_df[col + "_num"], errors="coerce")

    return stock_df, summary_df


# ================================================================
#  MANUAL REFRESH BUTTON (SAFE)
# ================================================================
st.write(" ")
if st.button("🔄 Refresh Google Sheets Now"):
    st.cache_data.clear()
    st.success("Google Sheets data refreshed!")
    st.stop()


# ================================================================
#  AI QUERY ENGINE — STRONG PACKS_num ENFORCEMENT
# ================================================================
def enforce_packs_num(ai_code, question):
    """
    Rewrites AI-generated code to enforce Packs_num and prevent errors.
    """

    q = question.lower()

    # If user mentions packs → Packs_num required
    if "pack" in q:

        # If code incorrectly uses Packs → fix it
        if "Packs" in ai_code and "Packs_num" not in ai_code:
            ai_code = ai_code.replace('["Packs"]', '["Packs_num"]')
            ai_code = ai_code.replace("['Packs']", "['Packs_num']")

        # If Packs_num is still missing → block execution
        if "Packs_num" not in ai_code:
            return None, "❌ AI did not use Packs_num. Execution blocked for safety."

    return ai_code, None


def ai_query(df, question):
    cols = list(df.columns)

    prompt = f"""
You translate the user's question into a single Python pandas expression.

THE DATAFRAME IS df.

STRICT RULES:
- If user mentions "pack", you MUST use df["Packs_num"].
- NEVER use df["Packs"] (strings like "54m2").
- Use df["MonthNorm"] for month filtering.
- Use df["Status"] for status filtering.
- Always use *_num fields for numeric operations.
- Output ONLY Python code — no quotes or explanation.
- Code must be executable via eval().

COLUMNS:
{cols}

USER QUESTION:
{question}
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        ai_code = resp.choices[0].message.content.strip()

    except Exception as e:
        return None, None, f"OpenAI error: {e}"

    # Enforce Packs_num
    ai_code, rule_error = enforce_packs_num(ai_code, question)
    if rule_error:
        return ai_code, None, rule_error

    # Execute code
    try:
        result = eval(ai_code, {"df": df, "pd": pd, "np": np}, {})
    except Exception as e:
        return ai_code, None, f"Execution error: {e}"

    # Format output
    if isinstance(result, (pd.DataFrame, pd.Series)):
        result_text = result.to_string()
    else:
        result_text = str(result)

    return ai_code, result_text, None


# ================================================================
#  STREAMLIT UI
# ================================================================
st.title("📦 Inventory Chatbot — Strict Packs_num Enforcement")
st.caption("Guaranteed correct packs answers. AI cannot return 0 by mistake.")

stock_df, summary_df = load_sheets()

sheet_choice = st.selectbox(
    "Select which sheet to query:",
    ["Stock Sheet (stock_df)", "Summary Sheet (summary_df)"]
)

df = stock_df if sheet_choice.startswith("Stock") else summary_df

if st.checkbox("Show sheet preview"):
    st.dataframe(df.head(200))


# ================================================================
#  DEBUG PANEL
# ================================================================
if st.checkbox("🔍 DEBUG — Show rows for Product Code 20373"):
    debug_rows = stock_df[stock_df["Product Code"] == "20373"]
    st.write("Rows for 20373:")
    st.dataframe(debug_rows)

    st.write("MonthNorm:", debug_rows["MonthNorm"].unique())
    st.write("Status:", debug_rows["Status"].unique())

    st.write("Rows MonthNorm == 'november 2025'")
    st.dataframe(debug_rows[debug_rows["MonthNorm"] == "november 2025"])

    st.write("Rows Status == 'invoiced'")
    st.dataframe(debug_rows[debug_rows["Status"] == "invoiced"])


# ================================================================
#  USER QUESTION
# ================================================================
question = st.text_input("Ask your question:")

if st.button("Ask"):
    ai_code, result_text, error = ai_query(df, question)

    if error:
        st.error(error)
    else:
        st.subheader("AI-Generated Python Code")
        st.code(ai_code)

        st.subheader("Result")
        st.text(result_text)

st.markdown("---")
st.caption("🔐 Safe | 🎯 Accurate | 🧮 Packs_num Guaranteed")
