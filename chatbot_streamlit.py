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
#  AUTO REFRESH (EVERY 30 SECONDS)
# ================================================================
try:
    st.autorefresh(interval=30000, key="auto_refresh")  # 30 seconds
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
#  LOAD AND CLEAN SHEETS — AUTO REFRESH EVERY 60s
# ================================================================
SPREADSHEET_ID = "1UG_N-zkgwCpObWTgmg8EPS7-N08aqintu8h3kN8yRmM"
WORKSHEET_NAME = "Stock"
WORKSHEET_NAME2 = "Summary"

@st.cache_data(show_spinner=True, ttl=60)  # Auto-refresh cache every 60 seconds
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
    if "Product Code" in stock_df:
        stock_df["Product Code"] = (
            stock_df["Product Code"]
            .str.replace(r"[^0-9]", "", regex=True)
            .str.strip()
        )

    # Convert Date Required → MonthNorm
    date_col = None
    for c in stock_df.columns:
        if "date" in c.lower() and "required" in c.lower():
            date_col = c
            break

    if date_col:
        stock_df[date_col] = pd.to_datetime(stock_df[date_col], errors="coerce")
        stock_df["MonthNorm"] = (
            stock_df[date_col].dt.strftime("%B %Y").str.lower()
        )

    # Packs cleanup
    if "Packs" in stock_df:
        stock_df["Packs_num"] = (
            stock_df["Packs"]
            .astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
        )
        stock_df["Packs_num"] = pd.to_numeric(stock_df["Packs_num"], errors="coerce")

    # Status normalization with invoice aliases
    if "Status" in stock_df:
        def normalize_status(t):
            t = t.lower().strip()
            aliases = ["inv", "invo", "invoice", "invoiced", "invc", "inv.", "invoicing"]
            for a in aliases:
                if t == a or t.startswith(a):
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
        if col in summary_df:
            summary_df[col + "_num"] = summary_df[col].str.replace(r"[^0-9.]", "", regex=True)
            summary_df[col + "_num"] = pd.to_numeric(summary_df[col + "_num"], errors="coerce")

    return stock_df, summary_df


# ================================================================
#  MANUAL REFRESH BUTTON
# ================================================================
st.write(" ")
if st.button("🔄 Refresh Google Sheets Now"):
    st.cache_data.clear()
    st.success("Google Sheets data refreshed!")
    st.experimental_rerun()


# ================================================================
#  AI QUERY ENGINE
# ================================================================
def ai_query(df, question):
    cols = list(df.columns)

    prompt = f"""
Convert the user's question into a single Python expression using ONLY 
the DataFrame named df.

Columns available:
{cols}

Rules:
- Use df["column"] syntax.
- Use *_num numeric columns.
- Use MonthNorm for month filtering.
- Use Status for status filtering.
- Return ONLY code. No markdown or explanation.
- Must return a scalar, Series, or DataFrame.

User question:
{question}
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        ai_code = resp.choices[0].message.content.strip()
    except Exception as e:
        return None, None, f"OpenAI error: {e}"

    try:
        result = eval(ai_code, {"df": df, "pd": pd, "np": np}, {})
    except Exception as e:
        return ai_code, None, f"Execution error: {e}"

    if isinstance(result, (pd.DataFrame, pd.Series)):
        result_text = result.to_string()
    else:
        result_text = str(result)

    return ai_code, result_text, None


# ================================================================
#  STREAMLIT UI
# ================================================================
st.title("📦 Inventory Chatbot — Live Updating")
st.caption("Auto-refresh enabled. Google Sheets updates appear every 30–60s.")

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
    st.subheader("Rows for Product Code 20373")

    debug_rows = stock_df[stock_df["Product Code"] == "20373"]
    st.dataframe(debug_rows)

    st.write("MonthNorm values:", debug_rows["MonthNorm"].unique())
    st.write("Status values:", debug_rows["Status"].unique())

    st.write("Rows where MonthNorm == 'november 2025'")
    st.dataframe(debug_rows[debug_rows["MonthNorm"] == "november 2025"])

    st.write("Rows where Status == 'invoiced'")
    st.dataframe(debug_rows[debug_rows["Status"] == "invoiced"])

    st.write("Rows matching ALL conditions:")
    st.dataframe(
        debug_rows[
            (debug_rows["MonthNorm"] == "november 2025") &
            (debug_rows["Status"] == "invoiced")
        ]
    )


# ================================================================
#  QUESTION INPUT + AI EXECUTION
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
st.caption("🔐 Secure Secrets | 🔄 Auto-Refresh | ✔ Live Google Sheet Sync")
