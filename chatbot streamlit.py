#!/usr/bin/env python
# coding: utf-8

# In[20]:


# app.py
import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2 import service_account
from googleapiclient.discovery import build
from openai import OpenAI

# =====================================
# STREAMLIT SECRETS
# =====================================
OPENAI_KEY = st.secrets["openai"]["api_key"]
GCP_CREDS = st.secrets["gcp_service_account"]  # full service account JSON

SERVICE_ACCOUNT_INFO = dict(GCP_CREDS)

# -------------------------------------
# GOOGLE SHEETS CONFIG
# -------------------------------------
SPREADSHEET_ID = "1UG_N-zkgwCpObWTgmg8EPS7-N08aqintu8h3kN8yRmM"
WORKSHEET_NAME = "Stock"
WORKSHEET_NAME2 = "Summary"

# -------------------------------------
# OPENAI CONFIG
# -------------------------------------
MODEL = "gpt-4.1-mini"
client = OpenAI(api_key=OPENAI_KEY)

# -------------------------------------
# GOOGLE AUTH
# -------------------------------------
def google_auth():
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly"
    ]
    creds = service_account.Credentials.from_service_account_info(
        SERVICE_ACCOUNT_INFO,
        scopes=SCOPES
    )
    gc = gspread.authorize(creds)
    drive_service = build("drive", "v3", credentials=creds)
    return gc, drive_service


# ============================================================
# Load Sheets
# ============================================================
@st.cache_data(show_spinner=True)
def load_sheets():
    gc, drive = google_auth()

    # STOCK
    stock_ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    stock_rows = stock_ws.get_all_values()
    stock_df = pd.DataFrame(stock_rows)
    stock_df.columns = stock_df.iloc[0]
    stock_df = stock_df[1:].reset_index(drop=True)
    stock_df = stock_df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")

    # SUMMARY
    summary_ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME2)
    summary_rows = summary_ws.get_all_values()
    summary_df = pd.DataFrame(summary_rows)
    summary_df.columns = summary_df.iloc[0]
    summary_df = summary_df[1:].reset_index(drop=True)
    summary_df = summary_df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")

    return stock_df, summary_df


def clean_numeric(df, col):
    """Turn things like '25.99m2' into 25.99"""
    df[col + "_num"] = (
        df[col]
        .astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    df[col + "_num"] = pd.to_numeric(df[col + "_num"], errors="coerce")


# ============================================================
# Prepare Merged Data
# ============================================================
@st.cache_data(show_spinner=True)
def prepare_data():
    stock_df, summary_df = load_sheets()

    # Clean numeric columns
    if "Packs" in stock_df.columns:
        clean_numeric(stock_df, "Packs")

    for col in ["ORDERED", "AVAILABLE", " SOH COST ", " SOO COST "]:
        if col in summary_df.columns:
            clean_numeric(summary_df, col)

    # Fix join keys (string+strip)
    stock_df["Product Code"] = stock_df["Product Code"].astype(str).strip()
    summary_df["ITEM #"] = summary_df["ITEM #"].astype(str).strip()

    # Merge both sheets
    merged_df = pd.merge(
        stock_df,
        summary_df,
        left_on="Product Code",
        right_on="ITEM #",
        how="left",
        suffixes=("_stock", "_summary")
    )

    return stock_df, summary_df, merged_df


# ============================================================
# Direct lookup for product-code questions
# ============================================================
def direct_lookup(merged_df, code):
    code = str(code).strip()
    row = merged_df.loc[merged_df["Product Code"] == code]

    if row.empty:
        return None, "No matching product code."

    if "AVAILABLE_num" in merged_df.columns:
        val = row["AVAILABLE_num"].iloc[0]
        if pd.isna(val):
            raw = row["AVAILABLE"].iloc[0]
            return raw, f"Numeric unavailable; raw AVAILABLE = {raw}"
        return val, None

    return row["AVAILABLE"].iloc[0], None


# ============================================================
# General AI query (fallback)
# ============================================================
def ai_query(merged_df, question):
    merged_columns = list(merged_df.columns)

    prompt = f"""
You are a data analyst. Convert the user's question into valid Python code.
The code must run on a pandas DataFrame named merged_df.

VALID COLUMNS:
{merged_columns}

RULES:
- Use only the column names above.
- For numeric work use *_num columns.
- For product code questions, filter with merged_df["Product Code"].
- Return ONLY executable Python code (no markdown, no comments).

Question:
{question}
"""

    ai_resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    ai_code = ai_resp.choices[0].message.content

    # Execute code
    try:
        local_vars = {"merged_df": merged_df, "pd": pd, "np": np}
        result = eval(ai_code, {}, local_vars)
    except Exception as e:
        return ai_code, None, f"Execution error: {e}"

    # Format result
    result_text = result.to_string() if isinstance(result, (pd.DataFrame, pd.Series)) else str(result)

    # Explanation
    expl_prompt = f"""
Explain this result clearly:

Result:
{result_text}

Question:
{question}
"""
    ai_expl = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": expl_prompt}]
    ).choices[0].message.content

    return ai_code, result_text, ai_expl


# ============================================================
# STREAMLIT UI
# ============================================================
st.title("📊 Streamlit Inventory Chatbot (Google Sheets + OpenAI)")

st.info("Loading data…")
stock_df, summary_df, merged_df = prepare_data()
st.success("Data loaded!")

if st.checkbox("Show merged dataframe preview"):
    st.dataframe(merged_df.head(200))


# USER INPUT
question = st.text_input("Ask about inventory:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Enter a question.")
    else:
        # Detect product-code queries
        code_match = re.search(r"product\s*code\s*[:#]?\s*(\d+)", question, re.I)
        if code_match:
            code = code_match.group(1)
            val, note = direct_lookup(merged_df, code)
            st.subheader(f"Product Code {code} — AVAILABLE")
            st.write(val)
            if note:
                st.caption(note)

        else:
            # AI fallback
            ai_code, result_text, explanation = ai_query(merged_df, question)

            st.subheader("AI-Generated Code")
            st.code(ai_code, language="python")

            st.subheader("Result")
            st.text(result_text)

            st.subheader("Explanation")
            st.write(explanation)


