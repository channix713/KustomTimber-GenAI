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

# --------- CONFIG ----------
SERVICE_ACCOUNT_FILE = "credentials.json"
SPREADSHEET_ID = "1UG_N-zkgwCpObWTgmg8EPS7-N08aqintu8h3kN8yRmM"
WORKSHEET_NAME = "Stock"
WORKSHEET_NAME2 = "Summary"
MODEL = "gpt-4.1-mini"   # change if you prefer another model
# --------------------------

st.set_page_config(page_title="Sheets Chatbot (Streamlit)", layout="wide")

@st.cache_data(show_spinner=False)
def load_sheets():
    # Authenticate Google
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly"
    ]
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    gc = gspread.authorize(creds)
    drive_service = build("drive", "v3", credentials=creds)

    # Load stock
    stock_ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    stock_rows = stock_ws.get_all_values()
    stock_df = pd.DataFrame.from_records(stock_rows)
    stock_df.columns = stock_df.iloc[0]
    stock_df = stock_df[1:].reset_index(drop=True)
    stock_df = stock_df.replace(r'^\s*$', np.nan, regex=True)
    stock_df = stock_df.dropna(how='all')

    # Load summary
    summary_ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME2)
    summary_rows = summary_ws.get_all_values()
    summary_df = pd.DataFrame.from_records(summary_rows)
    summary_df.columns = summary_df.iloc[0]
    summary_df = summary_df[1:].reset_index(drop=True)
    summary_df = summary_df.replace(r'^\s*$', np.nan, regex=True)
    summary_df = summary_df.dropna(how='all')

    return stock_df, summary_df

def clean_numeric_column(df, col):
    if col in df.columns:
        df[col + "_num"] = (
            df[col].astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)
        )
        df[col + "_num"] = pd.to_numeric(df[col + "_num"], errors="coerce")

@st.cache_data(show_spinner=False)
def prepare_data():
    stock_df, summary_df = load_sheets()

    # Clean numeric columns
    if "Packs" in stock_df.columns:
        clean_numeric_column(stock_df, "Packs")
    for col in ["ORDERED", "AVAILABLE", " SOH COST ", " SOO COST "]:
        if col in summary_df.columns:
            clean_numeric_column(summary_df, col)

    # Trim join keys and merge
    stock_df["Product Code"] = stock_df["Product Code"].astype(str).str.strip()
    summary_df["ITEM #"] = summary_df["ITEM #"].astype(str).str.strip()

    merged_df = pd.merge(
        stock_df,
        summary_df,
        left_on="Product Code",
        right_on="ITEM #",
        how="left",
        suffixes=("_stock", "_summary")
    )

    return stock_df, summary_df, merged_df

# Initialize OpenAI client
def get_openai_client():
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        st.error("OpenAI API key not found. Set environment variable OPENAI_API_KEY.")
        return None
    return OpenAI(api_key=openai_key)

# Safe product-code direct lookup (no AI)
def direct_product_lookup(merged_df, product_code):
    pc = str(product_code).strip()
    row = merged_df.loc[merged_df["Product Code"] == pc]
    if row.empty:
        return None, f"No rows for Product Code {pc}"
    # Prefer AVAILABLE_num if present, else AVAILABLE text
    if "AVAILABLE_num" in merged_df.columns:
        val = row["AVAILABLE_num"].iloc[0]
        if pd.isna(val):
            # fallback to raw AVAILABLE
            val_raw = row.get("AVAILABLE", pd.Series([np.nan])).iloc[0]
            return val_raw, f"AVAILABLE_num is NaN; raw AVAILABLE = {val_raw}"
        return val, None
    else:
        val_raw = row.get("AVAILABLE", pd.Series([np.nan])).iloc[0]
        return val_raw, None

# General AI-assisted query (falls back to safe patterns)
def ai_query(merged_df, question, client):
    # Build schema-aware prompt
    merged_columns = list(merged_df.columns)
    interpretation_prompt = f"""
You are a data planner. Convert the user's question into VALID Python code that runs on a pandas DataFrame named merged_df.

merged_df columns:
{merged_columns}

RULES:
- Use ONLY the exact column names listed above.
- For numeric work, use *_num columns (e.g. AVAILABLE_num, ORDERED_num, Packs_num).
- For questions about product codes, filter with merged_df["Product Code"].
- OUTPUT ONLY a single line (or expression) of valid Python that returns a pandas object or scalar.
- NO commentary, NO markdown, NO explanations.

USER QUESTION:
{question}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": interpretation_prompt}]
        )
    except Exception as e:
        return None, None, f"OpenAI call error: {e}"

    ai_code = response.choices[0].message.content

    # Execute safely (local_vars only provides merged_df)
    local_vars = {"merged_df": merged_df, "pd": pd, "np": np}
    try:
        result = eval(ai_code, {}, local_vars)
    except Exception as e:
        return ai_code, None, f"Error executing AI code: {e}"

    # Convert result to readable text
    if isinstance(result, (pd.DataFrame, pd.Series)):
        result_text = result.to_string()
    else:
        result_text = str(result)

    # Ask AI to explain result
    explain_prompt = f"""
Explain this result to the user clearly and concisely.

RESULT:
{result_text}

Original question:
{question}
"""
    try:
        explain_resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": explain_prompt}]
        )
        explanation = explain_resp.choices[0].message.content
    except Exception as e:
        explanation = f"(Could not get explanation from OpenAI: {e})"

    return ai_code, result_text, explanation

# ----------------- Streamlit UI -----------------
st.title("📊 Google Sheets Inventory Chatbot (Streamlit)")

with st.sidebar:
    st.markdown("### Configuration")
    st.write("Model:", MODEL)
    st.markdown("Place your OpenAI API key in environment variable `OPENAI_API_KEY`.")
    st.markdown("Google service account credentials: `credentials.json`")

st.info("Loading data from Google Sheets... (cached)")
stock_df, summary_df, merged_df = prepare_data()
st.success("Data loaded and merged.")

# Show small preview
if st.checkbox("Show merged dataframe (preview)"):
    st.dataframe(merged_df.head(200))

client = get_openai_client()

# Input area
st.markdown("## Ask about your inventory")
user_question = st.text_input("Type your question (e.g. How many Available for product code 20339?)")

col1, col2 = st.columns([1,3])
with col1:
    if st.button("Ask") and user_question:
        # detect direct product code queries: look for "product code" + digits
        pc_match = re.search(r"product\s*code\s*[:#]?\s*(\d+)", user_question, re.IGNORECASE)
        if pc_match:
            code = pc_match.group(1)
            value, note = direct_product_lookup(merged_df, code)
            if value is None:
                st.warning(f"No match for Product Code {code}.")
                # fall back to AI if desired
                if client:
                    ai_code, result_text, explanation = ai_query(merged_df, user_question, client)
                    if ai_code:
                        st.subheader("AI-generated code")
                        st.code(ai_code, language="python")
                    if result_text:
                        st.subheader("Result")
                        st.text(result_text)
                    if explanation:
                        st.subheader("Explanation")
                        st.write(explanation)
                else:
                    st.error("No OpenAI client to fall back.")
            else:
                st.subheader(f"Product Code {code} — AVAILABLE")
                st.write(value)
                if note:
                    st.caption(note)
        else:
            # general question -> use AI path
            if client is None:
                st.error("OpenAI client not available. Set OPENAI_API_KEY to use AI features.")
            else:
                ai_code, result_text, explanation = ai_query(merged_df, user_question, client)
                if ai_code:
                    st.subheader("AI-generated code")
                    st.code(ai_code, language="python")
                if result_text:
                    st.subheader("Result")
                    st.text(result_text)
                if explanation:
                    st.subheader("Explanation")
                    st.write(explanation)

with col2:
    st.markdown("### Examples")
    st.write("- How many Available for product code 20339?")
    st.write("- Which product has the lowest Packs_num?")
    st.write("- Show items with AVAILABLE_num < 10")
    st.write("- What is the total ORDERED?")

st.markdown("---")
st.caption("Notes: product-code queries use a direct lookup (no AI). Other queries use OpenAI to generate Python expressions executed on the merged dataframe.")

