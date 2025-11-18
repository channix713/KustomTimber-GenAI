# app.py with Product Code & Batch Number Filters Added

"""
This version includes:
- Product Code filter
- Everything else unchanged
"""

import os
import json
from pathlib import Path
import pandas as pd
import gspread
import streamlit as st
from google.oauth2.service_account import Credentials
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------
# Load OpenAI Key
# -----------------------
try:
    ENV_PATH = Path(__file__).parent / "OpenAIKey.env"
except NameError:
    ENV_PATH = Path(os.getcwd()) / "OpenAIKey.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -----------------------
# Google Service Account
# -----------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def get_google_creds():
    info = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
    return Credentials.from_service_account_info(info, scopes=SCOPES)

# Fixed sheet
FIXED_SPREADSHEET_KEY = "1oeFZRrqr7YI52L5EAl972ghesCsBYcPiCOi09n_mCgY"

st.set_page_config(page_title="Kustom Timber Stock AI Assistant", layout="wide")

# -----------------------
# Load Worksheet
# -----------------------
@st.cache_data(ttl=300)
def load_fixed_sheet(worksheet_name="Sheet1"):
    creds = get_google_creds()
    gs = gspread.authorize(creds)

    sheet = gs.open_by_key(FIXED_SPREADSHEET_KEY).worksheet(worksheet_name)
    raw = sheet.get_all_values()
    if not raw:
        return pd.DataFrame()

    headers = raw[0]
    unique_headers = []
    for h in headers:
        if h not in unique_headers:
            unique_headers.append(h)
        else:
            count = sum(x.startswith(h) for x in unique_headers)
            unique_headers.append(f"{h}_{count}")

    df = pd.DataFrame(raw[1:], columns=unique_headers)

    # SAFE number conversion (preserves product codes)
    for col in df.columns:
        if df[col].astype(str).str.contains(r"[A-Za-z]", regex=True).any():
            continue
        cleaned = df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        numeric_count = cleaned.str.match(r"^-?\d+(?:\.\d+)?$").sum()
        if numeric_count > len(df[col]) * 0.5:
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    return df

# -----------------------
# AI Query
# -----------------------
def ask_data_question_full(question, df, model_name="gpt-4.1"):
    if client is None:
        return "Error: OpenAI key missing."
    if df.empty:
        return "Dataset is empty."

    df_trim = df.head(300)
    col_map = {col: f"c{i}" for i, col in enumerate(df_trim.columns)}
    short_df = df_trim.rename(columns=col_map)
    full_data = short_df.to_dict(orient="records")

    system_prompt = (
        "You are a senior data analyst AI. Use ONLY the provided data. "
        "Translate shorthand columns using the column map."
    )

    user_prompt = (
        f"COLUMN MAP:\n{json.dumps(col_map, indent=2)}\n\n"
        f"DATASET (first 300 rows):\n{json.dumps(full_data, ensure_ascii=False)}\n\n"
        f"QUESTION: {question}"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.15,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {e}"

# -----------------------
# Streamlit UI
# -----------------------
st.title("📦 Kustom Timber Stock Inventory – AI Assistant")

with st.sidebar:
    st.header("Settings")
    worksheet_name = st.text_input("Worksheet tab name", "Sheet1")
    model_choice = st.selectbox("Model", ["gpt-4.1", "gpt-3.5-turbo"], 0)
    load_btn = st.button("Load Stock Inventory Data")

if load_btn:
    with st.spinner("Loading Google Sheet..."):
        try:
            df_all = load_fixed_sheet(worksheet_name)
            if df_all.empty:
                st.error("No data found.")
            else:
                st.success(f"Loaded {df_all.shape[0]} rows.")
                st.session_state["df_all"] = df_all
        except Exception as e:
            st.error(f"Failed: {e}")

# -----------------------
# If data loaded
# -----------------------
if "df_all" in st.session_state:

    df_all = st.session_state["df_all"].copy()

    # FILTERS
    st.subheader("🔍 Filters")

    # Filter by Product Code
    if "PRODUCT CODE" in df_all.columns:
        product_codes = sorted(df_all["PRODUCT CODE"].dropna().unique())
        selected_product = st.selectbox("Filter by Product Code", ["All"] + product_codes)
        if selected_product != "All":
            df_all = df_all[df_all["PRODUCT CODE"] == selected_product]

    # Filter by Batch Number
    if "BATCH NUMBER" in df_all.columns:
        batch_numbers = sorted(df_all["BATCH NUMBER"].dropna().unique())
        selected_batch = st.selectbox("Filter by Batch Number", ["All"] + batch_numbers)
        if selected_batch != "All":
            df_all = df_all[df_all["BATCH NUMBER"] == selected_batch]

    # PREVIEW
    st.subheader("📋 Data Preview (first 200 rows)")
    st.dataframe(df_all, use_container_width=True)

    # AI QUESTION
    st.subheader("💬 Ask a question about your stock data")
    question = st.text_input("Enter question (e.g., 'total stock for TEL-20211')")

    if st.button("Ask AI"):
        with st.spinner("Analyzing..."):
            answer = ask_data_question_full(question, df_all, model_choice)
        st.markdown("### 🤖 AI Answer:")
        st.write(answer)

else:
    st.info("Load sheet data from the sidebar.")
