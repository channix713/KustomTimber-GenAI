# app.py (Standard Optimized Version for Streamlit Cloud)
"""
Optimized Streamlit app:
- Faster Google Sheets loading
- Cleaner structure
- Safer OpenAI prompt formatting
- Reduced token usage
- Better error handling
- Fixed sheet ID
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
# Load OpenAI Key (Secrets > .env fallback)
# -----------------------
try:
    ENV_PATH = Path(__file__).parent / "OpenAIKey.env"
except NameError:
    ENV_PATH = Path(os.getcwd()) / "OpenAIKey.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -----------------------
# Google Service Account (from Secrets)
# -----------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def get_google_creds():
    try:
        info = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
        return Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception as e:
        st.error(f"Google credentials error: {e}")
        raise

# -----------------------
# Fixed Spreadsheet ID
# -----------------------
FIXED_SPREADSHEET_KEY = "1oeFZRrqr7YI52L5EAl972ghesCsBYcPiCOi09n_mCgY"

st.set_page_config(page_title="Kustom Timber Stock AI Assistant", layout="wide")

# -----------------------
# Load worksheet
# -----------------------
@st.cache_data(ttl=300)
def load_fixed_sheet(worksheet_name: str = "Sheet1") -> pd.DataFrame:
    creds = get_google_creds()
    gs = gspread.authorize(creds)

    sheet = gs.open_by_key(FIXED_SPREADSHEET_KEY).worksheet(worksheet_name)
    raw = sheet.get_all_values()
    if not raw:
        return pd.DataFrame()

    headers = raw[0]

    # Fix duplicate headers
    unique_headers = []
    for h in headers:
        if h not in unique_headers:
            unique_headers.append(h)
        else:
            count = sum(x.startswith(h) for x in unique_headers)
            unique_headers.append(f"{h}_{count}")

    df = pd.DataFrame(raw[1:], columns=unique_headers)

    # Auto numeric conversion
    for col in df.columns:
    # Skip columns that clearly contain text with letters
    if df[col].astype(str).str.contains(r"[A-Za-z]", regex=True).any():
        continue

    cleaned = df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    numeric_count = cleaned.str.match(r"^-?\d+(\.\d+)?$").sum()

    # Convert only if MOST of the rows are numeric-like
    if numeric_count > len(df[col]) * 0.5:
        df[col] = pd.to_numeric(cleaned, errors="coerce")

    return df

# -----------------------
# Ask model (Optimized)
# -----------------------

def ask_data_question_full(question: str, df: pd.DataFrame, model_name: str = "gpt-4.1") -> str:
    if client is None:
        return "Error: OpenAI client not configured. Add OPENAI_API_KEY to secrets."

    if df.empty:
        return "Dataset is empty. Cannot analyze."

    df_trim = df.head(300)
    col_map = {col: f"c{i}" for i, col in enumerate(df_trim.columns)}

    short_df = df_trim.rename(columns=col_map)
    full_data = short_df.to_dict(orient="records")

    system_prompt = (
        "You are a senior data analyst AI. Use ONLY the provided data for calculations. "
        "Translate shorthand columns using the map provided. Be precise and concise."
    )

    user_prompt = (
        f"COLUMN MAP (short -> original):\n{json.dumps(col_map, indent=2)}\n\n"
        f"DATASET (trimmed to 300 rows):\n{json.dumps(full_data, ensure_ascii=False)}\n\n"
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
        return f"⚠️ AI Error: {e}"

# -----------------------
# Streamlit UI
# -----------------------

st.title("📦 Kustom Timber Stock Inventory – AI Assistant")
st.write("Analyze your Google Sheet instantly using AI.")

with st.sidebar:
    st.header("Settings")
    worksheet_name = st.text_input("Worksheet tab name", value="Sheet1")
    model_choice = st.selectbox("Model", ["gpt-4.1", "gpt-3.5-turbo"], index=0)
    load_btn = st.button("Load Stock Inventory Data")

if load_btn:
    with st.spinner("Loading Google Sheet..."):
        try:
            df_all = load_fixed_sheet(worksheet_name)
            if df_all.empty:
                st.error("No data found. Check worksheet name and access.")
            else:
                st.success(f"Loaded {df_all.shape[0]} rows and {df_all.shape[1]} columns.")
                st.session_state["df_all"] = df_all
        except Exception as e:
            st.error(f"Failed to load sheet: {e}")

# -----------------------
# If data is loaded
# -----------------------
if "df_all" in st.session_state:

    df_all = st.session_state["df_all"]

    st.subheader("🔍 Data Preview (first 200 rows)")
    st.dataframe(df_all.head(200), use_container_width=True)

    st.subheader("💬 Ask a question about your stock data")
    question = st.text_input("Enter your question (e.g., 'total stock for TEL-20211')")

    if st.button("Ask AI") and question.strip():
        with st.spinner("Analyzing..."):
            answer = ask_data_question_full(question, df_all, model_choice)
        st.markdown("### 🤖 AI Answer:")
        st.write(answer)

    st.divider()
    if st.button("Show column names"):
        st.write(list(df_all.columns))

else:
    st.info("Load the stock sheet using the sidebar.")
