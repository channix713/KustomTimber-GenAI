# app.py (updated for Streamlit Cloud — loads Google credentials from secrets)
"""
Streamlit app: Google Sheets + Generative AI Data Assistant
- Spreadsheet ID is FIXED
- Google credentials come from Streamlit Secrets
- No credentials.json file needed
"""

import os
import json
from pathlib import Path
from typing import List

import pandas as pd
import gspread
import streamlit as st
from google.oauth2.service_account import Credentials
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------
# Load OpenAI Key (Streamlit Secrets OR local .env for dev)
# -----------------------
try:
    ENV_PATH = Path(__file__).parent / "OpenAIKey.env"
except NameError:
    ENV_PATH = Path(os.getcwd()) / "OpenAIKey.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -----------------------
# Google credentials from secrets
# -----------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def get_google_creds():
    json_str = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(json_str)
    return Credentials.from_service_account_info(info, scopes=SCOPES)

# -----------------------
# Fixed Spreadsheet ID
# -----------------------
FIXED_SPREADSHEET_KEY = "1oeFZRrqr7YI52L5EAl972ghesCsBYcPiCOi09n_mCgY"

st.set_page_config(page_title="Sheets → GenAI Assistant", layout="wide")

# -----------------------
# Load worksheet (fixed sheet)
# -----------------------
@st.cache_data(ttl=300)
def load_fixed_sheet(worksheet_name: str = "Sheet1") -> pd.DataFrame:
    creds = get_google_creds()
    client_gs = gspread.authorize(creds)

    sheet = client_gs.open_by_key(FIXED_SPREADSHEET_KEY).worksheet(worksheet_name)
    raw_values = sheet.get_all_values()
    if not raw_values:
        return pd.DataFrame()

    headers = raw_values[0]
    unique_headers = []
    for h in headers:
        if h not in unique_headers:
            unique_headers.append(h)
        else:
            count = sum(x.startswith(h) for x in unique_headers)
            unique_headers.append(f"{h}_{count}")

    df = pd.DataFrame(raw_values[1:], columns=unique_headers)
    return df

# -----------------------
# Ask model
# -----------------------
def ask_data_question_full(question: str, df: pd.DataFrame, model_name: str = "gpt-4.1") -> str:
    """Ask the model a question about the dataset.
    Keeps prompts compact and safe.
    """
    if client is None:
        return "Error: OpenAI client not configured. Add OPENAI_API_KEY to secrets."

    if df is None or df.empty:
        return "Dataset is empty. Cannot analyze."

    # Limit rows to reduce token usage
    df_trimmed = df.head(300)

    col_map = {col: f"c{i}" for i, col in enumerate(df_trimmed.columns)}
    short_df = df_trimmed.rename(columns=col_map).copy()
    full_data = short_df.to_dict(orient="records")

    system_prompt = (
        "You are a senior data analyst AI. Use ONLY the provided data for calculations. "
        "Map shortened column names back to originals when presenting results. Be concise."
    )

    user_prompt = (
        f"COLUMN MAP (short -> original):\n{json.dumps(col_map, indent=2)}\n\n"
        f"DATASET (trimmed to 300 rows):\n{json.dumps(full_data, ensure_ascii=False)}\n\n"
        f"QUESTION: {question}\n"
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
st.title("📊 Kustom Timber Stock Inventory AI Assistant")

with st.sidebar:
    st.header("Settings")
    worksheet_name = st.text_input("Worksheet tab name", value="Sheet1")
    model_choice = st.selectbox("Model", ["gpt-4.1", "gpt-3.5-turbo"], index=0)
    load_btn = st.button("Load Stock Inventory Data")

if load_btn:
    with st.spinner("Loading sheet..."):
        try:
            df_all = load_fixed_sheet(worksheet_name)
            if df_all.empty:
                st.error("No data loaded. Check worksheet name and sharing settings.")
            else:
                st.success(f"Loaded {len(df_all)} rows, {df_all.shape[1]} columns")
                st.session_state["df_all"] = df_all
        except Exception as e:
            st.error(f"Failed to load sheet: {e}")

if "df_all" in st.session_state:
    df_all = st.session_state["df_all"]

    st.subheader("Preview (first 200 rows)")
    st.dataframe(df_all.head(200))

    st.subheader("Ask a question about the data")
    question = st.text_input("Your question")

    if st.button("Ask") and question.strip():
        with st.spinner("Querying model..."):
            answer = ask_data_question_full(question, df_all, model_choice)
        st.markdown("**🤖 Answer:**")
        st.write(answer)

    st.divider()

    if st.button("Show column names"):
        st.write(list(df_all.columns))
else:
    st.info("Please Load the data in the left pane.")
