# app._v2.py
"""
Kustom Timber Inventory — Full Optimized Streamlit App (app._v2.py)
Features:
- Fixed spreadsheet ID (Streamlit Secrets for credentials)
- Safe OpenAI usage (Streamlit Secrets)
- Smart pre-filtering to reduce token usage
- Built-in analytics: sum, average, count, group-by
- Column filtering + global search
- Export to CSV / JSON / Excel
- Caching for sheet loads and model results
- Clear error handling and helpful messages
"""

import io
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import gspread
import streamlit as st
from google.oauth2.service_account import Credentials
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------
# Configuration
# -----------------------
# Replace with your fixed spreadsheet key (already set earlier)
FIXED_SPREADSHEET_KEY = "1oeFZRrqr7YI52L5EAl972ghesCsBYcPiCOi09n_mCgY"

# Local .env fallback for development
try:
    ENV_PATH = Path(__file__).parent / "OpenAIKey.env"
except NameError:
    ENV_PATH = Path(os.getcwd()) / "OpenAIKey.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

# -----------------------
# Init clients
# -----------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# -----------------------
# Helpers: Google creds
# -----------------------

def get_google_creds() -> Credentials:
    """Load Google Service Account info from Streamlit secrets.
    The secret should be a JSON string stored under key GOOGLE_SERVICE_ACCOUNT_JSON.
    """
    try:
        json_str = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
    except Exception as e:
        raise RuntimeError("Google credentials missing in st.secrets (GOOGLE_SERVICE_ACCOUNT_JSON)") from e

    info = json.loads(json_str)
    return Credentials.from_service_account_info(info, scopes=SCOPES)

# -----------------------
# Load sheet (cached)
# -----------------------
@st.cache_data(ttl=300)
def load_sheet(worksheet_name: str = "Sheet1") -> pd.DataFrame:
    """Load and return worksheet as DataFrame. Auto-fixes duplicate headers and tries to convert numeric-like columns."""
    creds = get_google_creds()
    gs = gspread.authorize(creds)
    sh = gs.open_by_key(FIXED_SPREADSHEET_KEY)
    ws = sh.worksheet(worksheet_name)
    raw = ws.get_all_values()
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

    # Try simple numeric conversion
    for col in df.columns:
        s = df[col].astype(str).str.strip()
        cleaned = s.str.replace(r"[^0-9\.-]", "", regex=True)
        numeric_count = cleaned.str.match(r"^-?\d+(?:\.\d+)?$").sum()
        if len(s) and numeric_count > len(s) * 0.5:
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    return df

# -----------------------
# Export helpers
# -----------------------

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buffer.getvalue()

# -----------------------
# Smart filtering for AI (reduce tokens)
# -----------------------

def filter_rows_for_question(df: pd.DataFrame, question: str, max_rows: int = 300) -> pd.DataFrame:
    """Attempt to keep only rows likely relevant to the question by keyword matching.
    Falls back to head(max_rows) if no match found.
    """
    if not question or df.empty:
        return df.head(max_rows)

    # extract candidate keywords (length>2) and unique
    tokens = [t.lower() for t in pd.Series(question.split()).astype(str) if len(t) > 2]
    keywords = list(dict.fromkeys(tokens))[:10]

    if not keywords:
        return df.head(max_rows)

    mask = pd.Series(False, index=df.index)
    for kw in keywords:
        for col in df.columns:
            try:
                mask |= df[col].astype(str).str.contains(kw, case=False, na=False)
            except Exception:
                # non-string column, skip
                continue

    filtered = df[mask]
    if filtered.empty:
        return df.head(max_rows)
    return filtered.head(max_rows)

# -----------------------
# AI query (compact prompts)
# -----------------------
@st.cache_data(ttl=600, show_spinner=False)
def query_ai(question: str, df_snippet: pd.DataFrame, model: str = "gpt-4.1") -> str:
    """Send a compact prompt to the model and return the raw text reply. Cached to avoid repeat costs."""
    if client is None:
        return "Error: OpenAI API key not configured. Add OPENAI_API_KEY to Streamlit secrets."

    if df_snippet is None or df_snippet.empty:
        return "No data available to analyze."

    col_map = {col: f"c{i}" for i, col in enumerate(df_snippet.columns)}
    short_df = df_snippet.rename(columns=col_map)
    rows = short_df.to_dict(orient="records")

    system_prompt = (
        "You are a helpful data analyst. Use ONLY the provided rows to compute answers. "
        "Translate shorthand column names using the column map. Be concise and show calculations when helpful. "
        "If the requested value cannot be computed exactly from the data, say you cannot compute it."
    )

    user_prompt = (
        f"COLUMN MAP:\n{json.dumps(col_map)}\n\n"
        f"ROWS (list of objects):\n{json.dumps(rows, ensure_ascii=False)}\n\n"
        f"QUESTION: {question}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Error: {e}"

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="Kustom Timber Inventory (AI)", layout="wide")
st.title("📦 Kustom Timber Inventory — AI Assistant")
st.markdown("Use the sidebar to load the fixed Google Sheet, filter data, run quick analytics and ask AI questions.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    worksheet_name = st.text_input("Worksheet tab name", value="Sheet1")
    model_choice = st.selectbox("Model", ["gpt-4.1", "gpt-3.5-turbo"], index=0)
    load = st.button("Load sheet")
    st.markdown("---")
    st.caption("Your Google service account JSON and OpenAI key must be stored in Streamlit Secrets.")

# Load sheet
if load:
    with st.spinner("Loading sheet..."):
        try:
            df_loaded = load_sheet(worksheet_name)
            if df_loaded.empty:
                st.error("Sheet loaded but contains no data. Check worksheet name and sharing settings.")
            else:
                st.success(f"Loaded {df_loaded.shape[0]} rows and {df_loaded.shape[1]} columns")
                st.session_state["df_all"] = df_loaded
        except Exception as e:
            st.error(f"Failed to load sheet: {e}")

# Show controls and data when loaded
if "df_all" in st.session_state:
    df_all = st.session_state["df_all"].copy()

    # Top-row actions
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("Filters")

    # Filter by Product Code
    if 'PRODUCT CODE' in df_all.columns:
        product_codes = sorted(df_all['PRODUCT CODE'].dropna().unique())
        selected_product = st.selectbox("Filter by Product Code", ["All"] + product_codes)
        if selected_product != "All":
            df_all = df_all[df_all['PRODUCT CODE'] == selected_product]

    # Filter by Batch Number
    if 'BATCH NUMBER' in df_all.columns:
        batch_numbers = sorted(df_all['BATCH NUMBER'].dropna().unique())
        selected_batch = st.selectbox("Filter by Batch Number", ["All"] + batch_numbers)
        if selected_batch != "All":
            df_all = df_all[df_all['BATCH NUMBER'] == selected_batch]

    st.subheader("🔍 Data Preview")
    with col2:
        if st.button("Reload sheet"):
            try:
                st.session_state["df_all"] = load_sheet(worksheet_name)
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Reload failed: {e}")
    with col3:
        theme = st.selectbox("Theme", ["Light", "Dark"], index=0)

    # Filtering UI
    st.markdown("---")
    st.subheader("Filters & Search")
    c1, c2 = st.columns([2, 3])
    with c1:
        filter_col = st.selectbox("Filter column (optional)", [None] + list(df_all.columns))
        filter_val = None
        if filter_col:
            filter_val = st.text_input("Filter value (regex supported)")
            if filter_val:
                try:
                    df_all = df_all[df_all[filter_col].astype(str).str.contains(filter_val, case=False, na=False)]
                except Exception as e:
                    st.warning(f"Filter failed: {e}")
    with c2:
        global_search = st.text_input("Global search across all columns")
        if global_search:
            df_all = df_all[df_all.apply(lambda r: r.astype(str).str.contains(global_search, case=False, na=False).any(), axis=1)]

    st.dataframe(df_all.head(200), use_container_width=True)

    # Quick analytics
    st.markdown("---")
    st.markdown("---")
    st.subheader("Ask AI about this data")
    ai_question = st.text_input("Type a question for the AI (e.g., 'total stock for TEL-20211')")
    if st.button("Ask AI") and ai_question.strip():
        with st.spinner("Preparing data for AI..."):
            snippet = filter_rows_for_question(df_all, ai_question)
            result = query_ai(ai_question, snippet, model_choice)
            st.markdown("### 🤖 AI Answer")
            st.write(result)

else:
    st.info("Use the sidebar to load the fixed Google Sheet.")
