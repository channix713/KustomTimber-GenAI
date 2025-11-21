# ======================================================================
# STREAMLIT GOOGLE SHEETS CHATBOT (Stock + Summary)
# PRODUCTION-SAFE VERSION (no st.chat_input; chat-style UI with text_input)
# ======================================================================

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import gspread
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from openai import OpenAI

# ======================================================================
# STREAMLIT SETUP
# ======================================================================
st.set_page_config(page_title="Inventory Chatbot", layout="wide")
st.title("📦 Kustom Timber Stock Inventory Chatbot")

# ======================================================================
# CONSTANTS
# ======================================================================
MODEL = "gpt-4.1-mini"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

SPREADSHEET_ID = "1UG_N-zkgwCpObWTgmg8EPS7-N08aqintu8h3kN8yRmM"
WS_STOCK = "Stock"
WS_SUMMARY = "Summary"

CANONICAL_STATUSES = ["Invoiced", "Shipped", "Landed", "Ordered"]

STATUS_KEYWORDS = {
    "Invoiced": ["inv", "invoice", "invoiced"],
    "Shipped": ["ship", "shipped"],
    "Landed": ["land", "landed"],
    "Ordered": ["order", "ordered", "ord"],
}

SUMMARY_NUMERIC_COLS = [
    "COST", "PACK SIZE", "ORDERED", "LANDED", "Shipped", "SOH (DC)",
    "Packs (DC)", "Invoiced", "AVAILABLE", "SOH + SOO", "SOO COST", "SOH COST"
]

MONTH_MAP = {
    s: MONTH
    for MONTH, keys in {
        "January": ["jan", "january"],
        "February": ["feb", "february"],
        "March": ["mar", "march"],
        "April": ["apr", "april"],
        "May": ["may"],
        "June": ["jun", "june"],
        "July": ["jul", "july"],
        "August": ["aug", "august"],
        "September": ["sep", "sept", "september"],
        "October": ["oct", "october"],
        "November": ["nov", "november"],
        "December": ["dec", "december"],
    }.items() for s in keys
}
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# ======================================================================
# LOAD OPENAI KEY
# ======================================================================
try:
    ENV_PATH = Path(__file__).parent / "OpenAIKey.env"
except NameError:
    ENV_PATH = Path(os.getcwd()) / "OpenAIKey.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================================================
# GOOGLE CREDS
# ======================================================================
def get_google_creds() -> Credentials:
    """Load Google service account credentials from Streamlit secrets."""
    if "GOOGLE_SERVICE_ACCOUNT_JSON" not in st.secrets:
        st.error("❌ GOOGLE_SERVICE_ACCOUNT_JSON missing.")
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
# NORMALIZATION HELPERS
# ======================================================================
def normalize_month_string(m: Any) -> Optional[str]:
    """Return canonical 'September 2025' format."""
    if not isinstance(m, str):
        return None
    t = re.sub(r"[-_/.,]+", " ", m.strip().lower())
    parts = t.split()

    # "sep 2025"
    if len(parts) == 2 and parts[0] in MONTH_MAP and parts[1].isdigit():
        return f"{MONTH_MAP[parts[0]]} {parts[1]}"

    # "09 2025"
    if re.match(r"^\d{1,2} \d{4}$", t):
        mm, yyyy = t.split()
        mm = int(mm)
        if 1 <= mm <= 12:
            return f"{MONTH_NAMES[mm - 1]} {yyyy}"

    # "september 2025"
    if len(parts) == 2 and parts[0].capitalize() in MONTH_NAMES and parts[1].isdigit():
        return f"{parts[0].capitalize()} {parts[1]}"

    return None


def normalize_status(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip().lower()
    if not t:
        return ""
    for canonical, words in STATUS_KEYWORDS.items():
        if any(w in t for w in words):
            return canonical
    return s.strip().title()


def detect_status_from_question(q: str) -> Optional[str]:
    found = set()
    text = q.lower()
    for canonical, words in STATUS_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", text) for w in words):
            found.add(canonical)
    if len(found) == 0:
        return None
    if len(found) > 1:
        return "MULTI"
    return list(found)[0]

# ======================================================================
# REFRESH BUTTON
# ======================================================================
def clear_sheet_cache_and_rerun():
    load_sheets.clear()  # type: ignore
    st.success("🔄 Sheets refreshed.")
    st.experimental_rerun()

# ======================================================================
# LOAD SHEETS (CACHED)
# ======================================================================
@st.cache_data(show_spinner=True)
def load_sheets():
    sh = gc.open_by_key(SPREADSHEET_ID)

    # STOCK
    ws = sh.worksheet(WS_STOCK)
    stock_df = pd.DataFrame(ws.get_all_values())
    stock_df.columns = stock_df.iloc[0].str.strip()
    stock_df = stock_df[1:].reset_index(drop=True)
    stock_df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    if "Product Code" in stock_df:
        stock_df["Product Code"] = pd.to_numeric(stock_df["Product Code"], errors="coerce")

    if "PACKS" in stock_df:
        stock_df["PACKS"] = pd.to_numeric(stock_df["PACKS"], errors="coerce").fillna(0)

    if "Month" in stock_df:
        stock_df["Month"] = stock_df["Month"].astype(str).str.strip()

    if "Status" in stock_df:
        stock_df["Status"] = stock_df["Status"].apply(normalize_status)

    # SUMMARY
    ws2 = sh.worksheet(WS_SUMMARY)
    summary_df = pd.DataFrame(ws2.get_all_values())
    summary_df.columns = summary_df.iloc[0].str.strip()
    summary_df = summary_df[1:].reset_index(drop=True)
    summary_df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    for col in SUMMARY_NUMERIC_COLS:
        if col in summary_df:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce").fillna(0)

    if "ITEM #" in summary_df:
        summary_df["ITEM #"] = pd.to_numeric(summary_df["ITEM #"], errors="coerce")

    summary_df["AVAILABLE_num"] = (
        pd.to_numeric(summary_df.get("AVAILABLE", np.nan), errors="coerce").fillna(0)
    )

    return stock_df, summary_df

stock_df, summary_df = load_sheets()

# ======================================================================
# PLANNER + EXECUTION
# ======================================================================
def build_condition(df, col, op, val):
    s = df[col]
    if op == "==": return s == val
    if op == "!=": return s != val
    if op == "<": return s < val
    if op == "<=": return s <= val
    if op == ">": return s > val
    if op == ">=": return s >= val
    if op == "contains":
        return s.astype(str).str.contains(str(val), case=False, na=False)
    raise ValueError(op)


def apply_plan(plan, df, df_name):
    if not isinstance(plan, dict):
        return None, "❌ Invalid plan structure."

    filters = plan.get("filters", [])
    metric = plan.get("metric")
    agg = plan.get("aggregation", "rows")
    limit = plan.get("limit")

    mask = pd.Series(True, index=df.index)

    for f in filters:
        col, op, val = f["column"], f["op"], f.get("value")

        if col == "Month":
            val = normalize_month_string(val) or val
        if col == "Status":
            val = normalize_status(val)

        mask &= build_condition(df, col, op, val)

    out = df[mask]
    if out.empty:
        return out, "⚠ No matching rows found."

    if metric and metric in df.columns:
        series = out[metric]
        if agg == "sum": return series.sum(), None
        if agg == "max": return series.max(), None
        if agg == "min": return series.min(), None
        if agg == "list": return series.tolist(), None

    if isinstance(limit, int) and limit > 0:
        return out.head(min(limit, 500)), None

    return out, None


def build_planner_prompt(question, df_name, df, detected_status):
    cols = list(df.columns)

    if df_name == "stock":
        id_col = "Product Code"
        metric_col = "PACKS"
    else:
        id_col = "ITEM #"
        metric_col = "AVAILABLE_num"

    return f"""
You are a strict JSON query planner for ONE Pandas DataFrame.

COLUMNS = {cols}

RULES:
- Only produce JSON.
- Never invent columns.
- Filter product/item code using '{id_col}'.
- For counts/totals, use metric '{metric_col}' unless user names another numeric column.
- If a month appears and 'Month' exists, include it.
- For stock_df, if a status is detected, include it: {detected_status}

USER QUESTION:
{question}

Return ONLY a JSON object.
"""


@st.cache_data(show_spinner=False)
def get_plan(question, df_name, detected_status):
    df = stock_df if df_name == "stock" else summary_df
    prompt = build_planner_prompt(question, df_name, df, detected_status)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None


def answer_question(question, df_name):
    df = stock_df if df_name == "stock" else summary_df

    # Stock sheet requires explicit status
    if df_name == "stock":
        status = detect_status_from_question(question)
        if status is None:
            return (
                "⚠ Missing status. Please include ONE of: "
                "Invoiced, Shipped, Landed, Ordered.",
                None,
                None,
            )
        if status == "MULTI":
            return "⚠ Multiple statuses detected. Use only ONE.", None, None
    else:
        status = None

    plan = get_plan(question, df_name, status)
    if plan is None:
        return "❌ Failed to generate query plan.", None, None

    result, err = apply_plan(plan, df, df_name)
    if err:
        return err, plan, result

    preview = (
        result.head(15).to_string()
        if isinstance(result, pd.DataFrame)
        else str(result)
    )

    explain_prompt = f"""
Explain the following result to a non-technical user:

USER QUESTION:
{question}

PLAN:
{json.dumps(plan)}

RESULT PREVIEW:
{preview}
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": explain_prompt}],
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip(), plan, result

# ======================================================================
# UI (Chat-style with text_input + form)
# ======================================================================

# Init session state safely
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, message)
if "preset_question" not in st.session_state:
    st.session_state.preset_question = ""

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.header("⚙️ Settings")

    sheet_choice = st.radio(
        "Select Sheet:",
        ["Stock Sheet (stock_df)", "Summary Sheet (summary_df)"],
    )

    df_name = "stock" if sheet_choice.startswith("Stock") else "summary"
    df_selected = stock_df if df_name == "stock" else summary_df

    if st.button("🔄 Refresh Sheets"):
        clear_sheet_cache_and_rerun()

    debug_mode = st.checkbox("Show Debug Info")
    show_preview = st.checkbox("Show DataFrame Preview")

    if show_preview:
        with st.expander("📄 DataFrame Preview"):
            st.dataframe(df_selected, use_container_width=True)

# ================================
# HEADER
# ================================
st.markdown("### 💬 Inventory Chatbot")

if df_name == "stock":
    st.info(
        "On **stock_df**, your question must include exactly ONE status:\n"
        "**Invoiced**, **Shipped**, **Landed**, or **Ordered**."
    )

# ================================
# EXAMPLE QUERIES
# ================================
with st.expander("📌 Example Questions"):
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Ordered packs 20373 (Nov 2025)"):
            st.session_state.preset_question = (
                "How many Ordered packs for 20373 for November 2025?"
            )
        if st.button("Landed packs 20588 (Sep 2025)"):
            st.session_state.preset_question = (
                "How many Landed packs for 20588 for September 2025?"
            )

    with col2:
        if st.button("Invoiced packs 20373 (Nov 2025)"):
            st.session_state.preset_question = (
                "How many Invoiced packs for 20373 for November 2025?"
            )
        if st.button("AVAILABLE for item 20246"):
            st.session_state.preset_question = (
                "How many AVAILABLE for ITEM # 20246?"
            )

# ================================
# CHAT HISTORY DISPLAY
# ================================
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

# ================================
# INPUT FORM (no chat_input; fully compatible)
# ================================
default_question = st.session_state.preset_question

with st.form("chat_form", clear_on_submit=True):
    question = st.text_input(
        "Ask your question:",
        value=default_question,
        placeholder=(
            "e.g., How many Landed packs for 20373 for September 2025?"
            if df_name == "stock"
            else "e.g., How many AVAILABLE for ITEM # 20373?"
        ),
    )
    submitted = st.form_submit_button("Ask")

# Clear preset after rendering it once
st.session_state.preset_question = ""

# ================================
# PROCESS INPUT
# ================================
if submitted and question.strip():
    q = question.strip()

    # Save user message
    st.session_state.history.append(("user", q))

    explanation, plan, result = answer_question(q, df_name)

    # Show assistant response
    st.session_state.history.append(("assistant", explanation))
    st.chat_message("assistant").write(explanation)

    # Debug info
    if debug_mode and plan is not None:
        with st.expander("🛠 Debug Info"):
            st.json(plan)
            if isinstance(result, (pd.DataFrame, pd.Series)):
                st.dataframe(result)
            else:
                st.write(result)
