# ======================================================================
# STREAMLIT GOOGLE SHEETS CHATBOT (Stock + Summary)
# HYBRID JSON PLANNER VERSION (no exec, no eval)
# With:
#  - Automatic Status detection (Invoiced / Shipped / Landed / Ordered)
#  - Ask user to specify Status if missing
#  - Refactored utils, safer planner, better cache & refresh
#  - Fully rewritten modern chat UI (fixed NameError)
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
    "jan": "January", "january": "January",
    "feb": "February", "february": "February",
    "mar": "March", "march": "March",
    "apr": "April", "april": "April",
    "may": "May",
    "jun": "June", "june": "June",
    "jul": "July", "july": "July",
    "aug": "August", "august": "August",
    "sep": "September", "sept": "September", "september": "September",
    "oct": "October", "october": "October",
    "nov": "November", "november": "November",
    "dec": "December", "december": "December",
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
    st.error("❌ Missing OPENAI_API_KEY in secrets or environment")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================================================
# GOOGLE CREDS
# ======================================================================
def get_google_creds() -> Credentials:
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
# NORMALIZATION HELPERS
# ======================================================================
def normalize_month_string(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None
    raw = text.strip().lower()
    if not raw:
        return None
    raw = re.sub(r"[-_/.,]+", " ", raw)
    parts = raw.split()

    if len(parts) == 2:
        m, y = parts
        if m in MONTH_MAP and re.match(r"^\d{4}$", y):
            return f"{MONTH_MAP[m]} {y}"

    m = re.match(r"^(\d{1,2})\s+(\d{4})$", raw)
    if m:
        mm = int(m.group(1))
        y = m.group(2)
        if 1 <= mm <= 12:
            return f"{MONTH_NAMES[mm-1]} {y}"

    m2 = re.match(
        r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})$",
        raw,
    )
    if m2:
        return f"{m2.group(1).capitalize()} {m2.group(2)}"

    return None


def normalize_status(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    t = raw.strip().lower()
    if not t:
        return ""
    for canonical, keywords in STATUS_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return canonical
    return raw.strip().title()


def detect_status_from_question(question: str) -> Optional[str]:
    text = question.lower()
    hits = set()
    for canonical, keywords in STATUS_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(kw)}\b", text) for kw in keywords):
            hits.add(canonical)
    if len(hits) == 0:
        return None
    if len(hits) > 1:
        return "MULTI"
    return list(hits)[0]


# ======================================================================
# REFRESH (CACHE CLEAR)
# ======================================================================
def clear_sheet_cache_and_rerun():
    load_sheets.clear()  # type: ignore
    st.success("🔄 Sheets refreshed!")
    st.experimental_rerun()

# ======================================================================
# LOAD SHEETS (CACHED)
# ======================================================================
@st.cache_data(show_spinner=True)
def load_sheets():
    sh = gc.open_by_key(SPREADSHEET_ID)

    # STOCK
    ws = sh.worksheet(WS_STOCK)
    df1 = pd.DataFrame(ws.get_all_values())
    df1.columns = df1.iloc[0].str.strip()
    df1 = df1[1:].reset_index(drop=True)
    df1.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    if "Product Code" in df1:
        df1["Product Code"] = pd.to_numeric(df1["Product Code"], errors="coerce")

    if "PACKS" in df1:
        df1["PACKS"] = pd.to_numeric(df1["PACKS"], errors="coerce").fillna(0)

    if "Month" in df1:
        df1["Month"] = df1["Month"].astype(str).str.strip()

    if "Status" in df1:
        df1["Status"] = df1["Status"].apply(normalize_status)

    # SUMMARY
    ws2 = sh.worksheet(WS_SUMMARY)
    df2 = pd.DataFrame(ws2.get_all_values())
    df2.columns = df2.iloc[0].str.strip()
    df2 = df2[1:].reset_index(drop=True)
    df2.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    for col in SUMMARY_NUMERIC_COLS:
        if col in df2:
            df2[col] = pd.to_numeric(df2[col], errors="coerce").fillna(0)

    if "ITEM #" in df2:
        df2["ITEM #"] = pd.to_numeric(df2["ITEM #"], errors="coerce")

    df2["AVAILABLE_num"] = pd.to_numeric(df2.get("AVAILABLE", np.nan), errors="coerce").fillna(0)

    return df1, df2


stock_df, summary_df = load_sheets()

# ======================================================================
# BUILD FILTERS + APPLY PLAN
# ======================================================================
def build_condition(df, col, op, val):
    s = df[col]
    if op == "==":
        return s == val
    if op == "!=":
        return s != val
    if op == "<":
        return s < val
    if op == "<=":
        return s <= val
    if op == ">":
        return s > val
    if op == ">=":
        return s >= val
    if op == "contains":
        return s.astype(str).str.contains(str(val), case=False, na=False)
    raise ValueError(f"Unsupported operator: {op}")


def apply_plan(plan, df, df_name):
    if not isinstance(plan, dict):
        return None, "❌ Invalid plan format."

    filters = plan.get("filters", [])
    metric = plan.get("metric")
    agg = plan.get("aggregation", "rows")
    limit = plan.get("limit")

    mask = pd.Series(True, index=df.index)

    for f in filters:
        col, op, val = f["column"], f.get("op", "=="), f.get("value")

        if col == "Month" and isinstance(val, str):
            val = normalize_month_string(val) or val

        if col == "Status":
            val = normalize_status(val)

        cond = build_condition(df, col, op, val)
        mask &= cond

    filtered = df[mask].copy()

    if filtered.empty:
        return filtered, "⚠ No matching rows found."

    if metric and metric in df.columns:
        s = filtered[metric]
        if agg == "sum":
            return s.sum(), None
        if agg == "max":
            return s.max(), None
        if agg == "min":
            return s.min(), None
        if agg == "list":
            return s.tolist(), None

    if isinstance(limit, int) and limit > 0:
        return filtered.head(min(limit, 500)), None

    return filtered, None


# ======================================================================
# GPT PLANNER
# ======================================================================
def build_planner_prompt(question, df_name, df, status):
    cols = list(df.columns)

    if df_name == "stock":
        id_col = "Product Code"
        metric_col = "PACKS"
        month_col = "Month"
        status_col = "Status"
    else:
        id_col = "ITEM #"
        metric_col = "AVAILABLE_num"
        month_col = None
        status_col = None

    prompt = f"""
You are a strict JSON query planner for ONE DataFrame.

COLUMNS: {cols}

RULES:
- ONLY produce valid JSON.
- Never invent column names.
- For product/item code, filter on "{id_col}".
- For 'how many' or totals, use metric "{metric_col}" unless user specifies another numeric column.
- If a month is mentioned and 'Month' exists, include it.
- For stock_df, if a status is detected, include it as a filter.

Detected Status: {status}

User Question:
{question}

Return ONLY a JSON object, no explanations.
"""

    return prompt


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
    except:
        return None


# ======================================================================
# ANSWER QUESTION
# ======================================================================
def answer_question(question, df_name):
    df = stock_df if df_name == "stock" else summary_df

    detected_status = None
    if df_name == "stock":
        detected_status = detect_status_from_question(question)

        if detected_status is None:
            return (
                "⚠ I couldn't detect a status. Please include one: "
                "**Invoiced**, **Shipped**, **Landed**, **Ordered**.",
                None,
                None,
            )

        if detected_status == "MULTI":
            return (
                "⚠ Multiple statuses detected. Please specify only ONE.",
                None,
                None,
            )

    plan = get_plan(question, df_name, detected_status)
    if plan is None:
        return "❌ Unable to create query plan.", None, None

    result, err = apply_plan(plan, df, df_name)
    if err:
        return err, plan, result

    # Build explanation
    preview = (
        result.head(15).to_string() if isinstance(result, pd.DataFrame)
        else str(result)
    )

    explain_prompt = f"""
Explain this result to a non-technical user:

User Question:
{question}

Plan:
{json.dumps(plan)}

Result Preview:
{preview}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": explain_prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip(), plan, result


# ======================================================================
# UI (FIXED — Modern Chat Interface)
# ======================================================================

# Ensure chat history exists (CORRECT LOCATION)
if "history" not in st.session_state:
    st.session_state.history = []

if "preset_question" not in st.session_state:
    st.session_state.preset_question = ""


# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.header("⚙️ Settings")

    sheet_choice = st.radio(
        "Select Sheet:",
        ["Stock Sheet (stock_df)", "Summary Sheet (summary_df)"]
    )

    df_name = "stock" if sheet_choice.startswith("Stock") else "summary"
    df_selected = stock_df if df_name == "stock" else summary_df

    if st.button("🔄 Refresh Sheets Now"):
        clear_sheet_cache_and_rerun()

    debug_mode = st.checkbox("Show Debug Info")
    show_preview = st.checkbox("Show DataFrame Preview")

    if show_preview:
        with st.expander("📄 DataFrame Preview"):
            st.dataframe(df_selected, use_container_width=True)


# ================================
# HEADER
# ================================
st.markdown("### 💬 Inventory Chatbot (Google Sheets-Powered)")

if df_name == "stock":
    st.info(
        "Queries on **stock_df** must include exactly ONE status:\n"
        "**Invoiced**, **Shipped**, **Landed**, or **Ordered**."
    )


# ================================
# EXAMPLE QUESTIONS
# ================================
with st.expander("📌 Example questions"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ordered packs for 20373 (Nov 2025)"):
            st.session_state.preset_question = (
                "How many Ordered packs for 20373 for November 2025?"
            )
        if st.button("Landed packs for 20588 (Sep 2025)"):
            st.session_state.preset_question = (
                "How many Landed packs for 20588 for September 2025?"
            )

    with col2:
        if st.button("Invoiced packs for 20373 (Nov 2025)"):
            st.session_state.preset_question = (
                "How many Invoiced packs for 20373 for November 2025?"
            )
        if st.button("Available for item 20246"):
            st.session_state.preset_question = (
                "How many AVAILABLE for ITEM # 20246?"
            )


# ================================
# DISPLAY CHAT HISTORY
# ================================
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)


# ================================
# INPUT FIELD
# ================================
question = st.chat_input(
    "Ask your question...",
    placeholder=(
        "e.g., How many Landed packs for 20373 for September 2025?"
        if df_name == "stock" else
        "e.g., How many AVAILABLE for ITEM # 20373?"
    ),
)

if not question:
    # Use preset if user clicked an example button
    question = st.session_state.preset_question
    st.session_state.preset_question = ""


# ================================
# PROCESS QUESTION
# ================================
if question:
    st.session_state.history.append(("user", question))

    explanation, plan, result = answer_question(question, df_name)

    st.session_state.history.append(("assistant", explanation))
    st.chat_message("assistant").write(explanation)

    if debug_mode and plan is not None:
        with st.expander("🛠 Debug Info"):
            st.json(plan)
            if isinstance(result, (pd.DataFrame, pd.Series)):
                st.dataframe(result)
            else:
                st.write(result)

    st.experimental_rerun()
