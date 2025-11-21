# ======================================================================
# STREAMLIT GOOGLE SHEETS CHATBOT (Stock + Summary)
# HYBRID JSON PLANNER VERSION (no exec, no eval)
# With:
#  - Automatic Status detection (Invoiced / Shipped / Landed / Ordered)
#  - Ask user to specify Status if missing (Option 3)
#  - Refactored utils, safer planner, better cache & refresh
# ======================================================================

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
import gspread
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from openai import OpenAI

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
    st.error("❌ Missing OPENAI_API_KEY")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================================================
# GOOGLE CREDS
# ======================================================================
def get_google_creds() -> Credentials:
    if "GOOGLE_SERVICE_ACCOUNT_JSON" not in st.secrets:
        st.error("❌ Missing GOOGLE_SERVICE_ACCOUNT_JSON")
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


def normalize_month_string(user_text: Any) -> Optional[str]:
    if not isinstance(user_text, str):
        return None
    text = user_text.strip().lower()
    if not text:
        return None
    text = re.sub(r"[-_/.,]+", " ", text)
    parts = text.split()
    if len(parts) == 2:
        m_raw, y_raw = parts
        if m_raw in MONTH_MAP and re.match(r"^\d{4}$", y_raw):
            return f"{MONTH_MAP[m_raw]} {y_raw}"
    m = re.match(r"^(\d{1,2})\s+(\d{4})$", text)
    if m:
        mm = int(m.group(1))
        yyyy = m.group(2)
        if 1 <= mm <= 12:
            return f"{MONTH_NAMES[mm - 1]} {yyyy}"
    m2 = re.match(
        r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})$",
        text,
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
    if not isinstance(question, str):
        return None
    text = question.lower()
    found = set()
    for canonical, keywords in STATUS_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                found.add(canonical)
                break
    if len(found) == 0:
        return None
    if len(found) > 1:
        return "MULTI"
    return list(found)[0]


# ======================================================================
# REFRESH (FIXED)
# ======================================================================
def clear_sheet_cache_and_rerun():
    """Correct Streamlit refresh for cached Google Sheets"""
    st.cache_data.clear()  # ✅ clears cache safely
    st.success("🔄 Sheets cache cleared! Reloading…")
    st.rerun()             # ✅ modern rerun API


if st.button("🔄 Refresh Sheets Now"):
    clear_sheet_cache_and_rerun()


# ======================================================================
# LOAD SHEETS (CACHED)
# ======================================================================
@st.cache_data(show_spinner=True)
def load_sheets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    sh = gc.open_by_key(SPREADSHEET_ID)

    ws_stock = sh.worksheet(WS_STOCK)
    stock_df = pd.DataFrame(ws_stock.get_all_values())
    stock_df.columns = stock_df.iloc[0].str.strip()
    stock_df = stock_df[1:].reset_index(drop=True)
    stock_df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    if "Product Code" in stock_df.columns:
        stock_df["Product Code"] = pd.to_numeric(stock_df["Product Code"], errors="coerce")
    if "PACKS" in stock_df.columns:
        stock_df["PACKS"] = pd.to_numeric(stock_df["PACKS"], errors="coerce").fillna(0)
    if "Month" in stock_df.columns:
        stock_df["Month"] = stock_df["Month"].astype(str).str.strip()
    if "Status" in stock_df.columns:
        stock_df["Status"] = stock_df["Status"].apply(normalize_status)

    ws_summary = sh.worksheet(WS_SUMMARY)
    summary_df = pd.DataFrame(ws_summary.get_all_values())
    summary_df.columns = summary_df.iloc[0].str.strip()
    summary_df = summary_df[1:].reset_index(drop=True)
    summary_df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    for col in SUMMARY_NUMERIC_COLS:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce").fillna(0)

    if "ITEM #" in summary_df.columns:
        summary_df["ITEM #"] = pd.to_numeric(summary_df["ITEM #"], errors="coerce")

    summary_df["AVAILABLE_num"] = pd.to_numeric(summary_df.get("AVAILABLE", np.nan), errors="coerce").fillna(0)

    return stock_df, summary_df


stock_df, summary_df = load_sheets()


# ======================================================================
# EXECUTION HELPERS (unchanged)
# ======================================================================
def build_condition(df: pd.DataFrame, column: str, op: str, value: Any) -> pd.Series:
    s = df[column]
    if op == "==": return s == value
    if op == "!=": return s != value
    if op == "<": return s < value
    if op == "<=": return s <= value
    if op == ">": return s > value
    if op == ">=": return s >= value
    if op == "contains": return s.astype(str).str.contains(str(value), case=False, na=False)
    raise ValueError(f"Unsupported operator: {op}")


def apply_plan(plan: Dict[str, Any], df: pd.DataFrame, df_name: str):
    if not isinstance(plan, dict):
        return None, "❌ Invalid plan format."

    filters = plan.get("filters", [])
    metric = plan.get("metric")
    agg = plan.get("aggregation", "rows")
    limit = plan.get("limit")

    for f in filters:
        if f.get("column") not in df.columns:
            return None, f"❌ Invalid column in plan: {f.get('column')}"

    if len(df) == 0:
        return df.copy(), "⚠ DataFrame empty."

    mask = pd.Series(True, index=df.index)

    for f in filters:
        col, op, val = f.get("column"), f.get("op", "=="), f.get("value")

        if col == "Month" and isinstance(val, str):
            val = normalize_month_string(val) or val
        if col == "Status" and isinstance(val, str):
            val = normalize_status(val)

        try:
            mask &= build_condition(df, col, op, val)
        except Exception as e:
            return None, f"❌ Invalid filter: {e}"

    filtered = df[mask].copy()
    if filtered.empty:
        return filtered, "⚠ No matching rows."

    if metric:
        s = filtered[metric]
        if agg == "sum": return s.sum(), None
        if agg == "max": return s.max(), None
        if agg == "min": return s.min(), None
        if agg == "list": return s.tolist(), None

    if isinstance(limit, int) and limit > 0:
        return filtered.head(min(limit, 500)), None

    return filtered, None


# ======================================================================
# AI PLANNER
# ======================================================================
def build_planner_prompt(question, df_name, df, detected_status):
    cols = list(df.columns)

    if df_name == "stock":
        id_col = "Product Code"
        numeric_col = "PACKS"
        month_col = "Month"
        status_col = "Status"
        status_examples = CANONICAL_STATUSES
    else:
        id_col = "ITEM #"
        numeric_col = "AVAILABLE_num"
        month_col = None
        status_col = None
        status_examples = []

    examples = []

    if df_name == "summary":
        examples.append({
            "user_question": "How many AVAILABLE for ITEM # 20373?",
            "plan": {
                "filters": [{"column": "ITEM #", "op": "==", "value": 20373}],
                "metric": "AVAILABLE_num",
                "aggregation": "sum",
                "limit": 0,
            },
        })

    if df_name == "stock":
        examples.append({
            "user_question": "How many PACKS for product code 20373 in November 2025 with status invoiced?",
            "plan": {
                "filters": [
                    {"column": "Product Code", "op": "==", "value": 20373},
                    {"column": "Month", "op": "==", "value": "November 2025"},
                    {"column": "Status", "op": "==", "value": "Invoiced"},
                ],
                "metric": "PACKS",
                "aggregation": "sum",
                "limit": 0,
            },
        })

    prompt = f"""
You are a strict JSON query planner.

DATAFRAME NAME: {df_name}_df
COLUMNS: {cols}

Return ONLY a JSON object:
{{
  "filters": [...],
  "metric": "<column or null>",
  "aggregation": "sum" | "max" | "min" | "rows" | "list",
  "limit": <int or 0>
}}
"""

    if df_name == "stock" and detected_status in CANONICAL_STATUSES:
        prompt += f"""
Detected status: {detected_status}.
You MUST include filter: {{"column":"Status","op":"==","value":"{detected_status}"}}.
"""

    for ex in examples:
        prompt += f"""
USER_QUESTION_EXAMPLE:
{ex['user_question']}

JSON_PLAN_EXAMPLE:
{json.dumps(ex['plan'])}
"""

    prompt += f"\nUser question:\n{question}\nReturn only valid JSON."
    return prompt


@st.cache_data(show_spinner=False)
def get_plan(question, df_name, detected_status):
    df = stock_df if df_name == "stock" else summary_df
    prompt = build_planner_prompt(question, df_name, df, detected_status)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        plan = json.loads(resp.choices[0].message.content)
    except:
        return None

    if not isinstance(plan, dict):
        return None

    plan.setdefault("filters", [])
    plan.setdefault("aggregation", "rows")
    if not isinstance(plan.get("limit"), int):
        plan["limit"] = 0
    return plan


# ======================================================================
# MAIN ANSWER
# ======================================================================
def answer_question(question: str, df_name: str):
    df = stock_df if df_name == "stock" else summary_df
    detected_status = None

    if df_name == "stock":
        detected_status = detect_status_from_question(question)

        if detected_status is None:
            return (
                "⚠ Please specify exactly ONE status: Invoiced, Shipped, Landed, Ordered.",
                None,
                None,
            )

        if detected_status == "MULTI":
            return (
                "⚠ Your question includes more than one status. Please choose only ONE.",
                None,
                None,
            )

    plan = get_plan(question, df_name, detected_status)
    if plan is None:
        return "❌ I could not build a valid JSON plan.", None, None

    result, err = apply_plan(plan, df, df_name)
    if err:
        return err, plan, result

    if isinstance(result, pd.DataFrame):
        preview = result.head(20).to_string()
    else:
        preview = str(result)

    explain_prompt = f"""
Explain this result to a non-technical user:

User question:
{question}

Plan:
{json.dumps(plan)}

Raw result:
{preview}
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": explain_prompt}],
        temperature=0.2,
    )

    explanation = resp.choices[0].message.content.strip()
    return explanation, plan, result


# ======================================================================
# UI — CHAT STYLE
# ======================================================================

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Title
st.markdown(
    """
    <h1 style='text-align:center; margin-bottom: 10px;'>📦 Kustom Timber Stock Inventory Chatbot</h1>
    """,
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Choose Sheet")
    sheet_choice = st.selectbox(
        "Sheet:",
        ["Stock Sheet (stock_df)", "Summary Sheet (summary_df)"]
    )

    df_name = "stock" if sheet_choice.startswith("Stock") else "summary"
    df_selected = stock_df if df_name == "stock" else summary_df

    if st.checkbox("Show DataFrame Preview"):
        st.dataframe(df_selected, use_container_width=True)

    st.subheader("Quick Questions")
    if st.button("Ordered 20373 - Nov 2025"):
        st.session_state["preset_question"] = (
            "how many Ordered packs for 20373 for November 2025?"
        )
    if st.button("Invoiced 20373 - Nov 2025"):
        st.session_state["preset_question"] = (
            "how many Invoiced packs for 20373 for November 2025?"
        )
    if st.button("Landed 20588 - Sep 2025"):
        st.session_state["preset_question"] = (
            "how many status (Landed) packs for 20588 for September 2025?"
        )
    if st.button("Available for 20246"):
        st.session_state["preset_question"] = "how many available for 20246?"

    show_debug = st.checkbox("🛠 Show debug plan & raw result")

    st.markdown("---")
    if st.button("🧹 Clear Chat History"):
        st.session_state["chat_history"] = []
        st.rerun()


# ---------------- CHAT WINDOW ----------------
chat_container = st.container()

scroll_css = """
<style>
.chat-box {
    max-height: 420px;
    overflow-y: auto;
    padding-right: 12px;
    border: 1px solid #555;
    border-radius: 10px;
    background: #2B2B2B;
}

.chat-bubble-user {
    background:#3A3A3A;
    padding:12px;
    border-radius:10px;
    margin-bottom:8px;
    width:fit-content;
    max-width:80%;
    color:#FFF;
}

.chat-bubble-bot {
    background:#1F1F1F;
    padding:12px;
    border-radius:10px;
    margin-bottom:8px;
    width:fit-content;
    max-width:80%;
    color:#FFF;
}
</style>
"""
st.markdown(scroll_css, unsafe_allow_html=True)

with chat_container:
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-bubble-user'><strong>You:</strong><br>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='chat-bubble-bot'><strong>Bot:</strong><br>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- INPUT ----------------
default_q = st.session_state.get("preset_question", "")
question = st.text_input(
    "Ask a question:",
    value=default_q,
    placeholder="e.g., how many landed packs for 20588 for September 2025?"
)

if st.button("Send"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        st.session_state["chat_history"].append({"role": "user", "content": question})

        explanation, plan, result = answer_question(question, df_name)

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": explanation}
        )

        st.session_state["preset_question"] = ""

        st.rerun()


# ---------------- DEBUG ----------------
if show_debug and "last_question" in st.session_state:
    st.markdown("---")
    st.markdown("### 🛠 Debug Info")

    explanation, plan, result = answer_question(
        st.session_state.get("last_question", ""), df_name
    )

    if isinstance(plan, dict):
        st.markdown("### 🧩 JSON Plan")
        st.json(plan)

    st.markdown("### 📄 Raw Result")
    if isinstance(result, (pd.DataFrame, pd.Series)):
        st.dataframe(result)
    else:
        st.write(result)
