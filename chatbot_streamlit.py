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

# Canonical status names
CANONICAL_STATUSES = ["Invoiced", "Shipped", "Landed", "Ordered"]

# Keywords for status detection in user questions
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
    st.error("❌ Missing OPENAI_API_KEY in secrets or environment")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================================================
# GOOGLE CREDS
# ======================================================================
def get_google_creds() -> Credentials:
    """Load Google service account credentials from Streamlit secrets."""
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
    """
    Convert 'sep 2025', '09/2025', 'September 2025' etc into 'September 2025'
    Returns None if the string cannot be parsed.
    """
    if not isinstance(user_text, str):
        return None

    text = user_text.strip().lower()
    if not text:
        return None

    # Replace delimiters with space
    text = re.sub(r"[-_/.,]+", " ", text)
    parts = text.split()

    # Pattern: "sep 2025"
    if len(parts) == 2:
        m_raw, y_raw = parts
        if m_raw in MONTH_MAP and re.match(r"^\d{4}$", y_raw):
            return f"{MONTH_MAP[m_raw]} {y_raw}"

    # Pattern: "09 2025"
    m = re.match(r"^(\d{1,2})\s+(\d{4})$", text)
    if m:
        mm = int(m.group(1))
        yyyy = m.group(2)
        if 1 <= mm <= 12:
            return f"{MONTH_NAMES[mm - 1]} {yyyy}"

    # Already canonical (e.g. "september 2025")
    m2 = re.match(
        r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})$",
        text,
    )
    if m2:
        return f"{m2.group(1).capitalize()} {m2.group(2)}"

    return None


def normalize_status(raw: Any) -> str:
    """
    Normalize status values to one of: Invoiced, Shipped, Landed, Ordered.
    Returns empty string if not recognized.
    """
    if not isinstance(raw, str):
        return ""

    t = raw.strip().lower()
    if not t:
        return ""

    # Try canonical mapping from keywords
    for canonical, keywords in STATUS_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return canonical

    # Fallback: title-case
    return raw.strip().title()


def detect_status_from_question(question: str) -> Optional[str]:
    """
    Detect a single status from the question text.
    Returns one of: "Invoiced", "Shipped", "Landed", "Ordered", "MULTI", or None.
    """
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
# REFRESH BUTTON
# ======================================================================
def clear_sheet_cache_and_rerun():
    """Clear cached sheet data and rerun the app."""
    load_sheets.clear()  # type: ignore[attr-defined]
    st.success("🔄 Sheets cache cleared! Reloading...")
    st.experimental_rerun()


if st.button("🔄 Refresh Sheets Now"):
    clear_sheet_cache_and_rerun()

# ======================================================================
# LOAD SHEETS (CACHED)
# ======================================================================
@st.cache_data(show_spinner=True)
def load_sheets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean Stock and Summary sheets, cached."""
    # Open spreadsheet once
    sh = gc.open_by_key(SPREADSHEET_ID)

    # ---------- STOCK ----------
    ws_stock = sh.worksheet(WS_STOCK)
    stock_df = pd.DataFrame(ws_stock.get_all_values())
    stock_df.columns = stock_df.iloc[0].str.strip()
    stock_df = stock_df[1:].reset_index(drop=True)
    stock_df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    # Numeric cleanup
    if "Product Code" in stock_df.columns:
        stock_df["Product Code"] = pd.to_numeric(
            stock_df["Product Code"], errors="coerce"
        )

    if "PACKS" in stock_df.columns:
        stock_df["PACKS"] = pd.to_numeric(stock_df["PACKS"], errors="coerce")
        stock_df["PACKS"] = stock_df["PACKS"].fillna(0)

    if "Month" in stock_df.columns:
        stock_df["Month"] = stock_df["Month"].astype(str).str.strip()

    # Normalize Status if present
    if "Status" in stock_df.columns:
        stock_df["Status"] = stock_df["Status"].apply(normalize_status)

    # ---------- SUMMARY ----------
    ws_summary = sh.worksheet(WS_SUMMARY)
    summary_df = pd.DataFrame(ws_summary.get_all_values())
    summary_df.columns = summary_df.iloc[0].str.strip()
    summary_df = summary_df[1:].reset_index(drop=True)
    summary_df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    for col in SUMMARY_NUMERIC_COLS:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
            summary_df[col] = summary_df[col].fillna(0)

    if "ITEM #" in summary_df.columns:
        summary_df["ITEM #"] = pd.to_numeric(
            summary_df["ITEM #"], errors="coerce"
        )

    # ALWAYS create AVAILABLE_num as measure
    summary_df["AVAILABLE_num"] = pd.to_numeric(
        summary_df.get("AVAILABLE", np.nan),
        errors="coerce",
    ).fillna(0)

    return stock_df, summary_df


stock_df, summary_df = load_sheets()

# ======================================================================
# EXECUTE JSON PLAN
# ======================================================================
def build_condition(
    df: pd.DataFrame, column: str, op: str, value: Any
) -> pd.Series:
    """Build a boolean mask for a single filter condition."""
    series = df[column]

    if op == "==":
        return series == value
    elif op == "!=":
        return series != value
    elif op == "<":
        return series < value
    elif op == "<=":
        return series <= value
    elif op == ">":
        return series > value
    elif op == ">=":
        return series >= value
    elif op == "contains":
        return series.astype(str).str.contains(str(value), case=False, na=False)

    raise ValueError(f"Unsupported operator: {op}")


def apply_plan(
    plan: Dict[str, Any], df: pd.DataFrame, df_name: str
) -> Tuple[Any, Optional[str]]:
    """
    Execute a JSON plan on a dataframe.

    Plan schema:
    {
      "filters": [
        {"column": "ITEM #", "op": "==", "value": 20373},
        {"column": "Month", "op": "==", "value": "September 2025"},
        {"column": "Status", "op": "==", "value": "Landed"}
      ],
      "metric": "AVAILABLE_num",
      "aggregation": "sum" | "max" | "min" | "rows" | "list",
      "limit": 50
    }
    """
    if not isinstance(plan, dict):
        return None, "❌ Invalid plan format (not a dict)."

    filters = plan.get("filters", [])
    metric = plan.get("metric")
    agg = plan.get("aggregation", "rows")
    limit = plan.get("limit")

    df_cols = list(df.columns)

    # Basic validations
    if not isinstance(filters, list):
        return None, "❌ Plan 'filters' must be a list."

    if metric is not None and metric not in df_cols:
        return None, f"❌ Plan uses invalid metric column: {metric}"

    # Validate columns in filters
    for f in filters:
        col = f.get("column")
        if col not in df_cols:
            return None, f"❌ Plan uses invalid column: {col}"

    # Build combined mask
    if len(df) == 0:
        return df.copy(), "⚠ Dataframe is empty."

    mask = pd.Series(True, index=df.index)

    for f in filters:
        col = f.get("column")
        op = f.get("op", "==")
        val = f.get("value")

        # Month normalization
        if col == "Month" and isinstance(val, str):
            norm_m = normalize_month_string(val)
            if norm_m:
                val = norm_m

        # Status normalization
        if col == "Status" and isinstance(val, str):
            val = normalize_status(val)

        try:
            condition = build_condition(df, col, op, val)
        except Exception as e:
            return None, f"❌ Invalid filter {f}: {e}"

        mask &= condition

    filtered = df[mask].copy()

    if filtered.empty:
        return filtered, "⚠ No matching rows found for the given filters."

    # Aggregations
    if isinstance(metric, str):
        series = filtered[metric]
    else:
        series = None

    if agg == "sum" and series is not None:
        return series.sum(), None
    if agg == "max" and series is not None:
        return series.max(), None
    if agg == "min" and series is not None:
        return series.min(), None
    if agg == "list" and series is not None:
        return series.tolist(), None

    # rows mode
    if isinstance(limit, int) and limit > 0:
        # Clip limit to avoid huge results
        safe_limit = min(limit, 500)
        return filtered.head(safe_limit), None

    return filtered, None

# ======================================================================
# AI PLANNER
# ======================================================================
def build_planner_prompt(
    question: str,
    df_name: str,
    df: pd.DataFrame,
    detected_status: Optional[str],
) -> str:
    cols = list(df.columns)

    if df_name == "stock":
        id_col = "Product Code"
        numeric_col = "PACKS"
        month_col = "Month" if "Month" in df.columns else None
        status_col = "Status" if "Status" in df.columns else None
        status_examples = CANONICAL_STATUSES
    else:
        id_col = "ITEM #"
        numeric_col = "AVAILABLE_num"
        month_col = None
        status_col = None
        status_examples = []

    examples: List[Dict[str, Any]] = []

    if df_name == "summary":
        examples.append(
            {
                "user_question": "How many AVAILABLE for ITEM # 20373?",
                "plan": {
                    "filters": [
                        {"column": "ITEM #", "op": "==", "value": 20373},
                    ],
                    "metric": "AVAILABLE_num",
                    "aggregation": "sum",
                    "limit": 0,
                },
            }
        )

    if df_name == "stock" and month_col:
        examples.append(
            {
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
            }
        )

    prompt = f"""
You are a strict JSON query planner for ONE Pandas DataFrame.

DATAFRAME NAME: {df_name}_df
COLUMNS: {cols}

You MUST output ONLY a single JSON object and NOTHING else.

JSON SCHEMA:
{{
  "filters": [
    {{"column": "<column name>", "op": "==", "value": <value>}},
    ...
  ],
  "metric": "<numeric column or null>",
  "aggregation": "sum" | "max" | "min" | "rows" | "list",
  "limit": <integer or 0>
}}

STRICT RULES:
- Use ONLY column names that appear in COLUMNS.
- Never invent new columns.
- Product / item code MUST filter on "{id_col}" if the question mentions a product or item code.
- For 'how many', 'total', 'how much', you MUST use '{numeric_col}' as the metric,
  unless the user explicitly names a different numeric column.
- On the STOCK sheet, do NOT use any other numeric column by default except 'PACKS'
  unless the user explicitly mentions another numeric column.
- On the SUMMARY sheet, do NOT use any other numeric column by default except 'AVAILABLE_num'
  unless the user explicitly mentions another numeric column (like ORDERED, LANDED, etc.).
- If a month is mentioned and 'Month' exists, add a filter on 'Month' with op '=='.
- If a status is mentioned and '{status_col}' exists, add a filter on '{status_col}'.

If the user mentions any unknown field that is not a column name, ignore it.
If you are unsure, use an empty filters list: "filters": [].
"""

    if status_col:
        prompt += f"\nKnown example statuses (canonical): {status_examples}\n"

    # Inject detected status (from Python side) to enforce it
    if df_name == "stock" and status_col and detected_status in CANONICAL_STATUSES:
        prompt += f"""
The user's question contains the status keyword, interpreted as:
DETECTED_STATUS = "{detected_status}"

You MUST include a filter:
  {{"column": "Status", "op": "==", "value": "{detected_status}"}}
in the filters list of the JSON plan.
"""

    prompt += "\nEXAMPLES:\n"
    for ex in examples:
        prompt += "\nUSER_QUESTION_EXAMPLE:\n"
        prompt += ex["user_question"] + "\n"
        prompt += "JSON_PLAN_EXAMPLE:\n"
        prompt += json.dumps(ex["plan"]) + "\n"

    prompt += f"""
NOW THE REAL USER QUESTION:
{question}

Return ONLY valid JSON according to the schema. No explanation, no markdown.
"""
    return prompt


@st.cache_data(show_spinner=False)
def get_plan(
    question: str, df_name: str, detected_status: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Generate a JSON plan using GPT, cached by (question, df_name, detected_status).
    """
    df = stock_df if df_name == "stock" else summary_df
    prompt = build_planner_prompt(question, df_name, df, detected_status)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content.strip()

    try:
        plan: Dict[str, Any] = json.loads(raw)
    except Exception:
        return None

    if not isinstance(plan, dict):
        return None

    # Ensure required keys exist and are well-typed
    if "filters" not in plan or not isinstance(plan["filters"], list):
        plan["filters"] = []

    if "aggregation" not in plan or not isinstance(plan["aggregation"], str):
        plan["aggregation"] = "rows"

    # Normalize limit
    if "limit" in plan and not isinstance(plan["limit"], int):
        plan["limit"] = 0

    return plan

# ======================================================================
# MAIN ANSWER FUNCTION
# ======================================================================
def answer_question(
    question: str, df_name: str
) -> Tuple[str, Optional[Dict[str, Any]], Any]:
    df = stock_df if df_name == "stock" else summary_df
    detected_status: Optional[str] = None

    # For stock, enforce that status is explicitly present (Option 3)
    if df_name == "stock":
        detected_status = detect_status_from_question(question)

        if detected_status is None:
            msg = (
                "⚠ I couldn't detect a status in your question.\n\n"
                "Please rephrase your question including exactly ONE of these statuses:\n"
                "- Invoiced\n- Shipped\n- Landed\n- Ordered\n\n"
                "Example: `How many Landed packs for 20588 for September 2025?`"
            )
            return msg, None, None

        if detected_status == "MULTI":
            msg = (
                "⚠ Your question seems to mention more than one status.\n\n"
                "Please clarify using only ONE of:\n"
                "- Invoiced\n- Shipped\n- Landed\n- Ordered"
            )
            return msg, None, None

    # Build plan with detected_status (or None for summary_df)
    plan = get_plan(question, df_name, detected_status)
    if plan is None:
        return "❌ I couldn't create a valid plan for that question.", None, None

    result, err = apply_plan(plan, df, df_name)
    if err:
        return err, plan, result

    # Prepare a small text preview for the explainer prompt
    if isinstance(result, pd.DataFrame):
        result_preview = result.head(20).to_string()
    elif isinstance(result, pd.Series):
        result_preview = result.to_string()
    else:
        result_preview = str(result)

    explain_prompt = f"""
You are an assistant explaining data results in plain language.

USER QUESTION:
{question}

DATAFRAME: {df_name}_df

PLAN (JSON):
{json.dumps(plan)}

RAW RESULT (truncated preview):
{result_preview}

Explain the answer clearly and concisely for a non-technical user.
"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": explain_prompt}],
        temperature=0.2,
    )

    explanation = resp.choices[0].message.content.strip()
    return explanation, plan, result

# ======================================================================
# UI
# ======================================================================
# ======================================================================
# UI
# ======================================================================

# Centered main title
st.markdown(
    """
    <h1 style='text-align:center; margin-bottom: 10px;'>📦 Kustom Timber Stock Inventory Chatbot</h1>
    """,
    unsafe_allow_html=True,
)

# ------------------ SIDEBAR SETTINGS ------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Choose Sheet to Query")
    sheet_choice = st.selectbox(
        "Sheet:", ["Stock Sheet (stock_df)", "Summary Sheet (summary_df)"]
    )

    df_name = "stock" if sheet_choice.startswith("Stock") else "summary"
    df_selected = stock_df if df_name == "stock" else summary_df

    if st.checkbox("Show DataFrame Preview"):
        st.dataframe(df_selected, use_container_width=True)

    st.subheader("Quick Questions")
    if st.button("Ordered 20373 - Nov 2025"):
        st.session_state["preset_question"] = "how many Ordered packs for 20373 for November 2025?"
    if st.button("Invoiced 20373 - Nov 2025"):
        st.session_state["preset_question"] = "how many Invoiced packs for 20373 for November 2025?"
    if st.button("Landed 20588 - Sep 2025"):
        st.session_state["preset_question"] = "how many status (Landed) packs for 20588 for September 2025?"
    if st.button("Available for 20246"):
        st.session_state["preset_question"] = "how many available for 20246?"

    # Debug toggle
    show_debug = st.checkbox("🛠 Show debug plan & raw result")


# ------------------ MAIN CONTENT ------------------

default_q = st.session_state.get("preset_question", "")
question = st.text_input(
    "Ask your question:",
    value=default_q,
    placeholder="e.g., how many landed packs for 20588 for September 2025?"
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        st.session_state["last_question"] = question
        explanation, plan, result = answer_question(question, df_name)

        st.markdown("### 💬 Chatbot Answer")
        st.write(explanation)

        # Debug outputs
        if show_debug:
            if isinstance(plan, dict):
                st.markdown("### 🧩 JSON Plan")
                st.json(plan)

            st.markdown("### 📄 Raw Result")
            if isinstance(result, (pd.DataFrame, pd.Series)):
                st.dataframe(result)
            else:
                st.write(result)

