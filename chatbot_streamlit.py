# ======================================================================
# STREAMLIT GOOGLE SHEETS CHATBOT (Stock + Summary)
# HYBRID JSON PLANNER VERSION (no exec, no eval)
# With:
#  - Automatic Status detection (Invoiced / Shipped / Landed / Ordered)
#  - Ask user to specify Status if missing (Option 3)
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import gspread
import os
import json
import re
from pathlib import Path
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
MODEL = "gpt-4.1-mini"


# ======================================================================
# GOOGLE CREDS
# ======================================================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

def get_google_creds():
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
# SHEET CONFIG
# ======================================================================
SPREADSHEET_ID = "1UG_N-zkgwCpObWTgmg8EPS7-N08aqintu8h3kN8yRmM"
WS_STOCK = "Stock"
WS_SUMMARY = "Summary"


# ======================================================================
# REFRESH BUTTON (st.rerun)
# ======================================================================
if "refresh_triggered" not in st.session_state:
    st.session_state["refresh_triggered"] = False

def trigger_refresh():
    st.cache_data.clear()
    st.session_state["refresh_triggered"] = True
    st.success("🔄 Sheets refreshed! Reloading...")

if st.button("🔄 Refresh Sheets Now"):
    trigger_refresh()

if st.session_state["refresh_triggered"]:
    st.session_state["refresh_triggered"] = False
    st.rerun()


# ======================================================================
# LOAD SHEETS
# ======================================================================
@st.cache_data(show_spinner=True)
def load_sheets():

    # ---------- STOCK ----------
    ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WS_STOCK)
    stock_df = pd.DataFrame(ws.get_all_values())
    stock_df.columns = stock_df.iloc[0].str.strip()
    stock_df = stock_df[1:].reset_index(drop=True)
    stock_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Numeric cleanup
    if "Product Code" in stock_df:
        stock_df["Product Code"] = pd.to_numeric(stock_df["Product Code"], errors="coerce")

    if "PACKS" in stock_df:
        stock_df["PACKS"] = pd.to_numeric(stock_df["PACKS"], errors="coerce")
        # Empty numeric cells → 0
        stock_df["PACKS"] = stock_df["PACKS"].fillna(0)

    # Normalize Month column if present
    if "Month" in stock_df:
        stock_df["Month"] = stock_df["Month"].astype(str).str.strip()

    # Normalize Status if present
    if "Status" in stock_df:
        def norm_status(s):
            if not isinstance(s, str):
                return ""
            t = s.strip().lower()
            # Only allowed: Invoiced, Shipped, Landed, Ordered
            if t.startswith("inv"):
                return "Invoiced"
            if "ship" in t:
                return "Shipped"
            if "land" in t:
                return "Landed"
            if "order" in t or "ord" in t:
                return "Ordered"
            # fallback: title-case
            return s.strip().title()
        stock_df["Status"] = stock_df["Status"].apply(norm_status)

    # ---------- SUMMARY ----------
    ws2 = gc.open_by_key(SPREADSHEET_ID).worksheet(WS_SUMMARY)
    summary_df = pd.DataFrame(ws2.get_all_values())
    summary_df.columns = summary_df.iloc[0].str.strip()
    summary_df = summary_df[1:].reset_index(drop=True)
    summary_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    numeric_cols = [
        "COST","PACK SIZE","ORDERED","LANDED","Shipped","SOH (DC)",
        "Packs (DC)","Invoiced","AVAILABLE","SOH + SOO","SOO COST","SOH COST"
    ]
    for col in numeric_cols:
        if col in summary_df:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
            summary_df[col] = summary_df[col].fillna(0)  # empty → 0

    if "ITEM #" in summary_df:
        summary_df["ITEM #"] = pd.to_numeric(summary_df["ITEM #"], errors="coerce")

    # ALWAYS create AVAILABLE_num as measure
    summary_df["AVAILABLE_num"] = pd.to_numeric(
        summary_df.get("AVAILABLE", np.nan),
        errors="coerce"
    ).fillna(0)

    return stock_df, summary_df

stock_df, summary_df = load_sheets()


# ======================================================================
# MONTH NORMALIZATION
# ======================================================================
def normalize_month(user_text: str):
    """
    Convert 'sep 2025', '09/2025', 'September 2025' etc into 'September 2025'
    to match the Month column in stock_df.
    """
    if not isinstance(user_text, str):
        return user_text

    text = user_text.strip().lower()
    if not text:
        return None

    text = re.sub(r"[-_/.,]+", " ", text)

    month_map = {
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

    parts = text.split()
    # pattern: "sep 2025"
    if len(parts) == 2:
        m_raw, y_raw = parts
        if m_raw in month_map and re.match(r"^\d{4}$", y_raw):
            return f"{month_map[m_raw]} {y_raw}"

    # pattern: "09 2025"
    m = re.match(r"^(\d{1,2})\s+(\d{4})$", text)
    if m:
        mm = int(m.group(1))
        yyyy = m.group(2)
        if 1 <= mm <= 12:
            month_names = [
                "January","February","March","April","May","June",
                "July","August","September","October","November","December"
            ]
            return f"{month_names[mm-1]} {yyyy}"

    # already canonical (e.g. "september 2025")
    m2 = re.match(
        r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})$",
        text,
    )
    if m2:
        return f"{m2.group(1).capitalize()} {m2.group(2)}"

    return None


# ======================================================================
# STATUS KEYWORD DETECTION (FOR STOCK)
# ======================================================================
def detect_status_from_question(question: str) -> str | None:
    """
    Detect a single status from the question text.
    Returns one of: "Invoiced", "Shipped", "Landed", "Ordered", "MULTI", or None.
    """
    if not isinstance(question, str):
        return None

    text = question.lower()

    statuses = set()

    # Invoiced / invoice / inv
    if re.search(r"\b(inv|invoice|invoiced)\b", text):
        statuses.add("Invoiced")

    # Shipped / ship
    if re.search(r"\b(ship|shipped)\b", text):
        statuses.add("Shipped")

    # Landed / land
    if re.search(r"\b(land|landed)\b", text):
        statuses.add("Landed")

    # Ordered / order / ord
    if re.search(r"\b(order|ordered|ord)\b", text):
        statuses.add("Ordered")

    if len(statuses) == 0:
        return None
    if len(statuses) > 1:
        return "MULTI"
    return list(statuses)[0]


# ======================================================================
# EXECUTE JSON PLAN
# ======================================================================
def apply_plan(plan: dict, df: pd.DataFrame, df_name: str):
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

    # Validate columns
    for f in filters:
        col = f.get("column")
        if col not in df_cols:
            return None, f"❌ Plan uses invalid column: {col}"

    if metric is not None and metric not in df_cols:
        return None, f"❌ Plan uses invalid metric column: {metric}"

    mask = pd.Series(True, index=df.index)

    for f in filters:
        col = f.get("column")
        op = f.get("op", "==")
        val = f.get("value")

        # Month normalization (stock only)
        if col == "Month" and isinstance(val, str):
            norm = normalize_month(val)
            if norm:
                val = norm

        # Status normalization at query-time (stock only)
        if df_name == "stock" and col == "Status" and isinstance(val, str):
            val_norm = val.strip().lower()
            if val_norm.startswith("inv"):        # "invoiced", "invoice", "inv"
                val = "Invoiced"
            elif "ship" in val_norm:              # "shipped", "ship"
                val = "Shipped"
            elif "land" in val_norm:              # "landed"
                val = "Landed"
            elif "order" in val_norm or "ord" in val_norm:  # "ordered", "order"
                val = "Ordered"
            else:
                val = val.strip().title()

        series = df[col]

        if op == "==":
            mask &= (series == val)
        elif op == "!=":
            mask &= (series != val)
        elif op == "<":
            mask &= (series < val)
        elif op == "<=":
            mask &= (series <= val)
        elif op == ">":
            mask &= (series > val)
        elif op == ">=":
            mask &= (series >= val)
        elif op == "contains":
            mask &= series.astype(str).str.contains(str(val), case=False, na=False)
        else:
            return None, f"❌ Unsupported operator: {op}"

    filtered = df[mask].copy()

    if filtered.empty:
        return filtered, "⚠ No matching rows found for the given filters."

    # Aggregation
    if agg == "sum" and metric:
        return filtered[metric].sum(), None
    if agg == "max" and metric:
        return filtered[metric].max(), None
    if agg == "min" and metric:
        return filtered[metric].min(), None
    if agg == "list" and metric:
        return filtered[metric].tolist(), None

    # rows
    if isinstance(limit, int) and limit > 0:
        return filtered.head(limit), None
    return filtered, None


# ======================================================================
# AI PLANNER
# ======================================================================
def build_planner_prompt(question: str, df_name: str, df: pd.DataFrame, detected_status: str | None) -> str:
    cols = list(df.columns)

    if df_name == "stock":
        id_col = "Product Code"
        numeric_col = "PACKS"
        month_col = "Month" if "Month" in df.columns else None
        status_col = "Status" if "Status" in df.columns else None
        status_examples = ["Invoiced", "Shipped", "Landed", "Ordered"]
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
                "filters": [
                    {"column": "ITEM #", "op": "==", "value": 20373}
                ],
                "metric": "AVAILABLE_num",
                "aggregation": "sum",
                "limit": 0
            }
        })

    if df_name == "stock" and month_col:
        examples.append({
            "user_question": "How many PACKS for product code 20373 in November 2025 with status invoiced?",
            "plan": {
                "filters": [
                    {"column": "Product Code", "op": "==", "value": 20373},
                    {"column": "Month", "op": "==", "value": "November 2025"},
                    {"column": "Status", "op": "==", "value": "Invoiced"}
                ],
                "metric": "PACKS",
                "aggregation": "sum",
                "limit": 0
            }
        })

    prompt = f"""
You are a query planner for ONE Pandas DataFrame.

DATAFRAME NAME: {df_name}_df
COLUMNS: {cols}

You MUST output ONLY a single JSON object.

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
- Use ONLY columns from this dataframe.
- Product / item code MUST filter on "{id_col}".
- For 'how many', 'total', 'how much', you MUST use '{numeric_col}' as the metric.
- On the STOCK sheet, do NOT use any other numeric column by default except 'PACKS'
  unless the user explicitly mentions another numeric column.
- On the SUMMARY sheet, do NOT use any other numeric column by default except 'AVAILABLE_num'
  unless the user explicitly mentions another numeric column (like ORDERED, LANDED, etc.).
- If a month is mentioned and 'Month' exists, add a filter on 'Month' with op '=='.
- If status is mentioned and '{status_col}' exists, add a filter on '{status_col}'.
"""

    if status_col:
        prompt += f"\nKnown example statuses (canonical): {status_examples}\n"

    # Inject detected status (from Python side) to enforce it
    if df_name == "stock" and status_col and detected_status in ["Invoiced", "Shipped", "Landed", "Ordered"]:
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

Return ONLY valid JSON. No explanation, no markdown.
"""
    return prompt


@st.cache_data(show_spinner=False)
def get_plan(question: str, df_name: str, detected_status: str | None) -> dict | None:
    """
    Generate a JSON plan using GPT, cached by (question, df_name, detected_status).
    """
    df = stock_df if df_name == "stock" else summary_df
    prompt = build_planner_prompt(question, df_name, df, detected_status)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        plan = json.loads(raw)
    except Exception:
        return None

    if not isinstance(plan, dict):
        return None
    if "filters" not in plan:
        plan["filters"] = []
    return plan


# ======================================================================
# MAIN ANSWER FUNCTION
# ======================================================================
def answer_question(question: str, df_name: str):
    df = stock_df if df_name == "stock" else summary_df

    detected_status = None

    # For stock, enforce that status is explicitly present (Option 3)
    if df_name == "stock":
        detected_status = detect_status_from_question(question)

        if detected_status is None:
            # No status word found → ask user to clarify
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

    if isinstance(result, pd.DataFrame):
        result_preview = result.to_string()
    elif isinstance(result, pd.Series):
        result_preview = result.to_string()
    else:
        result_preview = str(result)

    explain_prompt = f"""
You are an assistant explaining data results.

USER QUESTION:
{question}

DATAFRAME: {df_name}_df

PLAN (JSON):
{json.dumps(plan)}

RAW RESULT:
{result_preview}

Explain the answer clearly for a non-technical user.
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
st.subheader("Choose Sheet to Query:\n Select stock_df to ask questions about IMR and summary_df for stock availability")
sheet_choice = st.selectbox("Sheet:", ["Stock Sheet (stock_df)", "Summary Sheet (summary_df)"])

df_name = "stock" if sheet_choice.startswith("Stock") else "summary"
df_selected = stock_df if df_name == "stock" else summary_df

if st.checkbox("Show DataFrame Preview"):
    st.dataframe(df_selected, use_container_width=True)

st.markdown("### Quick questions")

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("how many Ordered packs for 20373 for November 2025?"):
        st.session_state["preset_question"] = "how many Ordered packs for 20373 for November 2025?"
with c2:
    if st.button("how many Invoiced packs for 20373 for November 2025?"):
        st.session_state["preset_question"] = "how many Invoiced packs for 20373 for November 2025?"
with c3:
    if st.button("how many status (Landed) packs for 20588 for September 2025?"):
        st.session_state["preset_question"] = "how many status (Landed) packs for 20588 for September 2025?"
with c4:
    if st.button("how many available for 20246?"):
        st.session_state["preset_question"] = "how many available for 20246?"

default_q = st.session_state.get("preset_question", "")
question = st.text_input("Ask your question:", value=default_q)

show_debug = st.checkbox("🛠 Show debug plan & raw result")

if st.button("Ask"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        explanation, plan, result = answer_question(question, df_name)

        st.markdown("### Chatbot Answer")
        st.write(explanation)

        if show_debug:
            if isinstance(plan, dict):
                st.markdown("### 🧩 JSON Plan")
                st.json(plan)
            st.markdown("### 📄 Raw Result")
            if isinstance(result, (pd.DataFrame, pd.Series)):
                st.dataframe(result)
            else:
                st.write(result)
