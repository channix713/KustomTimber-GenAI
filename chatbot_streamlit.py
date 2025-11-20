# ======================================================================
# STREAMLIT GOOGLE SHEETS CHATBOT (Stock + Summary)
# SAFER JSON PLANNER VERSION (no exec, no eval)
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
st.title("📦 Inventory Chatbot — Stock & Summary Sheets (JSON Planner)")


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
# REFRESH BUTTON (now uses st.rerun())
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
    st.rerun()   # ✅ FIXED (experimental_rerun removed)


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

    if "Month" in stock_df:
        stock_df["Month"] = stock_df["Month"].astype(str).str.strip()

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

    if "ITEM #" in summary_df:
        summary_df["ITEM #"] = pd.to_numeric(summary_df["ITEM #"], errors="coerce")

    summary_df["AVAILABLE_num"] = pd.to_numeric(summary_df.get("AVAILABLE", np.nan), errors="coerce")

    return stock_df, summary_df

stock_df, summary_df = load_sheets()


# ======================================================================
# MONTH NORMALIZATION
# ======================================================================
def normalize_month(user_text: str):
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
    if len(parts) == 2:
        m_raw, y_raw = parts
        if m_raw in month_map and re.match(r"^\d{4}$", y_raw):
            return f"{month_map[m_raw]} {y_raw}"

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

    m2 = re.match(r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})$", text)
    if m2:
        return f"{m2.group(1).capitalize()} {m2.group(2)}"

    return None


# ======================================================================
# EXECUTE JSON PLAN
# ======================================================================
def apply_plan(plan: dict, df: pd.DataFrame, df_name: str):

    if not isinstance(plan, dict):
        return None, "❌ Invalid plan format"

    filters = plan.get("filters", [])
    metric = plan.get("metric")
    agg = plan.get("aggregation", "rows")
    limit = plan.get("limit")

    df_cols = list(df.columns)

    # Validate columns
    for f in filters:
        if f["column"] not in df_cols:
            return None, f"❌ Invalid column: {f['column']}"

    if metric and metric not in df_cols:
        return None, f"❌ Invalid metric: {metric}"

    mask = pd.Series(True, index=df.index)

    for f in filters:
        col = f["column"]
        op = f.get("op", "==")
        val = f.get("value")

        if col == "Month" and isinstance(val, str):
            norm = normalize_month(val)
            if norm:
                val = norm

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

    filtered = df[mask]

    if filtered.empty:
        return filtered, "⚠ No matching rows found."

    # Apply aggregation
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
def build_planner_prompt(question, df_name, df):

    cols = list(df.columns)

    if df_name == "stock":
        id_col = "Product Code"
        numeric_col = "PACKS"
        month_col = "Month" if "Month" in df.columns else None
    else:
        id_col = "ITEM #"
        numeric_col = "AVAILABLE_num"
        month_col = None

    examples = []

    if df_name == "summary":
        examples.append({
            "user_question": "How many AVAILABLE for ITEM # 20373?",
            "plan": {
                "filters": [{"column": "ITEM #", "op": "==", "value": 20373}],
                "metric": "AVAILABLE_num",
                "aggregation": "sum"
            }
        })

    if df_name == "stock" and month_col:
        examples.append({
            "user_question": "How many PACKS for product code 20373 in November 2025?",
            "plan": {
                "filters": [
                    {"column": "Product Code", "op": "==", "value": 20373},
                    {"column": "Month", "op": "==", "value": "November 2025"},
                ],
                "metric": "PACKS",
                "aggregation": "sum"
            }
        })

    prompt = f"""
You are a query planner for ONE Pandas DataFrame.

DATAFRAME: {df_name}_df
COLUMNS: {cols}

Return ONLY a JSON object describing how to answer the question.

JSON SCHEMA:
{{
  "filters": [
    {{ "column": "ITEM #", "op": "==", "value": 20373 }}
  ],
  "metric": "AVAILABLE_num",
  "aggregation": "sum" | "max" | "min" | "rows" | "list",
  "limit": 50
}}

Rules:
- Use ONLY columns from this dataframe.
- Product codes must filter on "{id_col}".
- Quantity questions MUST use "{numeric_col}".
- For stock_df use PACKS; for summary_df use AVAILABLE_num.
- Convert month references to column "Month" if present.
- If user asks "how many", use aggregation "sum".

EXAMPLES:
"""

    for ex in examples:
        prompt += "\nUSER:\n" + ex["user_question"]
        prompt += "\nPLAN:\n" + json.dumps(ex["plan"]) + "\n"

    prompt += f"\nNOW ANSWER THIS QUESTION:\n{question}\n\nReturn ONLY JSON."

    return prompt


@st.cache_data(show_spinner=False)
def get_plan(question, df_name, df_cols):
    dummy = pd.DataFrame(columns=df_cols)
    prompt = build_planner_prompt(question, df_name, dummy)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "")

    try:
        return json.loads(raw)
    except Exception:
        return None


# ======================================================================
# MAIN ANSWER FUNCTION
# ======================================================================
def answer_question(question, df_name):
    df = stock_df if df_name == "stock" else summary_df

    plan = get_plan(question, df_name, df.columns)
    if plan is None:
        return "❌ Could not generate a valid plan.", None, None

    result, err = apply_plan(plan, df, df_name)

    if err:
        return err, plan, result

    # Human-friendly explanation
    preview = result.to_string() if isinstance(result, (pd.DataFrame, pd.Series)) else str(result)

    explain_prompt = f"""
Explain for a non-technical user:

QUESTION:
{question}

PLAN:
{json.dumps(plan)}

RESULT:
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
# UI
# ======================================================================
st.subheader("Choose Sheet to Query")
sheet_choice = st.selectbox("Sheet:", ["Stock Sheet (stock_df)", "Summary Sheet (summary_df)"])

df_name = "stock" if sheet_choice.startswith("Stock") else "summary"
df_selected = stock_df if df_name == "stock" else summary_df

if st.checkbox("Show DataFrame Preview"):
    st.dataframe(df_selected, use_container_width=True)

st.markdown("### Quick questions")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("AVAILABLE for ITEM # 20373 (Summary)"):
        st.session_state["preset_question"] = "How many AVAILABLE for ITEM # 20373?"
with c2:
    if st.button("PACKS for product 20373 in November 2025 (Stock)"):
        st.session_state["preset_question"] = "How many PACKS for product code 20373 for month November 2025?"
with c3:
    if st.button("Show all rows for product 20373 (Stock)"):
        st.session_state["preset_question"] = "Show all rows for product code 20373."

question = st.text_input("Ask your question:", value=st.session_state.get("preset_question", ""))

show_debug = st.checkbox("🛠 Show debug plan & raw result")

if st.button("Ask"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        explanation, plan, result = answer_question(question, df_name)

        st.markdown("### Chatbot Answer")
        st.write(explanation)

        if show_debug:
            st.subheader("🧩 JSON Plan")
            st.json(plan)
            st.subheader("📄 Raw Result")
            if isinstance(result, (pd.DataFrame, pd.Series)):
                st.dataframe(result)
            else:
                st.write(result)
