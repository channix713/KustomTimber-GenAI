# ======================================================================
# UI (Refactored Modern Chat Interface)
# ======================================================================

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []


# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.header("⚙️ Settings")

    sheet_choice = st.radio(
        "Select Sheet",
        ["Stock Sheet (stock_df)", "Summary Sheet (summary_df)"],
        help="Choose which Google Sheet to query."
    )

    df_name = "stock" if sheet_choice.startswith("Stock") else "summary"
    df_selected = stock_df if df_name == "stock" else summary_df

    st.markdown("---")

    if st.button("🔄 Refresh Google Sheets"):
        clear_sheet_cache_and_rerun()

    st.markdown("---")

    debug_mode = st.checkbox("🛠 Show Debug Info")
    show_preview = st.checkbox("📄 Preview DataFrame")

    if show_preview:
        with st.expander("📘 DataFrame Preview"):
            st.dataframe(df_selected, use_container_width=True)


# ================================
# HEADER & INSTRUCTIONS
# ================================
st.markdown("""
### 💬 Inventory Chatbot  
Ask questions about inventory, availability, IMR, or stock status.

""")

if df_name == "stock":
    st.info(
        "For **stock_df**, your question must contain exactly ONE status:\n"
        "**Invoiced**, **Shipped**, **Landed**, or **Ordered**."
    )


# ================================
# QUICK EXAMPLES
# ================================
with st.expander("📌 Example questions you can ask"):
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Ordered packs for 20373 (Nov 2025)"):
            st.session_state["preset_question"] = (
                "how many Ordered packs for 20373 for November 2025?"
            )

        if st.button("Landed 20588 (Sep 2025)"):
            st.session_state["preset_question"] = (
                "how many Landed packs for 20588 for September 2025?"
            )

    with col2:
        if st.button("Invoiced packs for 20373 (Nov 2025)"):
            st.session_state["preset_question"] = (
                "how many Invoiced packs for 20373 for November 2025?"
            )

        if st.button("Availability for item 20246"):
            st.session_state["preset_question"] = "how many available for 20246?"


# ================================
# CHAT DISPLAY
# ================================
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)


# ================================
# INPUT FIELD
# ================================
preset = st.session_state.get("preset_question", "")

question = st.chat_input(
    "Ask your question...",
    key="user_query_input",
    placeholder=(
        "Example: How many Landed packs for 20373 for September 2025?"
        if df_name == "stock"
        else "Example: How many AVAILABLE for ITEM # 20373?"
    ),
)

# If the user pressed a quick button, preload it
if preset and not question:
    question = preset
    st.session_state["preset_question"] = ""


# ================================
# PROCESS QUESTION
# ================================
if question:

    # Add user message to chat history
    st.session_state.history.append(("user", question))

    explanation, plan, result = answer_question(question, df_name)

    # Display assistant message
    st.session_state.history.append(("assistant", explanation))
    st.chat_message("assistant").write(explanation)

    # Debug info section
    if debug_mode and plan is not None:
        with st.expander("🛠 Debug Info"):
            st.subheader("Generated JSON Plan")
            st.json(plan)

            st.subheader("Raw Result")
            if isinstance(result, (pd.DataFrame, pd.Series)):
                st.dataframe(result)
            else:
                st.write(result)

    # Ensure UI updates
    st.experimental_rerun()
