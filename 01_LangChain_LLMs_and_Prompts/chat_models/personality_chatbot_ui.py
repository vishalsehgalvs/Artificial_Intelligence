# UIchatbot.py — A mood-based chatbot with a proper web UI using Streamlit
# -------------------------------------------------------------------------
# This is the full UI version of chatbot.py
# Instead of typing in a terminal, you get a proper chat interface in your browser.
#
# Run it with:  streamlit run UIchatbot.py
# Requirements: pip install streamlit langchain-google-genai python-dotenv

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


# -------------------- Model --------------------
@st.cache_resource   # create the model once, reuse it across reruns
def get_model():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.9)

model = get_model()


# -------------------- Page Setup --------------------
st.set_page_config(
    page_title="AI Mood Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Mood Chatbot")
st.caption("Pick an AI personality and start chatting")


# -------------------- Personality Selector --------------------
mode_choice = st.radio(
    "Choose AI Personality:",
    ["😡 Angry", "😂 Funny", "🧙 Wise"],
    horizontal=True
)

if mode_choice == "😡 Angry":
    system_prompt = "You are a grumpy, impatient AI. You respond with frustration and sarcasm."
elif mode_choice == "😂 Funny":
    system_prompt = "You are a hilarious AI comedian. Every response includes a joke or pun."
else:
    system_prompt = "You are a wise, calm AI mentor. You give thoughtful, balanced advice."


# -------------------- Session State (Memory) --------------------
# st.session_state persists data across Streamlit reruns within a session
# This is how we maintain the chat history while the user keeps chatting

if "messages" not in st.session_state or st.session_state.get("current_mode") != mode_choice:
    # Reset history if personality changed or first load
    st.session_state.current_mode = mode_choice
    st.session_state.messages = [SystemMessage(content=system_prompt)]


# -------------------- Display Chat History --------------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
    # SystemMessage is never shown to the user — it's backstage instructions


# -------------------- User Input --------------------
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add to history and display
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    # Call the model with full history
    with st.spinner("Thinking..."):
        response = model.invoke(st.session_state.messages)

    # Add AI reply to history and display
    st.session_state.messages.append(AIMessage(content=response.content))
    with st.chat_message("assistant"):
        st.write(response.content)


# -------------------- Reset Button --------------------
st.divider()
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 Reset Chat"):
        st.session_state.messages = [SystemMessage(content=system_prompt)]
        st.rerun()
