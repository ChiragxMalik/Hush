import streamlit as st
import uuid
from datetime import datetime
from rag import get_response, test_vector_db  # Import RAG logic and test function

# ---- Helper Functions ----
def get_initial_message(mode):
    messages = {
        "Calm": "Hello! I sense you're in a peaceful space. I'm Hush, here to support your reflections and growth. What would you like to share today?",
        "Anxious": "Hi there. It sounds like you may be feeling anxious, and that's completely okay. You're safe here — would you like to talk about what's been on your mind?",
        "Low Mood": "Hello. I know things might feel heavy right now, and reaching out shows real strength. I'm here with you — what's been weighing on your heart?",
        "Panic": "Hey, I'm right here with you. If panic feels overwhelming, know that you're safe in this moment. Let's take it one breath at a time — what’s happening for you right now?"
    }
    return messages.get(mode, messages["Calm"])


# ---- Enhanced Modes with Better Descriptions ----
MODES = {
    "Calm": {
        "icon": "🧘", 
        "accent": "#4A90E2", 
        "description": "Feeling peaceful and want to reflect or maintain balance",
        "prompt_style": "Reflective and growth-oriented guidance"
    },
    "Anxious": {
        "icon": "😟", 
        "accent": "#F39C12", 
        "description": "Feeling worried, nervous, or overwhelmed by thoughts",
        "prompt_style": "Grounding and reassuring support"
    },
    "Low Mood": {
        "icon": "😔", 
        "accent": "#8E44AD", 
        "description": "Feeling sad, down, or lacking motivation",
        "prompt_style": "Gentle encouragement and hope-building"
    },
    "Panic": {
        "icon": "😨", 
        "accent": "#E74C3C", 
        "description": "Feeling intense fear or having a panic attack",
        "prompt_style": "Immediate grounding and safety techniques"
    }
}

# ---- Session State ----
if "mode" not in st.session_state:
    st.session_state.mode = "Calm"
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db_status" not in st.session_state:
    # Test vector database on first load
    st.session_state.db_status = test_vector_db()

# ---- Header ----
st.set_page_config(
    page_title="HeartHush - AI Companion", 
    page_icon="❤️",
    layout="wide"
)

st.title("❤️HeartHush - Your Empathetic AI Companion")

# ---- Database Status Indicator ----
if st.session_state.db_status:
    st.success("✅ Therapeutic knowledge base loaded successfully")
else:
    st.error("⚠️ Knowledge base not available. Please run ingest.py to load therapeutic content.")

# ---- Mode Selector with Better UX ----
st.markdown("### How are you feeling right now?")
st.markdown("*Choose the mode that best matches your current emotional state:*")

# Create mode selection with better styling
col1, col2, col3, col4 = st.columns(4)

mode_cols = [col1, col2, col3, col4]
mode_names = list(MODES.keys())

for i, (mode_name, col) in enumerate(zip(mode_names, mode_cols)):
    mode_info = MODES[mode_name]
    is_selected = st.session_state.mode == mode_name
    
    with col:
        # Create a more informative button
        button_style = "primary" if is_selected else "secondary"
        if st.button(
            f"{mode_info['icon']} {mode_name}", 
            use_container_width=True,
            type=button_style,
            help=mode_info['description']
        ):
            st.session_state.mode = mode_name
            st.rerun()

# Show current mode description
current_mode = MODES[st.session_state.mode]
st.info(f"**Current Mode: {current_mode['icon']} {st.session_state.mode}** - {current_mode['description']}")

st.markdown("---")

# ---- Enhanced Chat UI Styling ----
st.markdown(
    f"""
    <style>
        .user-chat-bubble {{
            background: linear-gradient(135deg, {current_mode['accent']}40, {current_mode['accent']}60);
            color: #ffffff;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 15px;
            border-top-right-radius: 5px;
            margin-left: 20%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .bot-chat-bubble {{
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: #ffffff;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 15px;
            border-top-left-radius: 5px;
            margin-right: 20%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            font-size: 0.7rem;
            color: #bdc3c7;
            margin-top: 0.5rem;
            font-style: italic;
        }}
        .mode-indicator {{
            background-color: {current_mode['accent']};
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            display: inline-block;
            margin-bottom: 0.5rem;
        }}
        .stTextInput > div > div > input {{
            border-radius: 25px;
            border: 2px solid {current_mode['accent']};
            padding: 1rem;
        }}
        .stButton > button {{
            border-radius: 25px;
            background-color: {current_mode['accent']};
            border: none;
            padding: 0.5rem 2rem;
            font-weight: bold;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Initial Bot Message ----
if len(st.session_state.chat_history) == 0:
    initial_message = get_initial_message(st.session_state.mode)
    st.markdown(
        f"<div class='bot-chat-bubble'>"
        f"<div class='mode-indicator'>{current_mode['icon']} {st.session_state.mode} Mode</div>"
        f"<strong>Hush:</strong> {initial_message}"
        f"<div class='timestamp'>{datetime.now().strftime('%I:%M %p')}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

# ---- Show Chat History ----
for entry in st.session_state.chat_history:
    # User message
    st.markdown(
        f"<div class='user-chat-bubble'>"
        f"<strong>You:</strong> {entry['user']}"
        f"<div class='timestamp'>{entry['time']}</div>"
        f"</div>", 
        unsafe_allow_html=True
    )
    
    # Bot message with mode indicator
    st.markdown(
        f"<div class='bot-chat-bubble'>"
        f"<div class='mode-indicator'>{MODES[entry.get('mode', 'Calm')]['icon']} {entry.get('mode', 'Calm')} Mode</div>"
        f"<strong>Hush:</strong> {entry['bot']}"
        f"<div class='timestamp'>{entry['time']}</div>"
        f"</div>", 
        unsafe_allow_html=True
    )

# ---- Input Form ----
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        f"Share what's on your mind... (responding in {st.session_state.mode} mode)",
        label_visibility="collapsed", 
        placeholder=f"I'm here to listen and support you in {st.session_state.mode} mode...",
        key="message_input"
    )
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        submit_button = st.form_submit_button("Send", use_container_width=True)
    with col3:
        clear_button = st.form_submit_button("Clear Chat", use_container_width=True)

# ---- Response Handler ----
if submit_button and user_input.strip():
    now = datetime.now().strftime("%I:%M %p")
    
    with st.spinner(f"Hush is responding in {st.session_state.mode} mode..."):
        try:
            bot_response = get_response(user_input, st.session_state.mode)  # Pass mode instead of prompt_style
            st.session_state.chat_history.append({
                "user": user_input,
                "bot": bot_response,
                "time": now,
                "mode": st.session_state.mode  # Store the mode used
            })
            st.rerun()
        except Exception as e:
            st.error(f"I apologize, but I encountered an error: {str(e)}")
            st.error("Please try again in a moment.")

# ---- Clear Chat Handler ----
if clear_button:
    st.session_state.chat_history = []
    st.rerun()

# ---- Sidebar with App Info ----
with st.sidebar:
    st.markdown("### About HeartHush")
    st.markdown("Your AI powered mental wellness companion. HeartHush adapts to your emotional state, offering gentle, personalized support.")
    
    st.markdown("### Current Session")
    st.markdown(f"**Mode:** {st.session_state.mode}")
    st.markdown(f"**Messages:** {len(st.session_state.chat_history)}")
    st.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    
    st.markdown("### HeartHush draws from proven approaches:")
    st.markdown("- Trauma-informed care")
    st.markdown("- Cognitive Behavioral Therapy (CBT)")
    st.markdown("- Dialectical Behavior Therapy (DBT)")
    st.markdown("- Mindfulness practices")
    
    if st.button("Reset Session", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# ---- Footer ----
st.markdown("---")
st.markdown("**Important:** HeartHush provides emotional support and therapeutic guidance but is not a replacement for professional mental health treatment. If you're experiencing a mental health crisis, please contact emergency services or a crisis helpline immediately.")
