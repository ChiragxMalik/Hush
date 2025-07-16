import streamlit as st
import uuid
from datetime import datetime
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---- Modes ----
MODES = {
    "Calm": {"icon": "🧘", "accent": "#ACC7D9", "prompt_style": "soft, supportive tone"},
    "Low Mood": {"icon": "😔", "accent": "#A77F2E", "prompt_style": "gentle, mood-lifting tone"},
    "Anxious": {"icon": "😟", "accent": "#B451A3", "prompt_style": "grounded, reassuring tone"},
    "Panic": {"icon": "😨", "accent": "#C31F49", "prompt_style": "urgent but calming tone"}
}

# ---- Session State ----
if "mode" not in st.session_state:
    st.session_state.mode = "Calm"
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Mode Selector UI ----
col1, col2, col3, col4 = st.columns(4)
if col1.button(f"{MODES['Calm']['icon']} Calm"):
    st.session_state.mode = "Calm"
if col2.button(f"{MODES['Anxious']['icon']} Anxious"):
    st.session_state.mode = "Anxious"
if col3.button(f"{MODES['Low Mood']['icon']} Low Mood"):
    st.session_state.mode = "Low Mood"
if col4.button(f"{MODES['Panic']['icon']} Panic"):
    st.session_state.mode = "Panic"

# ---- Accent Color CSS ----
st.markdown(
    f"""
    <style>
        .stButton > button {{
            background-color: {MODES[st.session_state.mode]['accent']};
            color: white;
            border: none;
            border-radius: 10px;
        }}
        .stTextInput > div > div > input {{
            border: 1px solid {MODES[st.session_state.mode]['accent']};
            border-radius: 8px;
            color: white;
            background-color: #1a1a2e;
        }}
        .chat-bubble {{
            background-color: #26264c;
            color: white;
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 10px;
        }}
        .timestamp {{
            font-size: 0.75rem;
            color: #999;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Load Vector DB and QA Chain ----
@st.cache_resource
def get_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="vectordb", embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    return retriever

retriever = get_chain()

#prompttemplate
prompt_template = PromptTemplate(
    input_variables=["context", "question", "tone"],
    template="""
You are "Hush", a licensed clinical psychologist and compassionate mental health expert. You are trained in cognitive behavioral therapy (CBT), mindfulness, trauma-informed care, and crisis management. 

You are responding in a professional, emotionally intelligent, and empathetic tone — specifically adapted to the user's current mental mode: {tone}. Avoid sounding like a chatbot or motivational speaker. Do not nag or give generic affirmations. Provide thoughtful, structured, and validating responses based on psychological best practices.

Use the context below (from reference materials and therapy texts) to inform your answer.

---
Context:
{context}

User:
{question}

Respond with a thoughtful and therapeutic answer:
"""
)

# ---- Generate Bot Response ----
def get_response(user_input, tone):
    llm = Ollama(model="deepseek-r1:1.5b", temperature=0.7)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={
            "prompt": prompt_template.partial(tone=tone)
        }
    )
    return qa.run(user_input)

# ---- Initial Bot Message ----
if len(st.session_state.chat_history) == 0:
    st.markdown(f"<div class='chat-bubble'>Hello! I'm Hush, your empathetic AI companion. I'm here to listen, support, and help you navigate your thoughts and feelings. How are you doing today?<div class='timestamp'>{datetime.now().strftime('%I:%M %p')}</div></div>", unsafe_allow_html=True)

# ---- Show Chat History ----
for entry in st.session_state.chat_history:
    st.markdown(f"<div class='chat-bubble'><b>You:</b> {entry['user']}<div class='timestamp'>{entry['time']}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble'><b>Hush:</b> {entry['bot']}<div class='timestamp'>{entry['time']}</div></div>", unsafe_allow_html=True)

# ---- Input Field ----
user_input = st.text_input("Share what's on your mind...", label_visibility="collapsed", placeholder="Share what’s on your mind...")

# ---- Response Handler ----
if user_input:
    now = datetime.now().strftime("%I:%M %p")
    tone = MODES[st.session_state.mode]['prompt_style']
    bot_response = get_response(user_input, tone)

    st.session_state.chat_history.append({
        "user": user_input,
        "bot": bot_response,
        "time": now
    })

    


# ---- Footer ----
st.markdown("<hr>")
st.markdown("<small>HeartHush provides emotional support but is not a replacement for professional therapy.</small>")
