# ❤️ HeartHush – AI Mental Health Chatbot

HeartHush (**Hush**) is a therapeutic chatbot that adapts to your emotional state and provides supportive, evidence-based responses.
It combines **Retrieval-Augmented Generation (RAG)** with trusted resources like *Feeling Good*, *Cognitive Behavior Therapy: Basics and Beyond*, and *The Body Keeps the Score* to ground its replies in well-established therapeutic methods.

---

## ✨ Features

* **4 Emotional Modes**:

  * 🧘 Calm – reflective, gentle tone
  * 😨 Anxious – grounding, reassuring tone
  * 😔 Low Mood – validating, hopeful tone
  * 😱 Panic – urgent, stabilizing tone

* **RAG with Psychology Books**: responses are grounded in CBT, DBT, mindfulness, and trauma-informed care.

* **Dynamic Themes**: each mode also changes the chatbot’s **visual theme** to match the tone.

* **Session Awareness**: remembers patterns within a session to adapt its guidance.

---

## 📸 Demo
<img width="1920" height="931" alt="Screenshot (213)" src="https://github.com/user-attachments/assets/15b564c3-35b9-4b8d-ad6b-1f01a2f3033e" />
<img width="1920" height="933" alt="Screenshot (209)" src="https://github.com/user-attachments/assets/d365539f-872b-40ca-8b4f-b2bec6305574" />
<img width="1920" height="931" alt="Screenshot (210)" src="https://github.com/user-attachments/assets/5076127a-8297-4929-b541-1a6c328a8c4d" />
<img width="1920" height="937" alt="Screenshot (211)" src="https://github.com/user-attachments/assets/5eca3517-6a3e-41e2-9811-9b435ed7a177" />
<img width="1920" height="934" alt="Screenshot (212)" src="https://github.com/user-attachments/assets/dc489ba2-263d-4246-b3a8-9b14c39cbf07" />



---

## ⚙️ Installation

1. **Clone repo**

   ```bash
   git clone https://github.com/yourusername/hearthush.git
   cd hush
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Vector Database**

   * Extract the provided `vectordb.rar` into the project folder.
   * This contains the embeddings for the therapy books.

4. **Environment Variables**
   Create a `.env` file in the root directory:

   ```env
   GROQ_API_KEY=your_api_key_here
   ```

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

## 🛠 Tech Stack

* **Frontend/UI**: Streamlit
* **Backend**: Python, LangChain, ChromaDB
* **LLM Provider**: [Groq](https://groq.com/)
* **Embeddings**: Sentence-Transformers
* **Data**: curated psychotherapy books + research papers

---

## 📚 Knowledge Sources

HeartHush’s RAG pipeline is built on well-respected psychology resources, including:

* *Feeling Good: The New Mood Therapy* – David D. Burns
* *Cognitive Behavior Therapy: Basics and Beyond* – Judith S. Beck
* *The Body Keeps the Score* – Bessel van der Kolk
* Plus additional CBT, DBT, and mindfulness-based texts

---

## 🚀 Roadmap

* [ ] Multi-session memory persistence
* [ ] Voice input/output mode
* [ ] Mobile-friendly deployment
* [ ] More therapy frameworks (ACT, positive psychology)

---

## ⚠️ Disclaimer

HeartHush is **not a substitute for professional help**.
If you are in crisis or need urgent support, please contact your local emergency number or mental health hotline.

---

👉 Do you also want me to add a **Usage Examples section** (like showing how the *same question* produces 4 different mode responses)? That would make your demo really stand out.
