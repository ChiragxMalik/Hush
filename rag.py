import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Please set your GROQ_API_KEY in the .env file.")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# ---- Load Vector DB and Retriever ----
def get_retriever():
    """Load vector database with better error handling and verification"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Check if vectordb directory exists
        if not os.path.exists("vectordb"):
            raise FileNotFoundError("Vector database directory 'vectordb' not found. Please run ingest.py first.")
        
        # Initialize ChromaDB with simpler configuration
        vectordb = Chroma(
            persist_directory="vectordb", 
            embedding_function=embeddings
        )
        
        # Verify the database has content with simpler approach
        try:
            # Use similarity_search instead of get_relevant_documents for testing
            test_results = vectordb.similarity_search("test", k=1)
            if not test_results:
                print("Warning: Vector database appears to be empty. Please check your ingestion process.")
        except Exception as e:
            print(f"Warning: Could not verify vector database content: {e}")
        
        # Return retriever with simpler configuration (removed fetch_k which causes error)
        return vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Removed fetch_k parameter
        )
    except Exception as e:
        print(f"Error initializing vector database: {e}")
        raise

# Initialize retriever globally but with error handling
try:
    retriever = get_retriever()
    print("Vector database loaded successfully!")
except Exception as e:
    print(f"Failed to load vector database: {e}")
    retriever = None

# ---- Mode-Specific Prompt Engineering ----
def get_mode_specific_prompt(mode: str) -> str:
    """Return specialized prompts based on user's emotional state"""
    
    base_context = """
You are "Hush", a warm, experienced clinical psychologist with deep expertise in trauma-informed care, CBT, DBT, and mindfulness practices. You've helped thousands of people navigate difficult emotions.

CORE PRINCIPLES:
- Validate feelings before offering solutions
- Use evidence-based therapeutic techniques
- Speak as a caring human, not a clinical manual
- Keep responses focused and actionable (2-4 sentences)
- Match the user's emotional intensity appropriately
"""

    mode_specifics = {
        "Calm": {
            "tone": "Maintain this peaceful state with gentle guidance",
            "approach": """
CALM MODE APPROACH:
- Reinforce positive emotional regulation
- Suggest mindfulness practices or self-reflection
- Help them process thoughts clearly
- Use warm, steady language
- Focus on maintaining emotional balance and growth

Example tone: "I can sense the calm in your words. This is a perfect time to..."
"""
        },
        
        "Anxious": {
            "tone": "Ground them with steady, reassuring presence",
            "approach": """
ANXIOUS MODE APPROACH:
- Prioritize grounding techniques (5-4-3-2-1, breathing)
- Validate their anxiety as normal and manageable
- Break overwhelming thoughts into smaller pieces
- Use calm, steady language to model regulation
- Offer immediate coping strategies

Example tone: "I understand that anxiety feels overwhelming right now. Let's start with one simple thing you can control..."
"""
        },
        
        "Low Mood": {
            "tone": "Gentle encouragement with compassionate understanding",
            "approach": """
LOW MOOD MODE APPROACH:
- Validate their pain without minimizing it
- Look for small, achievable steps forward
- Use behavioral activation principles (gentle action)
- Offer hope while acknowledging current difficulty
- Focus on self-compassion and small wins

Example tone: "I hear how heavy things feel right now. That takes courage to share..."
"""
        },
        
        "Panic": {
            "tone": "Immediate stabilization with urgent but calming presence",
            "approach": """
PANIC MODE APPROACH:
- Immediate grounding and breathing techniques
- Normalize panic as temporary and survivable
- Use short, clear, directive language
- Focus on present moment safety
- Prioritize physiological calming first

Example tone: "Right now, you're safe. Let's slow this down together. Can you feel your feet on the ground?"
"""
        }
    }
    
    selected = mode_specifics.get(mode, mode_specifics["Calm"])
    
    return f"""
{base_context}

{selected['approach']}

RESPONSE TONE: {selected['tone']}

CONTEXT FROM THERAPEUTIC KNOWLEDGE BASE:
{{context}}

USER'S MESSAGE: {{question}}

Respond as Hush would - with professional warmth, evidence-based insight, and the specific approach needed for their current emotional state:
"""

# ---- Response Generator ----
def get_response(user_input: str, mode: str) -> str:
    """Generate contextual response using vector database and mode-specific prompting"""
    
    if not retriever:
        return "I'm sorry, but I'm having trouble accessing my knowledge base right now. Please make sure the vector database is set up properly by running ingest.py."
    
    try:
        # Get relevant context from vector database using invoke (new method)
        try:
            relevant_docs = retriever.invoke(user_input)
        except Exception:
            # Fallback to deprecated method if invoke doesn't work
            relevant_docs = retriever.get_relevant_documents(user_input)
        
        if not relevant_docs:
            print("Warning: No relevant documents found in vector database")
            context = "No specific therapeutic guidance found in knowledge base. Relying on general therapeutic principles."
        else:
            # Combine context from multiple documents with source info
            context_parts = []
            for i, doc in enumerate(relevant_docs[:3]):  # Use top 3 most relevant
                source = doc.metadata.get('source_file', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                context_parts.append(f"Source {i+1} ({source}, page {page}):\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            print(f"Using context from {len(relevant_docs)} relevant documents")
        
        # Get mode-specific prompt template
        prompt_template = get_mode_specific_prompt(mode)
        
        # Format the prompt with context and question
        formatted_prompt = prompt_template.format(
            context=context,
            question=user_input
        )
        
        # Generate response with mode-appropriate parameters
        temperature_map = {
            "Calm": 0.6,      # Slightly more creative for reflection
            "Anxious": 0.4,   # More consistent for grounding
            "Low Mood": 0.5,  # Balanced for encouragement
            "Panic": 0.3      # Most consistent for crisis support
        }
        
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",  # Changed to a more stable model
            messages=[
                {
                    "role": "system",
                    "content": "You are Hush, a skilled and empathetic clinical psychologist. Respond with warmth, professionalism, and evidence-based therapeutic guidance."
                },
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ],
            temperature=temperature_map.get(mode, 0.5),
            max_tokens=400,  # Changed from max_completion_tokens
            top_p=0.85,
            stream=False,
            stop=None
        )
        
        response = completion.choices[0].message.content
        
        # Add debug info (remove in production)
        print(f"Mode: {mode}, Temperature: {temperature_map.get(mode, 0.5)}")
        print(f"Context length: {len(context)} characters")
        
        return response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"I'm sorry, I encountered an error while trying to help. Please try again in a moment. If the problem persists, you may need to check your vector database setup."

# ---- Debug Functions ----
def test_vector_db():
    """Test function to verify vector database is working"""
    if not retriever:
        print("❌ Vector database not loaded")
        return False
        
    try:
        test_queries = ["anxiety", "depression", "breathing", "mindfulness"]
        for query in test_queries:
            # Use invoke method instead of deprecated get_relevant_documents
            try:
                docs = retriever.invoke(query)
            except Exception:
                # Fallback to similarity_search directly
                vectordb = Chroma(
                    persist_directory="vectordb", 
                    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                )
                docs = vectordb.similarity_search(query, k=4)
            
            print(f"Query '{query}': Found {len(docs)} relevant documents")
            if docs:
                print(f"  First result: {docs[0].page_content[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Vector database test failed: {e}")
        return False

def test_mode_responses(test_message="I'm feeling overwhelmed"):
    """Test different responses across all modes"""
    modes = ["Calm", "Anxious", "Low Mood", "Panic"]
    
    print(f"\nTesting responses for: '{test_message}'\n")
    
    for mode in modes:
        print(f"=== {mode.upper()} MODE ===")
        try:
            response = get_response(test_message, mode)
            print(f"Response: {response}\n")
        except Exception as e:
            print(f"Error in {mode} mode: {e}\n")

if __name__ == "__main__":
    # Run tests
    print("Testing vector database...")
    test_vector_db()
    
    print("\nTesting mode-specific responses...")
    test_mode_responses()