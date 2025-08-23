import os
import re
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Ignore warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Please set your GROQ_API_KEY in the .env file.")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Global session memory (in production, use Redis/database)
SESSION_MEMORY = {}

@dataclass
class EmotionalContext:
    core_emotion: str
    secondary_emotions: List[str]
    cognitive_patterns: List[str]
    intervention_type: str
    urgency_level: str
    timestamp: datetime

@dataclass
class SessionPattern:
    dominant_emotions: Dict[str, int]  # emotion -> count
    recurring_patterns: List[str]
    session_trend: str
    last_updated: datetime
    total_interactions: int

# ---- Session Memory Management ----
def get_session_id():
    """Get session ID - in production, this would come from your session management"""
    # For now, using a simple approach - in your app.py, you'd pass the actual session_id
    return "default_session"

def update_session_memory(session_id: str, emotional_context: EmotionalContext):
    """Track emotional patterns across the session"""
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = SessionPattern(
            dominant_emotions={},
            recurring_patterns=[],
            session_trend="exploring",
            last_updated=datetime.now(),
            total_interactions=0
        )
    
    session = SESSION_MEMORY[session_id]
    session.total_interactions += 1
    session.last_updated = datetime.now()
    
    # Track emotion frequency
    if emotional_context.core_emotion in session.dominant_emotions:
        session.dominant_emotions[emotional_context.core_emotion] += 1
    else:
        session.dominant_emotions[emotional_context.core_emotion] = 1
    
    # Track recurring cognitive patterns
    for pattern in emotional_context.cognitive_patterns:
        if pattern not in session.recurring_patterns:
            session.recurring_patterns.append(pattern)
    
    # Determine session trend
    if session.total_interactions >= 3:
        most_common_emotion = max(session.dominant_emotions.items(), key=lambda x: x[1])
        if most_common_emotion[1] >= 3:  # Same emotion 3+ times
            session.session_trend = f"persistent_{most_common_emotion[0]}"
        elif len(set(session.dominant_emotions.keys())) == 1:
            session.session_trend = f"focused_{most_common_emotion[0]}"
        else:
            session.session_trend = "varied_exploration"

def get_session_context(session_id: str) -> Optional[SessionPattern]:
    """Get current session patterns for personalization"""
    return SESSION_MEMORY.get(session_id)

# ---- Enhanced Emotional Analysis ----
def analyze_emotional_context(user_input: str, session_id: str = None) -> tuple:
    """Extract emotional context and determine intervention approach"""
    
    # Emotional concept mapping for better retrieval
    emotion_keywords = {
        "loneliness": ["nobody loves me", "alone", "isolated", "no friends", "abandoned", "rejected", "left out"],
        "worthlessness": ["not good enough", "failure", "useless", "waste of space", "disappointment", "pathetic"],
        "anxiety": ["worried", "scared", "panic", "can't stop thinking", "what if", "nervous", "overwhelmed"],
        "depression": ["hopeless", "empty", "numb", "tired", "no point", "dark", "can't go on"],
        "anger": ["furious", "hate", "unfair", "rage", "frustrated", "mad", "can't stand"],
        "shame": ["embarrassed", "humiliated", "disgusted with myself", "ashamed", "mortified"],
        "grief": ["miss", "lost", "gone", "died", "death", "goodbye", "never see again"],
        "fear": ["terrified", "afraid", "scared", "frightening", "nightmare", "dangerous"]
    }
    
    # Intervention strategies mapped to emotions
    interventions = {
        "loneliness": {
            "primary": "connection_building",
            "techniques": ["behavioral_activation", "social_skills", "attachment_work"],
            "search_terms": ["loneliness therapy", "building connections", "social anxiety", "attachment styles", "isolation"]
        },
        "worthlessness": {
            "primary": "cognitive_restructuring", 
            "techniques": ["CBT", "self_compassion", "core_beliefs"],
            "search_terms": ["self-worth therapy", "cognitive restructuring", "negative self-talk", "core beliefs", "self-esteem"]
        },
        "anxiety": {
            "primary": "anxiety_management",
            "techniques": ["grounding", "breathing", "exposure", "mindfulness"],
            "search_terms": ["anxiety techniques", "grounding exercises", "panic management", "mindfulness anxiety", "worry"]
        },
        "depression": {
            "primary": "depression_support",
            "techniques": ["behavioral_activation", "cognitive_therapy", "hope_building"],
            "search_terms": ["depression therapy", "behavioral activation", "hopelessness", "mood disorders", "sadness"]
        },
        "anger": {
            "primary": "anger_management",
            "techniques": ["emotion_regulation", "communication", "boundary_setting"],
            "search_terms": ["anger management", "emotion regulation", "frustration", "conflict resolution", "boundaries"]
        }
    }
    
    # Normalize input for analysis
    input_lower = user_input.lower()
    
    # Match emotional concepts
    detected_emotions = []
    emotion_scores = {}
    
    for emotion, keywords in emotion_keywords.items():
        matches = sum(1 for keyword in keywords if keyword in input_lower)
        if matches > 0:
            detected_emotions.append(emotion)
            emotion_scores[emotion] = matches
    
    # Sort by number of matches to get primary emotion
    if detected_emotions:
        primary_emotion = max(detected_emotions, key=lambda e: emotion_scores[e])
    else:
        primary_emotion = "general_distress"  # fallback
    
    # Analyze cognitive patterns
    cognitive_patterns = []
    if any(word in input_lower for word in ["always", "never", "everyone", "nobody", "all", "none"]):
        cognitive_patterns.append("all_or_nothing_thinking")
    
    if any(phrase in input_lower for phrase in ["should", "must", "have to", "supposed to", "need to"]):
        cognitive_patterns.append("should_statements")
        
    if any(phrase in input_lower for phrase in ["it's my fault", "i'm to blame", "because of me", "i caused"]):
        cognitive_patterns.append("personalization")
        
    if any(phrase in input_lower for phrase in ["what if", "probably will", "going to happen", "disaster", "terrible"]):
        cognitive_patterns.append("catastrophizing")
        
    if any(phrase in input_lower for phrase in ["can't do", "impossible", "too hard", "will fail"]):
        cognitive_patterns.append("helplessness")
    
    # Determine intervention type and urgency
    intervention_info = interventions.get(primary_emotion, {
        "primary": "supportive_counseling",
        "techniques": ["validation", "exploration"],
        "search_terms": ["general therapy", "emotional support", "counseling techniques"]
    })
    
    # Assess urgency
    crisis_keywords = ["kill myself", "end it all", "can't go on", "no way out", "suicide", "die"]
    high_urgency = ["can't cope", "breaking point", "can't handle", "too much", "emergency"]
    
    if any(keyword in input_lower for keyword in crisis_keywords):
        urgency = "crisis"
    elif any(keyword in input_lower for keyword in high_urgency):
        urgency = "high"
    else:
        urgency = "moderate"
    
    emotional_context = EmotionalContext(
        core_emotion=primary_emotion,
        secondary_emotions=[e for e in detected_emotions if e != primary_emotion],
        cognitive_patterns=cognitive_patterns,
        intervention_type=intervention_info["primary"],
        urgency_level=urgency,
        timestamp=datetime.now()
    )
    
    # Update session memory if session_id provided
    if session_id:
        update_session_memory(session_id, emotional_context)
    
    return emotional_context, intervention_info["search_terms"]

# ---- Load Vector DB and Retriever ----
def get_retriever():
    """Load vector database with better error handling and verification"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Check if vectordb directory exists
        if not os.path.exists("vectordb"):
            raise FileNotFoundError("Vector database directory 'vectordb' not found. Please run ingest.py first.")
        
        # Get all available collections
        import chromadb
        client = chromadb.PersistentClient(path="vectordb")
        collections = client.list_collections()
        
        if not collections:
            raise ValueError("No collections found in vector database. Please run ingest.py first.")
        
        print(f"Found {len(collections)} collections in vector database:")
        for col in collections:
            print(f"  - {col.name}: {col.count()} documents")
        
        # Initialize ChromaDB with the first collection that has documents
        # We'll use a collection that actually has content
        for col in collections:
            if col.count() > 0:
                print(f"Using collection: {col.name} with {col.count()} documents")
                vectordb = Chroma(
                    persist_directory="vectordb", 
                    embedding_function=embeddings,
                    collection_name=col.name
                )
                break
        else:
            # Fallback to default if no collections have documents
            vectordb = Chroma(
                persist_directory="vectordb", 
                embedding_function=embeddings
            )
        
        # Verify the database has content
        try:
            test_results = vectordb.similarity_search("therapy", k=1)
            if not test_results:
                print("Warning: Vector database appears to be empty. Please check your ingestion process.")
            else:
                print(f"Successfully loaded vector database with {vectordb._collection.count()} documents")
        except Exception as e:
            print(f"Warning: Could not verify vector database content: {e}")
        
        # Return retriever with enhanced configuration
        return vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Increased for better coverage, will be filtered later
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

# ---- Optimized Retrieval Function ----
def optimized_retrieval(user_input: str, emotional_context: EmotionalContext, search_terms: List[str]) -> str:
    """Optimized retrieval with single query and intelligent filtering"""
    
    if not retriever:
        return "Unable to access therapeutic knowledge base."
    
    try:
        # OPTIMIZATION 1: Create single combined query instead of multiple calls
        # Prioritize emotion-specific terms and add original input
        top_search_terms = search_terms[:2]  # Limit to top 2 emotion-specific terms
        combined_query = f"{' '.join(top_search_terms)} {user_input}"
        
        print(f"Optimized search query: '{combined_query}'")
        
        # Single retrieval call with higher k for filtering
        try:
            all_docs = retriever.invoke(combined_query)
        except Exception as e:
            print(f"Retriever failed, trying direct vectordb search: {e}")
            # Fallback to direct vectordb search with proper collection
            import chromadb
            client = chromadb.PersistentClient(path="vectordb")
            collections = client.list_collections()
            
            # Find a collection with documents
            for col in collections:
                if col.count() > 0:
                    vectordb = Chroma(
                        persist_directory="vectordb", 
                        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                        collection_name=col.name
                    )
                    all_docs = vectordb.similarity_search(combined_query, k=10)
                    break
            else:
                # If no collections found, return empty
                all_docs = []
        
        # OPTIMIZATION 2: Intelligent filtering and ranking
        scored_docs = []
        for doc in all_docs:
            score = 0.0
            content_lower = doc.page_content.lower()
            
            # Score based on emotion relevance
            for term in search_terms[:3]:  # Check top 3 search terms
                if term.lower() in content_lower:
                    score += 2.0
            
            # Score based on cognitive patterns
            for pattern in emotional_context.cognitive_patterns:
                pattern_words = pattern.replace("_", " ").lower()
                if pattern_words in content_lower:
                    score += 1.5
            
            # Score based on intervention type
            intervention_keywords = {
                "connection_building": ["connection", "relationship", "social", "attachment"],
                "cognitive_restructuring": ["cognitive", "thoughts", "beliefs", "thinking"],
                "anxiety_management": ["anxiety", "grounding", "breathing", "calm"],
                "depression_support": ["depression", "mood", "behavioral", "activation"]
            }
            
            intervention_words = intervention_keywords.get(emotional_context.intervention_type, [])
            for word in intervention_words:
                if word in content_lower:
                    score += 1.0
            
            scored_docs.append((doc, score))
        
        # Sort by score and take top 6
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        selected_docs = [doc for doc, score in scored_docs[:6]]
        
        # Format context
        context_parts = []
        for i, doc in enumerate(selected_docs):
            source = doc.metadata.get('source_file', 'Therapeutic Guidelines')
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(f"Source {i+1} ({source}, page {page}):\n{doc.page_content}")
        
        print(f"Optimized retrieval: Using {len(selected_docs)} top-scored documents")
        return "\n\n".join(context_parts)
        
    except Exception as e:
        print(f"Error in optimized retrieval: {e}")
        return "Unable to retrieve therapeutic context at this time."

# ---- Response Length Controller ----
def control_response_length(response: str, target_sentences: int = 4) -> str:
    """Post-process response to enforce length constraints"""
    
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= target_sentences:
        return response
    
    # If too long, intelligently truncate
    kept_sentences = sentences[:target_sentences]
    
    # Ensure the last sentence ends properly
    truncated = '. '.join(kept_sentences)
    if not truncated.endswith('.'):
        truncated += '.'
    
    print(f"Response truncated from {len(sentences)} to {target_sentences} sentences")
    return truncated

# ---- Session-Aware Prompt Engineering ----
def get_session_aware_prompt(mode: str, emotional_context: EmotionalContext, session_id: str = None) -> str:
    """Generate prompts with session awareness and personalization"""
    
    session_context = get_session_context(session_id) if session_id else None
    
    base_context = f"""
You are "Hush", a warm, experienced clinical psychologist with deep expertise in trauma-informed care, CBT, DBT, and mindfulness practices. You've helped thousands of people navigate difficult emotions.

CURRENT EMOTIONAL ANALYSIS:
- Primary emotion: {emotional_context.core_emotion}
- Secondary emotions: {', '.join(emotional_context.secondary_emotions) if emotional_context.secondary_emotions else 'None'}
- Cognitive patterns detected: {', '.join(emotional_context.cognitive_patterns) if emotional_context.cognitive_patterns else 'None identified'}
- Intervention approach: {emotional_context.intervention_type}
- Urgency level: {emotional_context.urgency_level}
"""

    # Add session context if available
    session_guidance = ""
    if session_context and session_context.total_interactions >= 2:
        dominant_emotion = max(session_context.dominant_emotions.items(), key=lambda x: x[1])[0] if session_context.dominant_emotions else None
        
        if session_context.session_trend.startswith("persistent_"):
            emotion = session_context.session_trend.replace("persistent_", "")
            session_guidance = f"""
SESSION PATTERN AWARENESS:
- This person has consistently expressed {emotion} across {session_context.total_interactions} interactions
- Consider deeper therapeutic work and acknowledge the pattern: "I notice we've talked about {emotion} feelings several times..."
- Focus on building longer-term coping strategies and exploring underlying themes
"""
        elif session_context.session_trend.startswith("focused_"):
            emotion = session_context.session_trend.replace("focused_", "")
            session_guidance = f"""
SESSION PATTERN AWARENESS:
- This session has focused primarily on {emotion}
- Build continuity: reference previous insights if appropriate
- Consider progressing the therapeutic conversation to deeper levels
"""
        elif session_context.recurring_patterns:
            session_guidance = f"""
SESSION PATTERN AWARENESS:
- Recurring cognitive patterns: {', '.join(session_context.recurring_patterns)}
- Gently acknowledge these patterns if relevant: "I'm noticing a pattern in how you think about..."
- Focus on pattern interruption and alternative perspectives
"""

    # Enhanced intervention guidance
    intervention_prompts = {
        "connection_building": """
LONELINESS/CONNECTION FOCUS:
- Validate their feelings of isolation without minimizing the pain
- Gently explore existing relationships and social connections
- Help identify small, safe steps toward connection
- Address any social anxiety or rejection fears
- Focus on quality over quantity of relationships
""",
        "cognitive_restructuring": """
WORTHLESSNESS/COGNITIVE FOCUS:
- Challenge all-or-nothing thinking patterns gently
- Help them identify evidence against negative self-beliefs
- Introduce self-compassion techniques
- Explore the origin of harsh self-criticism
- Offer balanced, realistic perspectives
""",
        "anxiety_management": """
ANXIETY/PANIC FOCUS:
- Prioritize immediate grounding and regulation techniques
- Validate that anxiety feels real and overwhelming
- Offer concrete coping strategies they can use right now
- Help them distinguish between realistic and anxious thoughts
- Focus on present-moment safety and control
""",
        "depression_support": """
DEPRESSION/HOPELESSNESS FOCUS:
- Acknowledge the depth of their pain without false optimism
- Look for small behavioral activation opportunities
- Help them remember their strengths and past resilience
- Address hopelessness while instilling gentle hope
- Focus on manageable next steps, not big changes
""",
        "supportive_counseling": """
GENERAL SUPPORT FOCUS:
- Provide warm validation and emotional attunement
- Help them feel truly heard and understood
- Explore their experience with curiosity and care
- Offer gentle guidance based on their specific needs
"""
    }

    mode_specifics = {
        "Calm": {
            "tone": "Maintain this peaceful state with gentle guidance and deeper exploration",
            "length": "3-4 sentences with thoughtful depth"
        },
        "Anxious": {
            "tone": "Ground them with steady, reassuring presence and immediate support",
            "length": "3-4 sentences focused on immediate relief"
        },
        "Low Mood": {
            "tone": "Gentle encouragement with deep compassionate understanding",
            "length": "3-4 sentences with hope and validation"
        },
        "Panic": {
            "tone": "Immediate stabilization with urgent but deeply calming presence",
            "length": "2-3 short, clear sentences for easy processing"
        }
    }
    
    selected_mode = mode_specifics.get(mode, mode_specifics["Calm"])
    intervention_guidance = intervention_prompts.get(
        emotional_context.intervention_type, 
        intervention_prompts["supportive_counseling"]
    )
    
    urgency_guidance = {
        "crisis": "🚨 CRISIS PRIORITY: Assess immediate safety. Provide crisis resources if needed. Focus entirely on stabilization and support.",
        "high": "HIGH URGENCY: This person needs extra support and validation. Be particularly gentle, thorough, and reassuring.",
        "moderate": "STANDARD SUPPORT: Provide appropriate therapeutic support with good depth and care."
    }
    
    return f"""
{base_context}

{session_guidance}

{intervention_guidance}

RESPONSE REQUIREMENTS:
- Tone: {selected_mode['tone']}
- Length: {selected_mode['length']} - BE CONCISE AND FOCUSED
- Urgency: {urgency_guidance.get(emotional_context.urgency_level, urgency_guidance["moderate"])}
- VARIETY: Use diverse opening phrases - avoid starting every response with "I hear". Instead use:
  * "It sounds like..." / "It seems..." / "I can sense..."
  * "What you're describing..." / "The way you're feeling..."
  * "When you mention..." / "Given what you've shared..."
  * "I understand..." / "I can see..." / "I recognize..."
  * Or start with a technique: "Let's try..." / "Here's something that might help..."

CONTEXT FROM THERAPEUTIC KNOWLEDGE BASE:
{{context}}

USER'S MESSAGE: {{question}}

Respond as Hush would - with professional warmth, evidence-based insight, session continuity, and the specific therapeutic approach needed. VARY YOUR OPENING PHRASES:
"""

# ---- Enhanced Response Generator ----
def get_response(user_input: str, mode: str, session_id: str = None) -> str:
    """Generate contextual response using all optimizations"""
    
    if not retriever:
        return "I'm sorry, but I'm having trouble accessing my knowledge base right now. Please make sure the vector database is set up properly by running ingest.py."
    
    try:
        # Step 1: Analyze emotional context with session awareness
        if not session_id:
            session_id = get_session_id()
            
        emotional_context, search_terms = analyze_emotional_context(user_input, session_id)
        print(f"Detected: {emotional_context.core_emotion} (urgency: {emotional_context.urgency_level})")
        
        session_context = get_session_context(session_id)
        if session_context and session_context.total_interactions >= 2:
            print(f"Session pattern: {session_context.session_trend} ({session_context.total_interactions} interactions)")
        
        if emotional_context.cognitive_patterns:
            print(f"Cognitive patterns: {', '.join(emotional_context.cognitive_patterns)}")
        
        # Step 2: Optimized retrieval
        context = optimized_retrieval(user_input, emotional_context, search_terms)
        
        # Step 3: Session-aware prompt generation
        prompt_template = get_session_aware_prompt(mode, emotional_context, session_id)
        
        # Format the prompt
        formatted_prompt = prompt_template.format(
            context=context,
            question=user_input
        )
        
        # Step 4: Generate response with optimized parameters
        temperature_map = {
            "Calm": 0.6,      # Slightly more creative for reflection
            "Anxious": 0.4,   # More consistent for grounding
            "Low Mood": 0.5,  # Balanced for encouragement
            "Panic": 0.3      # Most consistent for crisis support
        }
        
        # Adjust temperature based on urgency
        base_temp = temperature_map.get(mode, 0.5)
        if emotional_context.urgency_level == "crisis":
            base_temp = min(base_temp, 0.3)
        elif emotional_context.urgency_level == "high":
            base_temp = min(base_temp, 0.4)
        
        # Adjust max_tokens based on mode and urgency
        target_sentences = 2 if mode == "Panic" or emotional_context.urgency_level == "crisis" else 4
        max_tokens = 500 if target_sentences == 2 else 700
        
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are Hush, a skilled and empathetic clinical psychologist. Respond with warmth, professionalism, and evidence-based therapeutic guidance tailored to the person's specific emotional needs and session history. IMPORTANT: Vary your opening phrases - avoid starting every response with 'I hear'. Use diverse openings like 'It sounds like...', 'What you're describing...', 'Let's try...', etc."
                },
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ],
            temperature=base_temp,
            max_tokens=max_tokens,
            top_p=0.85,
            stream=False,
            stop=None
        )
        
        response = completion.choices[0].message.content
        
        # Step 5: Post-process for length control
        controlled_response = control_response_length(response, target_sentences)
        
        # Enhanced debug info
        print(f"Final response: {len(controlled_response.split('.'))} sentences, {len(controlled_response)} characters")
        
        return controlled_response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"I hear that you're struggling, and I want to help. I'm having some technical difficulties right now, but please know that what you're feeling is valid and you deserve support."

# ---- Debug Functions ----
def test_vector_db():
    """Test function to verify vector database is working"""
    if not retriever:
        print("❌ Vector database not loaded")
        return False
        
    try:
        test_queries = ["anxiety", "depression", "breathing", "mindfulness", "loneliness", "self-worth"]
        for query in test_queries:
            try:
                docs = retriever.invoke(query)
            except Exception as e:
                print(f"Retriever failed for '{query}', trying direct search: {e}")
                # Fallback to direct vectordb search with proper collection
                import chromadb
                client = chromadb.PersistentClient(path="vectordb")
                collections = client.list_collections()
                
                # Find a collection with documents
                for col in collections:
                    if col.count() > 0:
                        vectordb = Chroma(
                            persist_directory="vectordb", 
                            embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                            collection_name=col.name
                        )
                        docs = vectordb.similarity_search(query, k=8)
                        break
                else:
                    docs = []
            
            print(f"Query '{query}': Found {len(docs)} relevant documents")
            if docs:
                print(f"  First result: {docs[0].page_content[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Vector database test failed: {e}")
        return False

def test_session_memory():
    """Test session memory functionality"""
    print("\n=== SESSION MEMORY TESTING ===")
    
    test_session = "test_session_123"
    test_inputs = [
        "I feel so alone",
        "Nobody understands me", 
        "I'm always by myself",
        "What's the point of trying to connect"
    ]
    
    for i, test_input in enumerate(test_inputs):
        print(f"\nInput {i+1}: '{test_input}'")
        emotional_context, _ = analyze_emotional_context(test_input, test_session)
        session_context = get_session_context(test_session)
        
        print(f"  Detected emotion: {emotional_context.core_emotion}")
        print(f"  Session interactions: {session_context.total_interactions}")
        print(f"  Session trend: {session_context.session_trend}")
        print(f"  Dominant emotions: {session_context.dominant_emotions}")

def test_optimized_features():
    """Test all optimized features together"""
    print("\n=== OPTIMIZATION TESTING ===")
    
    test_session = "optimization_test"
    test_message = "I think nobody loves me and I'm always alone"
    
    print(f"Testing: '{test_message}'")
    
    # Test multiple responses to see session building
    for i in range(3):
        print(f"\n--- Response {i+1} ---")
        response = get_response(test_message, "Low Mood", test_session)
        print(f"Response: {response}")
        print(f"Length: {len(response.split('.'))} sentences")

if __name__ == "__main__":
    # Run all tests
    print("Testing vector database...")
    test_vector_db()
    
    print("\nTesting session memory...")
    test_session_memory()
    
    print("\nTesting optimized features...")
    test_optimized_features()
