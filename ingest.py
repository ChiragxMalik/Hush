import os
import re
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_FOLDER = "data"
DB_FOLDER = "vectordb"

# Load a single document (PDF or DOCX)
def load_document(file_path):
    if file_path.lower().endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    return loader.load()

# Split text into chunks
def split_documents(documents, chunk_size=1200, chunk_overlap=300):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Incorporated from remote
        add_start_index=True,
        separators=[
            "\n\n\n",  # Chapter/section breaks
            "\n\n",     # Paragraph breaks
            "\n",       # Line breaks
            ". ",       # Sentences
            "? ",
            "! ",
            "; ",       # For lists and techniques
            ", ",       # For shorter breaks
            " ",
            ""
        ]
    )
    return splitter.split_documents(documents)

# Embedding model
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sanitize collection name
def get_clean_collection_name(file_path):
    file_name = os.path.basename(file_path)
    file_name = file_name.replace(".pdf", "").replace(".docx", "")
    clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", file_name)
    clean_name = re.sub(r"^[^a-zA-Z0-9]+", "", clean_name)
    clean_name = re.sub(r"[^a-zA-Z0-9]+$", "", clean_name)
    return clean_name.lower()

# Store chunks as embeddings in ChromaDB
def store_embeddings(chunks, embedding_model, collection_name, persist_directory=DB_FOLDER):
    print(f"Creating/updating Chroma collection: '{collection_name}' in '{persist_directory}'")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    vector_db.persist()
    print(f"Successfully stored {len(chunks)} chunks for '{collection_name}'.")
    return vector_db

# Main pipeline
def process_all_documents(data_folder=DATA_FOLDER, db_folder=DB_FOLDER):
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder '{data_folder}' not found")
    
    os.makedirs(db_folder, exist_ok=True)
    embedding_model = get_embeddings_model()

    processed_files = []
    failed_files = []

    for filename in os.listdir(data_folder):
        if filename.lower().endswith(('.pdf', '.docx')):
            try:
                file_path = os.path.join(data_folder, filename)
                print(f"Processing: {file_path}")

                # --- Document loading ---
                documents = load_document(file_path)

                # --- Metadata handling ---
                for doc in documents:
                    doc.metadata['source_file'] = filename
                    # Ensure 'page' exists
                    if 'page' not in doc.metadata and 'index' in doc.metadata:
                        doc.metadata['page'] = doc.metadata['index']
                    elif 'page' not in doc.metadata:
                        doc.metadata['page'] = 'N/A'

                # --- Splitting + storing embeddings ---
                chunks = split_documents(documents)
                collection_name = get_clean_collection_name(file_path)
                store_embeddings(chunks, embedding_model, collection_name, db_folder)

                print(f"Stored vectors for: {filename}\n")
                processed_files.append(filename)

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                failed_files.append((filename, str(e)))
                continue

    # --- Summary report ---
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(processed_files)} files")
    if failed_files:
        print(f"Failed to process: {len(failed_files)} files")
        for file, error in failed_files:
            print(f"  - {file}: {error}")

            
# Run
if __name__ == "__main__":
    process_all_documents()
