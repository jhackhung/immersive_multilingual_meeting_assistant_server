import os
from langchain_community.document_loaders import (
    DirectoryLoader, 
    PyPDFLoader, 
    TextLoader, 
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)
from apis.rag_service import RAGService

# --- Constants ---
DOCUMENTS_PATH = "documents"

def ingest_documents():
    """
    Loads documents from the specified directory, processes them, 
    and adds them to the RAG knowledge base.
    """
    if not os.path.exists(DOCUMENTS_PATH):
        print(f"Error: The '{DOCUMENTS_PATH}' directory does not exist.")
        print(f"Please create it in the project root and add your documents (e.g., .pdf, .txt, .md, .docx, .pptx).")
        return

    print(f"Loading documents from '{DOCUMENTS_PATH}'...")

    # Configure loaders for different file types
    loader = DirectoryLoader(
        DOCUMENTS_PATH,
        glob="**/*",
        use_multithreading=True,
        show_progress=True
    )

    try:
        documents = loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        print("Please ensure you have the necessary libraries installed for the file types you are using.")
        print("For example, for PDF files, run: pip install pypdf")
        print("For Word/PowerPoint files, run: pip install \"unstructured[docx,pptx]\"")
        return

    if not documents:
        print("No documents were found in the directory. Nothing to ingest.")
        return

    print(f"Loaded {len(documents)} document(s).")

    # Initialize the RAG service and add the loaded documents
    try:
        rag_service = RAGService()
        rag_service.add_documents(documents)
        print("\n--- Ingestion Complete ---")
        print(f"The documents from the '{DOCUMENTS_PATH}' directory have been added to the knowledge base.")
        print("You can now start the server and query the assistant.")
    except Exception as e:
        print(f"An error occurred during the ingestion process: {e}")
        print("Please check the console output for more details.")

if __name__ == "__main__":
    ingest_documents()