
import os
import chromadb
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import torch
import numpy as np
from pathlib import Path

# --- Constants ---
# Use a local model that is good for multilingual tasks
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Define where to save the ONNX model and the vector database
MODEL_CACHE_PATH = Path(__file__).parent.parent / "models" / "onnx_embedding_model"
CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "meeting_assistant_rag"


class ONNXEmbeddings:
    """
    Custom LangChain-compatible embedding class that runs a local ONNX model.
    This uses Optimum and ONNX Runtime with the DirectML execution provider for NPU/GPU acceleration.
    """
    def __init__(self, model_name: str, cache_path: Path):
        self.cache_path = cache_path
        self.model_name = model_name
        
        # Create cache directory if it doesn't exist
        self.cache_path.mkdir(parents=True, exist_ok=True)

        try:
            # Load the model and tokenizer from local cache if available
            print(f"Attempting to load ONNX model from local cache: {self.cache_path}")
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                self.cache_path, 
                provider="CPUExecutionProvider"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.cache_path)
            print("Successfully loaded ONNX model from cache.")
        except Exception as e:
            print(f"Could not load model from cache ({e}). Downloading and converting model...")
            # If not cached, download, convert to ONNX, and save
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                self.model_name, 
                export=True, # This flag handles the conversion to ONNX
                provider="CPUExecutionProvider"
            )
            # Save the converted model and tokenizer for future use
            self.model.save_pretrained(self.cache_path)
            self.tokenizer.save_pretrained(self.cache_path)
            print(f"Model converted to ONNX and saved to {self.cache_path}")

    def _mean_pooling(self, model_output, attention_mask):
        """Helper function for pooling token embeddings"""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of documents."""
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = sentence_embeddings.cpu().numpy()
        # Normalize embeddings
        sentence_embeddings = sentence_embeddings / np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        return sentence_embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query."""
        return self.embed_documents([text])[0]


class RAGService:
    """
    Service to manage the RAG pipeline, including document ingestion and querying.
    """
    def __init__(self):
        print("Initializing RAG Service...")
        self.embedding_function = ONNXEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, 
            cache_path=MODEL_CACHE_PATH
        )
        
        # Initialize the local vector store
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_function, # Note: ChromaDB wants the function object itself
            persist_directory=CHROMA_PATH,
        )
        print(f"RAG Service Initialized. Vector store loaded from: {CHROMA_PATH}")

    def add_documents(self, documents: list[Document]):
        """
        Splits documents, creates embeddings, and adds them to the vector store.
        """
        if not documents:
            print("No documents to add.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            print("Documents could not be split into chunks.")
            return
            
        print(f"Adding {len(chunks)} chunks to the vector store...")
        self.vector_store.add_documents(chunks)
        self.vector_store.persist()
        print(f"Successfully added {len(chunks)} chunks. Database persisted.")

    def query(self, query_text: str, n_results: int = 4) -> list[Document]:
        """
        Queries the vector store for relevant documents.
        """
        print(f"Querying for: '{query_text}'")
        results = self.vector_store.similarity_search(query_text, k=n_results)
        print(f"Found {len(results)} relevant documents.")
        return results

# Example of how to use the service (for testing purposes)
if __name__ == '__main__':
    rag_service = RAGService()

    # Example: Add some documents
    # test_docs = [
    #     Document(page_content="會議記錄 2024-08-20: 專案 Alpha 的預算超支了 15%。Peter 建議我們需要重新評估供應商。"),
    #     Document(page_content="技術規格書 v1.2: 使用者認證模組必須支援 OAuth 2.0。"),
    #     Document(page_content="會議記錄 2024-08-21: Mary 報告說使用者回饋表示 UI 不夠直觀。我們決定下個 sprint 進行改進。")
    # ]
    # rag_service.add_documents(test_docs)

    # Example: Query the database
    # query_result = rag_service.query("預算出了什麼問題？")
    # for doc in query_result:
    #     print("--- Relevant Chunk ---")
    #     print(doc.page_content)

