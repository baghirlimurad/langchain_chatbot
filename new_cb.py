import os
import json
from typing import List, Dict, Any, Union, Optional

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA


class VectorSearch:
    """
    A class to handle vector search operations using document embeddings.
    Supports processing both JSON and text data formats.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the VectorSearch instance.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will attempt to load from environment.
        """
        # Setup API key
        if api_key:
            self.api_key = api_key
        else:
            load_dotenv()
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment variables")
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
        self.vector_store = None
        self.qa_chain = None

    def process_json_data(self, json_file_path: str, metadata_fields: List[str] = None) -> List[Document]:
        """
        Process JSON data file into Document objects.
        
        Args:
            json_file_path (str): Path to the JSON file
            metadata_fields (List[str], optional): Fields to extract as metadata. 
                                                  If None, will use all top-level fields.
        
        Returns:
            List[Document]: List of Document objects created from JSON data
        """
        # Load JSON data
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        documents = []
        
        # Process each item in the JSON data
        for item in data:
            # Convert item to string content
            content = json.dumps(item, indent=2)
            
            # Extract metadata fields
            if metadata_fields:
                metadata = {field: item.get(field) for field in metadata_fields if field in item}
            else:
                # Use all top-level fields as metadata
                metadata = {k: v for k, v in item.items() if not isinstance(v, (dict, list))}
            
            # Create Document object
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
            
        return documents

    def process_text_data(self, text_file_path: str, chunk_size: int = 100, 
                          chunk_overlap: int = 0) -> List[Document]:
        """
        Process text file into chunked Document objects.
        
        Args:
            text_file_path (str): Path to the text file
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
            
        Returns:
            List[Document]: List of chunked Document objects
        """
        # Load text data
        loader = TextLoader(text_file_path, encoding="utf-8")
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunked_docs = text_splitter.split_documents(documents)
        return chunked_docs

    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a vector store from documents.
        
        Args:
            documents (List[Document]): List of Document objects
        """
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
    def setup_qa_chain(self, chain_type: str = "stuff") -> None:
        """
        Set up a question-answering chain.
        
        Args:
            chain_type (str): Type of chain to use ("stuff", "map_reduce", etc.)
        
        Raises:
            ValueError: If vector_store is not initialized
        """
        if not self.vector_store:
            raise ValueError("Vector store must be initialized before setting up QA chain")
            
        retriever = self.vector_store.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(api_key=self.api_key), 
            chain_type=chain_type, 
            retriever=retriever
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            List[Document]: List of similar documents
            
        Raises:
            ValueError: If vector_store is not initialized
        """
        if not self.vector_store:
            raise ValueError("Vector store must be initialized before searching")
            
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            List[tuple]: List of (Document, score) tuples
            
        Raises:
            ValueError: If vector_store is not initialized
        """
        if not self.vector_store:
            raise ValueError("Vector store must be initialized before searching")
            
        return self.vector_store.similarity_search_with_score(query, k=k)

    def answer_question(self, question: str) -> Dict[str, str]:
        """
        Answer a question using the QA chain.
        
        Args:
            question (str): Question to answer
            
        Returns:
            Dict[str, str]: Result containing query and answer
            
        Raises:
            ValueError: If qa_chain is not initialized
        """
        if not self.qa_chain:
            raise ValueError("QA chain must be initialized before answering questions")
            
        return self.qa_chain.invoke(question)


# Example usage
def main():
    """Example usage of the VectorSearch class."""
    # Initialize
    vs = VectorSearch()
    
    # Process JSON data
    json_docs = vs.process_json_data(
        'data/data.json', 
        metadata_fields=['title', 'duration', 'max_participants']
    )
    
    # Create vector store
    vs.create_vector_store(json_docs)
    
    # Setup QA chain
    vs.setup_qa_chain()
    
    # Answer a question
    result = vs.answer_question("what is the duration of the compscience workshop?")
    print(result)
    
    # Alternative: Process text data
    # text_docs = vs.process_text_data('data/data.txt', chunk_size=100)
    # vs.create_vector_store(text_docs)
    # vs.setup_qa_chain()


if __name__ == "__main__":
    main()