import os
import time

import streamlit as st
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page configuration
st.set_page_config(
    page_title="AILAB Chatbot",
    page_icon="🤖",
    layout="wide"
)

# App title and description
st.title("AILAB Chatbot 🤖")
st.markdown("Get answers to your questions about your documents.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload your document.", type=["txt"])
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = ChatMessageHistory()
        st.experimental_rerun()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Check for OpenAI API key
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY is not set.")
    st.stop()

@st.cache_resource
def initialize_rag_pipeline(file_path:str = "data/data.txt") -> RunnableWithMessageHistory:
    """
    Initializes the Retrieval-Augmented Generation (RAG) pipeline.

    Args:
        file_path (str): Path to the text file containing the document data.

    Returns:
        RunnableWithMessageHistory: A LangChain-based conversational retrieval system.
    """

    try:
        # Load document
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        
        # Initialize LLM
        generative_model = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, 
            chunk_overlap=20,
            separators=["\n"]
        )   
        
        splits = text_splitter.split_documents(docs)
        
        # Create vector database
        vectorstore = FAISS.from_documents(
            documents=splits, 
            embedding=OpenAIEmbeddings(model="text-embedding-3-large")
        )
        
        retriever = vectorstore.as_retriever()
        
        # Setup reranker
        cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-2-v2")
        compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        
        # Context-aware retriever setup
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(
            generative_model, compression_retriever, contextualize_q_prompt
        )
        
        # QA chain setup
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use ONLY the provided retrieved context to answer the question. \
        If the context does not contain relevant information, simply respond with: \
        "I don't know based on the given information." \
        
        Retrieved context: 
        {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(generative_model, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Chat session management function
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            return st.session_state.chat_history
        
        # Create stateful chain
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        return conversational_rag_chain
    
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {str(e)}")
        return None

# Handle file upload
file_path = "data/data.txt"  # Default path
if uploaded_file:
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    # Save uploaded file
    temp_path = os.path.join("temp", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_path = temp_path
    st.sidebar.success(f"File uploaded: {uploaded_file.name}")

# Initialize RAG pipeline
rag_chain = initialize_rag_pipeline(file_path=file_path)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Process with RAG chain
            session_id = "streamlit_session"
            response = rag_chain.invoke(
                {"input": prompt},
                config={
                    "configurable": {"session_id": session_id}
                }
            )
            
            answer = response["answer"]
            
            # Update chat history in LangChain format
            st.session_state.chat_history.add_user_message(prompt)
            st.session_state.chat_history.add_ai_message(answer)
            
            # Display answer with typing effect
            full_response = ""
            for chunk in answer.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)
            
            message_placeholder.markdown(answer)
            
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("RAG Chatbot powered by LangChain and OpenAI")