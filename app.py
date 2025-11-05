import streamlit as st
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
import traceback

# ------------------ Page Configuration ------------------ #
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🏥",
    layout="centered"
)

# ------------------ Load API Keys from Streamlit Secrets ------------------ #
try:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error("⚠️ Please configure your API keys in Streamlit Cloud secrets!")
    st.stop()

# ------------------ Initialize Components (Cached) ------------------ #
@st.cache_resource
def initialize_chatbot():
    """Initialize embeddings, vector store, and LLM (runs once)"""
    try:
        # Embeddings & Pinecone
        embeddings = download_hugging_face_embeddings()
        index_name = "medical-chatbot"
        
        # Connect to Pinecone index
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        # Retriever
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # Gemini Model
        chatModel = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",  # Changed to stable version
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=0.2
        )
        
        # Prompt Template
        prompt = ChatPromptTemplate.from_template("""
You are a helpful AI medical assistant.
Use the following context to answer the user's question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{input}
""")
        
        # Combine Docs Function
        def combine_docs(docs):
            """Extract text content from retrieved documents."""
            try:
                text_parts = []
                for doc in docs:
                    if hasattr(doc, "page_content"):
                        text_parts.append(str(doc.page_content))
                    elif isinstance(doc, dict):
                        if "page_content" in doc:
                            text_parts.append(str(doc["page_content"]))
                        elif "text" in doc:
                            text_parts.append(str(doc["text"]))
                        elif "content" in doc:
                            text_parts.append(str(doc["content"]))
                        else:
                            text_parts.append(str(doc))
                    else:
                        text_parts.append(str(doc))
                
                result = "\n\n".join(text_parts)
                return result
            
            except Exception as e:
                st.error(f"❌ combine_docs error: {e}")
                return ""
        
        # Build RAG Chain
        rag_chain = (
            {
                "context": retriever | RunnableLambda(combine_docs),
                "input": RunnablePassthrough(),
            }
            | prompt
            | chatModel
        )
        
        return rag_chain
    
    except Exception as e:
        st.error(f"❌ Initialization Error: {e}")
        traceback.print_exc()
        return None

# ------------------ Streamlit UI ------------------ #
st.title("🏥 Medical Chatbot")
st.markdown("Ask me any medical question and I'll help you with information from medical resources.")

# Initialize chatbot
with st.spinner("🔄 Loading chatbot..."):
    rag_chain = initialize_chatbot()

if rag_chain is None:
    st.error("Failed to initialize chatbot. Please check your API keys and configuration.")
    st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your medical question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke(prompt)
                
                # Extract answer from response
                if hasattr(response, "content"):
                    answer = response.content
                elif isinstance(response, dict):
                    answer = (
                        response.get("answer")
                        or response.get("output")
                        or response.get("content")
                        or str(response)
                    )
                else:
                    answer = str(response)
                
                st.markdown(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                traceback.print_exc()

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This is a medical chatbot powered by:
    - 🤖 Google Gemini Pro
    - 📚 Pinecone Vector Database
    - 🔍 RAG (Retrieval Augmented Generation)
    
    **Disclaimer:** This chatbot provides information only. 
    Always consult with healthcare professionals for medical advice.
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
