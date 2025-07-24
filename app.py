import streamlit as st
import os
import tempfile
import io

# Streamlit Cloud compatible imports
try:
    # Try the newer package structure first
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_COMMUNITY_AVAILABLE = True
except ImportError:
    # Fallback to older structure
    try:
        from langchain.document_loaders import PyPDFLoader, TextLoader
        from langchain.vectorstores import FAISS
        LANGCHAIN_COMMUNITY_AVAILABLE = False
    except ImportError:
        st.error("""
        **Missing Required Packages**
        
        Please ensure your requirements.txt contains:
        ```
        streamlit
        langchain
        langchain-community
        langchain-openai
        openai
        faiss-cpu
        pypdf
        python-dotenv
        ```
        """)
        st.stop()

# Text splitter imports
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        st.error("Could not import text splitter. Please check your langchain installation.")
        st.stop()

# OpenAI imports
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except ImportError:
    try:
        from langchain.embeddings.openai import OpenAIEmbeddings
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        try:
            from langchain.embeddings import OpenAIEmbeddings
            from langchain.llms import OpenAI as ChatOpenAI
        except ImportError:
            st.error("Could not import OpenAI components. Please check your openai and langchain installation.")
            st.stop()

# Core langchain imports
try:
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
except ImportError:
    st.error("Could not import core langchain components.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 Document Q&A Chatbot")
st.markdown("Upload a document and ask questions about its content. The bot will only answer questions based on the uploaded document.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # OpenAI API Key input
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key to use the chatbot"
    )
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Enter your OpenAI API key above
    2. Upload a document (PDF or TXT)
    3. Wait for processing to complete
    4. Ask questions about the document
    """)
    
    # Debug info
    with st.expander("🔧 Debug Info"):
        st.write(f"LangChain Community Available: {LANGCHAIN_COMMUNITY_AVAILABLE}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

def load_document(file):
    """Load document based on file type"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(file.getbuffer())
            tmp_file_path = tmp_file.name
        
        # Load document based on file type
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        elif file.name.endswith('.txt'):
            loader = TextLoader(tmp_file_path, encoding='utf-8')
        else:
            st.error("Unsupported file type. Please upload a PDF or TXT file.")
            return None
        
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return documents
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None

def create_vectorstore(documents):
    """Create vector store from documents"""
    try:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Create vector store
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def create_qa_chain(vectorstore):
    """Create QA chain with custom prompt"""
    try:
        # Custom prompt template
        prompt_template = """You are a helpful assistant that answers questions based ONLY on the provided context from the uploaded document. 

IMPORTANT RULES:
1. Only answer questions if the information is present in the provided context
2. If the question cannot be answered from the context, respond with: "I cannot answer this question as it's not covered in the uploaded document. Please ask questions related to the document content."
3. Do not use any external knowledge or information not present in the context
4. Be specific and cite relevant parts of the document when possible
5. If you're unsure whether the context contains the answer, err on the side of caution and decline to answer

Context from the document:
{context}

Question: {question}

Answer based only on the provided context:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create LLM with explicit API key
        try:
            llm = ChatOpenAI(
                temperature=0,
                model="gpt-3.5-turbo",
                openai_api_key=openai_api_key
            )
        except Exception:
            try:
                llm = ChatOpenAI(
                    temperature=0,
                    model_name="gpt-3.5-turbo",
                    openai_api_key=openai_api_key
                )
            except Exception as e:
                st.error(f"Error creating ChatOpenAI: {str(e)}")
                return None
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

# File upload section
st.header("📄 Document Upload")
uploaded_file = st.file_uploader(
    "Choose a document",
    type=['pdf', 'txt'],
    help="Upload a PDF or TXT file to ask questions about"
)

if uploaded_file and openai_api_key:
    # Process the uploaded file
    with st.spinner("Processing document..."):
        # Load document
        documents = load_document(uploaded_file)
        
        if documents:
            # Create vector store
            vectorstore = create_vectorstore(documents)
            
            if vectorstore:
                # Create QA chain
                qa_chain = create_qa_chain(vectorstore)
                
                if qa_chain:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.qa_chain = qa_chain
                    st.success(f"✅ Document '{uploaded_file.name}' processed successfully! You can now ask questions.")
                    
                    # Display document info
                    st.info(f"📊 Document contains {len(documents)} pages/sections")

elif uploaded_file and not openai_api_key:
    st.warning(⚠️ Please enter your OpenAI API key in the sidebar to process the document.")

# Chat interface
if st.session_state.qa_chain:
    st.header("💬 Chat with your Document")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(f"```\n{source.page_content[:300]}...\n```")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain({"query": prompt})
                    response = result["result"]
                    source_documents = result.get("source_documents", [])
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": source_documents
                    })
                    
                    # Display sources
                    if source_documents:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(source_documents):
                                st.markdown(f"**Source {i+1}:**")
                                st.markdown(f"```\n{source.page_content[:300]}...\n```")
                
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_message
                    })
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

else:
    st.header("💬 Chat Interface")
    st.info("👆 Please upload a document first to start chatting!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8em;">
    Built with LangChain, Streamlit, and OpenAI | Upload a document and ask questions about its content
</div>
""", unsafe_allow_html=True)