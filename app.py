import os
import streamlit as st
from dotenv import load_dotenv

# langchain components
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Check for API Key properly
API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="TapNex AI Support", page_icon="🤖", layout="centered")

st.title("🤖 TapNex AI Customer Support Agent")
st.markdown("Ask any questions related to TapNex services, NFC payments, ticketing, and event infrastructure!")

@st.cache_resource
def init_rag_pipeline():
    # 1. Load Knowledge Base
    loader = TextLoader("knowledge_base.txt")
    docs = loader.load()

    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 3. Create Vector Store with Local Embeddings
    # Using HuggingFace embeddings (free and local)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store in FAISS
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return retriever

def main():
    if not API_KEY or API_KEY == "your_groq_api_key_here":
        st.warning("Please set your GROQ_API_KEY in the .env file to run this application.")
        return

    # Initialize RAG retriever
    with st.spinner("Initializing Knowledge Base..."):
        retriever = init_rag_pipeline()

    # Initialize LLM via Groq
    llm = ChatGroq(
        model="llama3-8b-8192", 
        temperature=0.1, 
        groq_api_key=API_KEY
    )

    # 4. Create Prompt Template strictly constrained to context
    system_prompt = (
        "You are an AI customer support assistant for TapNex, an event technology platform. "
        "Your task is to answer user queries accurately based ONLY on the provided context below. "
        "If the user asks a question that is unrelated to TapNex or not present in the context, "
        "you MUST NOT generate a general AI answer. Instead, you MUST reply EXACLTY with: "
        "\"I'm sorry, I couldn't find that information in the TapNex documentation.\"\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Build RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if query := st.chat_input("How can I help you with TapNex today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = rag_chain.invoke(query)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
