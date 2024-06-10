import streamlit as st
from langchain.schema import AIMessage, HumanMessage
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

load_dotenv()


def get_vectorstore_from_csv(file_path):
    loader = CSVLoader(file_path, encoding='utf-8')
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(document_chunks, embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    llm = ChatOpenAI(temperature=0)
    retriever = vector_store.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
    return qa


def get_response(user_input):
    conversation_chain = get_conversation_chain(st.session_state.vector_store)
    response = conversation_chain(
        {"question": user_input, "chat_history": st.session_state.chat_history})
    return response['answer']


# App config
st.set_page_config(page_title="Chat with CSV Data", page_icon="🤖")
st.title("Chat with CSV Data")

# CSV file path
csv_file_path = "web_content_embedded.csv"  # Replace with the actual path

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_csv(csv_file_path)

# User input
user_query = st.text_input("Type your message here...", key="input")
if user_query:
    response = get_response(user_query)
    st.session_state.chat_history.append(("Human", user_query))
    st.session_state.chat_history.append(("AI", response))

# Conversation
for message in st.session_state.chat_history:
    if message[0] == "AI":
        st.write(f"**AI:** {message[1]}")
    elif message[0] == "Human":
        st.write(f"**Human:** {message[1]}")
