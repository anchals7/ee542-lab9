"""Website"": upload documents and chat with them. (Lab 9, Part 3, I) """
import html
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import embed
import ingest
from chat import get_conversation_chain
from htmlTemplates import bot_template, css, user_template

load_dotenv()  # loads OPENAI_API_KEY from .env


def make_chain(vectorstore):
    return get_conversation_chain(ChatOpenAI(model="gpt-5-mini"), vectorstore)


def show(template, text):
    """Put text into one of the chat-bubble templates from htmlTemplates.py."""
    text = html.escape(text).replace("\n", "<br>")
    st.write(template.replace("{{MSG}}", text), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Chat with your documents!")
    st.write(css, unsafe_allow_html=True)
    st.header("Ask me anything about your documents:")
    st.subheader("Upload your documents on the left and click 'process'")


    # load the saved index (if there is one) and start an empty chat.
    if "chain" not in st.session_state:
        st.session_state.chain = None
        st.session_state.messages = []
        if Path("faiss_index").exists():
            vectorstore = FAISS.load_local(
                "faiss_index",
                OpenAIEmbeddings(model="text-embedding-3-small"),
                allow_dangerous_deserialization=True,  # OK: we created this index ourselves
            )
            st.session_state.chain = make_chain(vectorstore)

    # Sidebar to upload one or more files, then add them to MySQL and the index.
    with st.sidebar:
        st.subheader("Your documents")
        files = st.file_uploader(
            "Upload PDF, TXT or HTML files, then click Process",
            type=["pdf", "txt", "html"],
            accept_multiple_files=True,
        )
        if st.button("Process") and files:
            with st.spinner("Reading, chunking and embedding..."):
                for f in files:
                    (ingest.DATA_DIR / f.name).write_bytes(f.getvalue())
                ingest.main()                 # read + chunk + save to MySQL
                vectorstore = embed.build_index()  # re-embed everything into FAISS
                st.session_state.chain = make_chain(vectorstore)
                st.session_state.messages = []
            st.success(f"Added {len(files)} file(s).")

    # Chat feature ask a question, get an answer with its sources.
    question = st.chat_input("Ask a question about your documents")
    if question:
        if st.session_state.chain is None:
            st.warning("Upload a document and click Process first.")
        else:
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke({"question": question})
            sources = ", ".join(f"{d.metadata['source']} p.{d.metadata['page']}"
                                for d in result["source_documents"])
            st.session_state.messages.append((question, result["answer"], sources))

    for question, answer, sources in st.session_state.messages:
        show(user_template, question)
        show(bot_template, f"{answer}\n\nSources: {sources}")


if __name__ == "__main__":
    main()