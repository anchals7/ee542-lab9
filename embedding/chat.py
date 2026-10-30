"""Ask questions about the documents from the command line."""
from dotenv import load_dotenv
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()  # loads OPENAI_API_KEY from .env


def get_conversation_chain(llm, vectorstore):
    """connect the LLM, the vector store and a memory of the conversation."""
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),  # 4 closest chunks
        memory=memory,
        return_source_documents=True,  # so we can show which chunks were used
    )


def main():
    """driver: Ask a question, print the answer, repeat until 'exit'."""
    vectorstore = FAISS.load_local(
        "faiss_index",
        OpenAIEmbeddings(model="text-embedding-3-small"),
        allow_dangerous_deserialization=True,  # OK: we created this index ourselves
    )
    llm = ChatOpenAI(model="gpt-5-mini")
    chain = get_conversation_chain(llm, vectorstore)

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        result = chain.invoke({"question": question})
        print("\nAnswer:", result["answer"])
        sources = [f"{d.metadata['source']} p.{d.metadata['page']}"
                   for d in result["source_documents"]]
        print("Sources:", ", ".join(sources))


if __name__ == "__main__":
    main()
