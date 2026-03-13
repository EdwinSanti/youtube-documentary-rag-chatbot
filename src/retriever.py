from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def get_retriever():
    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever


if __name__ == "__main__":
    retriever = get_retriever()

    query = "What are elephants?"
    docs = retriever.invoke(query)

    print("\nTop results:\n")

    for doc in docs:
        print(doc.page_content)
        print("\n---\n")