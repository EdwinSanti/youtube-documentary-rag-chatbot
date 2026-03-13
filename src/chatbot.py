from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma


def ask_chatbot(question: str):
    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are an expert AI assistant that answers questions about documentary content.

Use ONLY the retrieved context below to answer the user's question.
Do not use outside knowledge.

If the answer is not contained in the context, say:
"I do not know based on the provided context."

If the exact answer is not stated in the context, provide the closest relevant information from the context and clearly say that the exact answer is not directly stated.

Provide a clear and concise explanation based on the context.
Do not quote large portions of the text directly.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, docs


if __name__ == "__main__":

    print("\nElephant Documentary Chatbot")
    print("Type 'exit' to quit\n")

    while True:
        question = input("Ask a question: ")

        if question.lower() == "exit":
            break

        answer, docs = ask_chatbot(question)

        print("\nAnswer:\n")
        print(answer)

        print("\nSources used:\n")
        for i, doc in enumerate(docs, start=1):
            print(f"Source {i}:")
            print(doc.page_content[:400])  # Print the first 400 characters of each source
            print("\n---\n")