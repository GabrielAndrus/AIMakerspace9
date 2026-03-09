import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import os
    from dotenv import load_dotenv

    load_dotenv()

    ENDPOINT_URL = "http://192.168.1.79:8080/v1"
    MODEL_B = "unsloth/glm-5"
    EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
    return MODEL_B, EMBEDDING_MODEL, ENDPOINT_URL


@app.cell
def _(ENDPOINT_URL):
    from langchain_openai import ChatOpenAI
    from langchain_fireworks import ChatFireworks

    llm_a = ChatFireworks(
        model="accounts/fireworks/models/glm-4.7-flash", #replace with my account, but do not save file
        openai_api_key="my-key"
    )
    
    llm_b = ChatOpenAI(
        model="unsloth/glm-5",
        openai_api_base=ENDPOINT_URL,
        openai_api_key="not-needed"
    )
    
    return llm_a, llm_b


@app.cell
def _():
    from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import tiktoken

    loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()

    def tiktoken_len(text: str) -> int:
        tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750, chunk_overlap=0, length_function=tiktoken_len
    )

    chunks = text_splitter.split_documents(documents)
    return (chunks,)


@app.cell
def _(ENDPOINT_URL, chunks):
    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain_qdrant import QdrantVectorStore

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-nomic-embed-text-v1.5",
        openai_api_base=ENDPOINT_URL,
        openai_api_key="not-needed",
        check_embedding_ctx_length=False
    )

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        location=":memory:",
        collection_name="cat_health_rag"
    )

    retriever = vectorstore.as_retriever()
    return (retriever,)


@app.cell
def _(llm_a, llm_b, retriever):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langgraph.graph import START, StateGraph
    from typing import TypedDict
    from langchain_core.documents import Document

    class RAGState(TypedDict):
        question: str
        context: list[Document]
        response: str

    human_template = (
        "\n#CONTEXT:\n{context}\n\nQUERY:\n{query}\n\n"
        "Use the provided context to answer the user query. "
        "Only use the provided context. If you don't know or it's not in context, say \"I don't know\""
    )
    chat_prompt = ChatPromptTemplate.from_messages([("human", human_template)])

    def build_rag_graph(llm):
        def retrieve(state: RAGState) -> dict:
            docs = retriever.invoke(state["question"])
            return {"context": docs}

        def generate(state: RAGState) -> dict:
            chain = chat_prompt | llm | StrOutputParser()
            response = chain.invoke({
                "query": state["question"],
                "context": state.get("context", [])
            })
            return {"response": response}

        builder = StateGraph(RAGState)
        builder.add_sequence([retrieve, generate])
        builder.add_edge(START, "retrieve")
        return builder.compile()

    rag_graph_a = build_rag_graph(llm_a)
    rag_graph_b = build_rag_graph(llm_b)
    
    return rag_graph_a, rag_graph_b


@app.cell
def _(rag_graph_a, rag_graph_b):
    import time

    test_queries = [
        "What are the key life stages for cats?",
        "How often should kittens be vaccinated?",
        "What is the recommended diet for senior cats?",
        "How can I prevent parasites in my cat?",
        "What behavioral changes indicate illness in cats?"
    ]

    def measure_response(graph, query: str):
        start = time.time()
        result = graph.invoke({"question": query})
        elapsed = time.time() - start
        return (result["response"], elapsed)

    results = []
    
    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("-" * 60)
        
        response_a, time_a = measure_response(rag_graph_a, query)
        print(f"\n[Fireworks API glm-4.7-flash] ({time_a:.2f}s):")
        print(response_a)
        
        response_b, time_b = measure_response(rag_graph_b, query)
        print(f"\n[Local Model glm-5] ({time_b:.2f}s):")
        print(response_b)
        
        results.append({
            "query": query,
            "time_a": time_a,
            "time_b": time_b,
            "response_a": response_a,
            "response_b": response_b
        })
    
    return (results,)


@app.cell
def _(results):
    import pandas as pd

    df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("LATENCY COMPARISON")
    print("=" * 60)
    print(f"Fireworks API avg latency: {df['time_a'].mean():.2f}s")
    print(f"Local Model (glm-5) avg latency: {df['time_b'].mean():.2f}s")
    print(f"\nLatency difference: {abs(df['time_a'].mean() - df['time_b'].mean()):.2f}s")
    
    if df["time_a"].mean() < df["time_b"].mean():
        print(f"Fireworks API is faster by {(df['time_b'].mean() / df['time_a'].mean() - 1) * 100:.1f}%")
    else:
        print(f"Local Model (glm-5) is faster by {(df['time_a'].mean() / df['time_b'].mean() - 1) * 100:.1f}%")
    
    return (pd,)


if __name__ == "__main__":
    app.run()
