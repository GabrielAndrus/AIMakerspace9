import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Session 11: Advanced Retrieval with LangChain

    ## Learning Objectives:

    - Understand and implement multiple retrieval strategies for RAG
    - Compare naive, BM25, multi-query, parent-document, contextual compression, ensemble, and semantic chunking approaches
    - Build RAG chains over a health and wellness knowledge base using LangChain and QDrant

    In the following notebook, we'll explore various methods of advanced retrieval using LangChain!

    We'll touch on:

    - Naive Retrieval
    - Best-Matching 25 (BM25)
    - Multi-Query Retrieval
    - Parent-Document Retrieval
    - Contextual Compression (a.k.a. Rerank)
    - Ensemble Retrieval
    - Semantic chunking

    We'll also discuss how these methods impact performance on our set of documents with a simple RAG chain.

    There will be two breakout rooms:

    - 🤝 Breakout Room Part #1
      - Task 1: Getting Dependencies!
      - Task 2: Data Collection and Preparation
      - Task 3: Setting Up QDrant!
      - Task 4-10: Retrieval Strategies
    - 🤝 Breakout Room Part #2
      - Activity: Evaluate with Ragas
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 🤝 Breakout Room Part #1
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 1: Getting Dependencies!

    We're going to need LangChain packages with a self-hosted LLM and embedding model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > NOTE: Create a `.env` file in this directory with `LLM_BASE_URL`, `LLM_MODEL`, and `EMBEDDING_MODEL` if needed. Defaults are configured for a local self-hosted instance at 192.168.1.79:8080.
    """)
    return


@app.cell
def _():
    import os
    from dotenv import load_dotenv

    load_dotenv()

    os.environ.setdefault("OPENAI_API_KEY", "dummy-key")
    os.environ.setdefault("LLM_BASE_URL", "http://192.168.1.79:8080/v1")
    os.environ.setdefault("LLM_MODEL", "minimax-m2.5-mlx@4bit")
    os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-qwen3-embedding-4b")
    os.environ.setdefault("EMBEDDING_BASE_URL", "http://192.168.1.79:8080/v1")
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 2: Data Collection and Preparation

    We'll be using our Health and Wellness Guide - a comprehensive resource covering exercise, nutrition, sleep, stress management, habits, and common health concerns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Preparation

    We'll load the wellness guide as a single document, then split it into smaller chunks using a `RecursiveCharacterTextSplitter` for our vector store. We also keep the raw (unsplit) document for use with the Parent Document Retriever and Semantic Chunker later.
    """)
    return


@app.cell
def _():
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = TextLoader("data/HealthWellnessGuide.txt")
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    wellness_docs = text_splitter.split_documents(raw_docs)
    return RecursiveCharacterTextSplitter, raw_docs, wellness_docs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's verify our data was loaded and split correctly!
    """)
    return


@app.cell
def _(raw_docs, wellness_docs):
    print(f"Raw documents: {len(raw_docs)}")
    print(f"Split chunks: {len(wellness_docs)}")
    print(f"\nExample chunk:\n{wellness_docs[0]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 3: Setting up QDrant!

    Now that we have our documents, let's create a QDrant VectorStore with the collection name "wellness_guide".

    We'll use the configured embedding model from environment variables.

    > NOTE: We'll be creating additional vectorstores where necessary, but this pattern is still extremely useful.
    """)
    return


@app.cell
def _(os, wellness_docs):
    from langchain_qdrant import QdrantVectorStore
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model=os.environ.get(
            "EMBEDDING_MODEL", "text-embedding-qwen3-embedding-4b"
        ),
        base_url=os.environ.get(
            "EMBEDDING_BASE_URL", "http://192.168.1.79:8080/v1"
        ),
        check_embedding_ctx_length=False,
    )

    vectorstore = QdrantVectorStore.from_documents(
        wellness_docs,
        embeddings,
        location=":memory:",
        collection_name="wellness_guide",
    )
    return OpenAIEmbeddings, QdrantVectorStore, embeddings, vectorstore


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 4: Naive RAG Chain

    Since we're focusing on the "R" in RAG today - we'll create our Retriever first.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### R - Retrieval

    This naive retriever will simply look at each review as a document, and use cosine-similarity to fetch the 10 most relevant documents.

    > NOTE: We're choosing `10` as our `k` here to provide enough documents for our reranking process later
    """)
    return


@app.cell
def _(vectorstore):
    naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    return (naive_retriever,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A - Augmented

    We're going to go with a standard prompt for our simple RAG chain today! Nothing fancy here, we want this to mostly be about the Retrieval process.
    """)
    return


@app.cell
def _():
    from langchain_core.prompts import ChatPromptTemplate

    RAG_TEMPLATE = """\
    You are a helpful and kind assistant. Use the context provided below to answer the question.

    If you do not know the answer, or are unsure, say you don't know.

    Query:
    {question}

    Context:
    {context}
    """

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    return (rag_prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### G - Generation

    We're going to leverage the configured LLM from environment variables.
    """)
    return


@app.cell
def _(os):
    from langchain_openai import ChatOpenAI

    chat_model = ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "minimax-m2.5-mlx@4bit"),
        base_url=os.environ.get("LLM_BASE_URL", "http://192.168.1.79:8080/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"),
    )
    return ChatOpenAI, chat_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### LCEL RAG Chain

    We're going to use LCEL to construct our chain.

    > NOTE: This chain will be exactly the same across the various examples with the exception of our Retriever!
    """)
    return


@app.cell
def _(chat_model, naive_retriever, rag_prompt):
    from langchain_core.runnables import RunnablePassthrough
    from operator import itemgetter
    from langchain_core.output_parsers import StrOutputParser

    naive_retrieval_chain = (
        # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
        # "question" : populated by getting the value of the "question" key
        # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
        {
            "context": itemgetter("question") | naive_retriever,
            "question": itemgetter("question"),
        }
        # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
        #              by getting the value of the "context" key from the previous step
        | RunnablePassthrough.assign(context=itemgetter("context"))
        # "response" : the "context" and "question" values are used to format our prompt object and then piped
        #              into the LLM and stored in a key called "response"
        # "context"  : populated by getting the value of the "context" key from the previous step
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )
    return RunnablePassthrough, itemgetter, naive_retrieval_chain


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see how this simple chain does on a few different prompts.

    > NOTE: You might think that we've cherry picked prompts that showcase the individual skill of each of the retrieval strategies - you'd be correct!
    """)
    return


@app.cell
def _(naive_retrieval_chain):
    naive_retrieval_chain.invoke(
        {"question": "What exercises can help with lower back pain?"}
    )["response"].content
    return


@app.cell
def _(naive_retrieval_chain):
    naive_retrieval_chain.invoke(
        {"question": "How does sleep affect overall health?"}
    )["response"].content
    return


@app.cell
def _(naive_retrieval_chain):
    naive_retrieval_chain.invoke(
        {"question": "What are some natural remedies for stress and headaches?"}
    )["response"].content
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Overall, this is not bad! Let's see if we can make it better!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 5: Best-Matching 25 (BM25) Retriever

    Taking a step back in time - [BM25](https://www.nowpublishers.com/article/Details/INR-019) is based on [Bag-Of-Words](https://en.wikipedia.org/wiki/Bag-of-words_model) which is a sparse representation of text.

    In essence, it's a way to compare how similar two pieces of text are based on the words they both contain.

    This retriever is very straightforward to set-up! Let's see it happen down below!
    """)
    return


@app.cell
def _(wellness_docs):
    from langchain_community.retrievers import BM25Retriever

    bm25_retriever = BM25Retriever.from_documents(wellness_docs)
    return (bm25_retriever,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll construct the same chain - only changing the retriever.
    """)
    return


@app.cell
def _(RunnablePassthrough, bm25_retriever, chat_model, itemgetter, rag_prompt):
    bm25_retrieval_chain = (
        {
            "context": itemgetter("question") | bm25_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )
    return (bm25_retrieval_chain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's look at the responses!
    """)
    return


@app.cell
def _(bm25_retrieval_chain):
    bm25_retrieval_chain.invoke(
        {"question": "What exercises can help with lower back pain?"}
    )["response"].content
    return


@app.cell
def _(bm25_retrieval_chain):
    bm25_retrieval_chain.invoke(
        {"question": "How does sleep affect overall health?"}
    )["response"].content
    return


@app.cell
def _(bm25_retrieval_chain):
    bm25_retrieval_chain.invoke(
        {"question": "What are some natural remedies for stress and headaches?"}
    )["response"].content
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It's not clear that this is better or worse, if only we had a way to test this (SPOILERS: We do, the second half of the notebook will cover this)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ❓ Question #1:

    Give an example query where BM25 is better than embeddings and justify your answer.

    ##### Answer:

    A query like "ICD-10 code for type 2 diabetes" would work better with BM25 because it requires exact keyword matching for specific codes and terminology. BM25 excels when you need precise term matching rather than semantic understanding, since it directly rewards documents containing the exact words in your query.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 6: Contextual Compression (Using Reranking)

    Contextual Compression is a fairly straightforward idea: We want to "compress" our retrieved context into just the most useful bits.

    There are a few ways we can achieve this - but we're going to look at a specific example called reranking.

    The basic idea here is this:

    - We retrieve lots of documents that are very likely related to our query vector
    - We "compress" those documents into a smaller set of *more* related documents using a reranking algorithm.

    We'll be leveraging Sentence Transformers CrossEncoder for our reranker!

    All we need to do is the following:

    - Create a basic retriever
    - Create a compressor (reranker, in this case)

    That's it!

    Let's see it in the code below!
    """)
    return


@app.cell
def _(naive_retriever):
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=naive_retriever
    )
    return (compression_retriever,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's create our chain again, and see how this does!
    """)
    return


@app.cell
def _(
    RunnablePassthrough,
    chat_model,
    compression_retriever,
    itemgetter,
    rag_prompt,
):
    contextual_compression_retrieval_chain = (
        {
            "context": itemgetter("question") | compression_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )
    return (contextual_compression_retrieval_chain,)


@app.cell
def _(contextual_compression_retrieval_chain):
    contextual_compression_retrieval_chain.invoke(
        {"question": "What exercises can help with lower back pain?"}
    )["response"].content
    return


@app.cell
def _(contextual_compression_retrieval_chain):
    contextual_compression_retrieval_chain.invoke(
        {"question": "How does sleep affect overall health?"}
    )["response"].content
    return


@app.cell
def _(contextual_compression_retrieval_chain):
    contextual_compression_retrieval_chain.invoke(
        {"question": "What are some natural remedies for stress and headaches?"}
    )["response"].content
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll need to rely on something like Ragas to help us get a better sense of how this is performing overall - but it "feels" better!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 7: Multi-Query Retriever

    Typically in RAG we have a single query - the one provided by the user.

    What if we had....more than one query!

    In essence, a Multi-Query Retriever works by:

    1. Taking the original user query and creating `n` number of new user queries using an LLM.
    2. Retrieving documents for each query.
    3. Using all unique retrieved documents as context

    So, how is it to set-up? Not bad! Let's see it down below!
    """)
    return


@app.cell
def _(chat_model, naive_retriever):
    from langchain.retrievers.multi_query import MultiQueryRetriever

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=naive_retriever, llm=chat_model
    )
    return (multi_query_retriever,)


@app.cell
def _(
    RunnablePassthrough,
    chat_model,
    itemgetter,
    multi_query_retriever,
    rag_prompt,
):
    multi_query_retrieval_chain = (
        {
            "context": itemgetter("question") | multi_query_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )
    return (multi_query_retrieval_chain,)


@app.cell
def _(multi_query_retrieval_chain):
    multi_query_retrieval_chain.invoke(
        {"question": "What exercises can help with lower back pain?"}
    )["response"].content
    return


@app.cell
def _(multi_query_retrieval_chain):
    multi_query_retrieval_chain.invoke(
        {"question": "How does sleep affect overall health?"}
    )["response"].content
    return


@app.cell
def _(multi_query_retrieval_chain):
    multi_query_retrieval_chain.invoke(
        {"question": "What are some natural remedies for stress and headaches?"}
    )["response"].content
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ❓ Question #2:

    Explain how generating multiple reformulations of a user query can improve recall.

    ##### Answer:

    By generating different versions of the user's question, you catch documents that use different words or phrasing to describe the same thing. Some reformulations will match documents that the original query would have missed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 8: Parent Document Retriever

    A "small-to-big" strategy - the Parent Document Retriever works based on a simple strategy:

    1. We split the full document into large "parent" chunks (e.g. 2000 characters).
    2. Each parent chunk is further split into smaller "child" chunks (e.g. 400 characters).
    3. The child chunks are stored in a VectorStore, while the parent chunks are stored in an in-memory docstore.
    4. When we query our Retriever, we do a similarity search comparing our query vector to the child chunks.
    5. Instead of returning the child chunks, we return their associated parent chunks.

    The basic idea is:

    - **Search** for small, focused chunks (better semantic matching)
    - **Return** big chunks (richer surrounding context)

    The intuition is that we're likely to find the most relevant information by limiting the amount of semantic information encoded in each embedding vector - but we're likely to miss relevant surrounding context if we only use that information.

    Let's start by defining our parent and child splitters.
    """)
    return


@app.cell
def _(RecursiveCharacterTextSplitter):
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import InMemoryStore
    from qdrant_client import QdrantClient, models

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50
    )
    return (
        InMemoryStore,
        ParentDocumentRetriever,
        QdrantClient,
        child_splitter,
        models,
        parent_splitter,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll need to set up a new QDrant vectorstore - and we'll use another useful pattern to do so!

    > NOTE: We are manually defining our embedding dimension, you'll need to change this if you're using a different embedding model.
    """)
    return


@app.cell
def _(OpenAIEmbeddings, QdrantClient, QdrantVectorStore, models, os):
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="wellness_parent_child",
        vectors_config=models.VectorParams(
            size=2560, distance=models.Distance.COSINE
        ),
    )
    parent_document_vectorstore = QdrantVectorStore(
        collection_name="wellness_parent_child",
        embedding=OpenAIEmbeddings(
            model=os.environ.get(
                "EMBEDDING_MODEL", "text-embedding-qwen3-embedding-4b"
            ),
            base_url=os.environ.get(
                "EMBEDDING_BASE_URL", "http://192.168.1.79:8080/v1"
            ),
            check_embedding_ctx_length=False,
        ),
        client=client,
    )
    return (parent_document_vectorstore,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can create our `InMemoryStore` that will hold our "parent documents" - and build our retriever!
    """)
    return


@app.cell
def _(
    InMemoryStore,
    ParentDocumentRetriever,
    child_splitter,
    parent_document_vectorstore,
    parent_splitter,
):
    store = InMemoryStore()

    parent_document_retriever = ParentDocumentRetriever(
        vectorstore=parent_document_vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    return (parent_document_retriever,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By default, this is empty as we haven't added any documents - let's add some now!
    """)
    return


@app.cell
def _(parent_document_retriever, raw_docs):
    parent_document_retriever.add_documents(raw_docs, ids=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll create the same chain we did before - but substitute our new `parent_document_retriever`.
    """)
    return


@app.cell
def _(
    RunnablePassthrough,
    chat_model,
    itemgetter,
    parent_document_retriever,
    rag_prompt,
):
    parent_document_retrieval_chain = (
        {
            "context": itemgetter("question") | parent_document_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )
    return (parent_document_retrieval_chain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's give it a whirl!
    """)
    return


@app.cell
def _(parent_document_retrieval_chain):
    parent_document_retrieval_chain.invoke(
        {"question": "What exercises can help with lower back pain?"}
    )["response"].content
    return


@app.cell
def _(parent_document_retrieval_chain):
    parent_document_retrieval_chain.invoke(
        {"question": "How does sleep affect overall health?"}
    )["response"].content
    return


@app.cell
def _(parent_document_retrieval_chain):
    parent_document_retrieval_chain.invoke(
        {"question": "What are some natural remedies for stress and headaches?"}
    )["response"].content
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Overall, the performance *seems* largely the same. We can leverage a tool like [Ragas]() to more effectively answer the question about the performance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 9: Ensemble Retriever

    In brief, an Ensemble Retriever simply takes 2, or more, retrievers and combines their retrieved documents based on a rank-fusion algorithm.

    In this case - we're using the [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) algorithm.

    Setting it up is as easy as providing a list of our desired retrievers - and the weights for each retriever.
    """)
    return


@app.cell
def _(
    bm25_retriever,
    compression_retriever,
    multi_query_retriever,
    naive_retriever,
    parent_document_retriever,
):
    from langchain.retrievers import EnsembleRetriever

    retriever_list = [
        bm25_retriever,
        naive_retriever,
        parent_document_retriever,
        compression_retriever,
        multi_query_retriever,
    ]
    equal_weighting = [1 / len(retriever_list)] * len(retriever_list)

    ensemble_retriever = EnsembleRetriever(
        retrievers=retriever_list, weights=equal_weighting
    )
    return (ensemble_retriever,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll pack *all* of these retrievers together in an ensemble.
    """)
    return


@app.cell
def _(
    RunnablePassthrough,
    chat_model,
    ensemble_retriever,
    itemgetter,
    rag_prompt,
):
    ensemble_retrieval_chain = (
        {
            "context": itemgetter("question") | ensemble_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )
    return (ensemble_retrieval_chain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's look at our results!
    """)
    return


@app.cell
def _(ensemble_retrieval_chain):
    ensemble_retrieval_chain.invoke(
        {"question": "What exercises can help with lower back pain?"}
    )["response"].content
    return


@app.cell
def _(ensemble_retrieval_chain):
    ensemble_retrieval_chain.invoke(
        {"question": "How does sleep affect overall health?"}
    )["response"].content
    return


@app.cell
def _(ensemble_retrieval_chain):
    ensemble_retrieval_chain.invoke(
        {"question": "What are some natural remedies for stress and headaches?"}
    )["response"].content
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 10: Semantic Chunking

    While this is not a retrieval method - it *is* an effective way of increasing retrieval performance on corpora that have clean semantic breaks in them.

    Essentially, Semantic Chunking is implemented by:

    1. Embedding all sentences in the corpus.
    2. Combining or splitting sequences of sentences based on their semantic similarity based on a number of [possible thresholding methods](https://python.langchain.com/docs/how_to/semantic-chunker/):
      - `percentile`
      - `standard_deviation`
      - `interquartile`
      - `gradient`
    3. Each sequence of related sentences is kept as a document!

    Let's see how to implement this!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll use the `percentile` thresholding method for this example which will:

    Calculate all distances between sentences, and then break apart sequences of setences that exceed a given percentile among all distances.
    """)
    return


@app.cell
def _(embeddings):
    from langchain_experimental.text_splitter import SemanticChunker

    semantic_chunker = SemanticChunker(
        embeddings, breakpoint_threshold_type="percentile"
    )
    return (semantic_chunker,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can split our documents.
    """)
    return


@app.cell
def _(raw_docs, semantic_chunker):
    semantic_documents = semantic_chunker.split_documents(raw_docs)
    return (semantic_documents,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's create a new vector store.
    """)
    return


@app.cell
def _(QdrantVectorStore, embeddings, semantic_documents):
    semantic_vectorstore = QdrantVectorStore.from_documents(
        semantic_documents,
        embeddings,
        location=":memory:",
        collection_name="wellness_guide_semantic_chunks",
    )
    return (semantic_vectorstore,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll use naive retrieval for this example.
    """)
    return


@app.cell
def _(semantic_vectorstore):
    semantic_retriever = semantic_vectorstore.as_retriever(search_kwargs={"k": 10})
    return (semantic_retriever,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally we can create our classic chain!
    """)
    return


@app.cell
def _(
    RunnablePassthrough,
    chat_model,
    itemgetter,
    rag_prompt,
    semantic_retriever,
):
    semantic_retrieval_chain = (
        {
            "context": itemgetter("question") | semantic_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )
    return (semantic_retrieval_chain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And view the results!
    """)
    return


@app.cell
def _(semantic_retrieval_chain):
    semantic_retrieval_chain.invoke(
        {"question": "What exercises can help with lower back pain?"}
    )["response"].content
    return


@app.cell
def _(semantic_retrieval_chain):
    semantic_retrieval_chain.invoke(
        {"question": "How does sleep affect overall health?"}
    )["response"].content
    return


@app.cell
def _(semantic_retrieval_chain):
    semantic_retrieval_chain.invoke(
        {"question": "What are some natural remedies for stress and headaches?"}
    )["response"].content
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ❓ Question #3:

    If sentences are short and highly repetitive (e.g., FAQs), how might semantic chunking behave, and how would you adjust the algorithm?

    ##### Answer:

    Semantic chunking will likely merge most or all sentences together since short, repetitive phrases have very similar embeddings - there won't be meaningful semantic breaks to split on. You'd need to increase the breakpoint threshold or combine it with a hard character/token limit to force smaller chunks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 🤝 Breakout Room Part #2
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🏗️ Activity #1:

    Your task is to evaluate the various Retriever methods against each other.

    You are expected to:

    1. Create a "golden dataset"
     - Use Synthetic Data Generation (powered by Ragas, or otherwise) to create this dataset
    2. Evaluate each retriever with *retriever specific* Ragas metrics
     - Semantic Chunking is not considered a retriever method and will not be required for marks, but you may find it useful to do a "semantic chunking on" vs. "semantic chunking off" comparison between them
    3. Compile these in a list and write a small paragraph about which is best for this particular data and why.

    Your analysis should factor in:
      - Cost
      - Latency
      - Performance

    > NOTE: This is **NOT** required to be completed in class. Please spend time in your breakout rooms creating a plan before moving on to writing code.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### HINTS:

    - LangSmith provides detailed information about latency and cost.
    """)
    return


@app.cell
def _(raw_docs):
    ### YOUR CODE HERE

    # Step 1: Load documents for knowledge graph generation

    # Docs already loaded in cell 8

    print(f"Loaded {len(raw_docs)} documents")
    return


@app.cell
def _(ChatOpenAI, OpenAIEmbeddings):
    # Step 2: Set up generator LLM and embeddings for Ragas
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    generator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="minimax-m2.5-mlx@4bit",
            base_url="http://192.168.1.79:8080/v1",
        )
    )
    generator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-qwen3-embedding-4b",
            base_url="http://192.168.1.79:8080/v1",
            check_embedding_ctx_length=False,
        )
    )
    return (
        LangchainEmbeddingsWrapper,
        LangchainLLMWrapper,
        generator_embeddings,
        generator_llm,
    )


@app.cell
def _(raw_docs):
    # Step 3: Build Knowledge Graph from documents
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType

    kg = KnowledgeGraph()

    # Add document nodes
    for doc in raw_docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
        )

    print(f"Added {len(kg.nodes)} document nodes to knowledge graph")
    return (kg,)


@app.cell
def _(generator_embeddings, generator_llm, kg, raw_docs):
    # Step 4: Apply default transforms to build relationships
    from ragas.testset.transforms import default_transforms, apply_transforms

    default_transforms_list = default_transforms(
        documents=raw_docs, llm=generator_llm, embedding_model=generator_embeddings
    )
    apply_transforms(kg, default_transforms_list)

    print(
        f"Knowledge graph built: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships"
    )
    return


@app.cell
def _(generator_embeddings, generator_llm, kg):
    # Step 5: Create testset generator
    from ragas.testset import TestsetGenerator

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
    )

    print("Testset generator ready")
    return (generator,)


@app.cell
def _(generator, raw_docs):
    # Step 6: Generate testset using the knowledge graph
    print("Generating testset...")

    # Use abstracted method which handles prompting better with self-hosted LLMs
    testset = generator.generate_with_langchain_docs(raw_docs, testset_size=10)
    print(f"Generated {len(testset.samples)} test samples")
    return (testset,)


@app.cell
def _(testset):
    # Step 5: Run queries through retrievers to populate context and response
    # This cell will be slow - running each query through all retrievers

    from concurrent.futures import ThreadPoolExecutor
    import time


    # We'll need to define our chain function here since it's used in multiple places
    def run_retriever_query(retriever, question):
        """Run a single retriever query and return context + response"""
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]

        # For evaluation, we primarily care about retrieved contexts
        # Response generation adds significant time but isn't needed for retriever eval
        return {
            "question": question,
            "retrieved_contexts": contexts,
            "num_docs": len(docs),
        }


    # Get test questions from the generated testset
    test_questions = [sample.eval_sample.user_input for sample in testset.samples]

    print(f"Running {len(test_questions)} queries through each retriever...")
    return run_retriever_query, test_questions, time


@app.cell
def _(
    bm25_retriever,
    compression_retriever,
    ensemble_retriever,
    multi_query_retriever,
    naive_retriever,
    parent_document_retriever,
    run_retriever_query,
    semantic_retriever,
    test_questions,
    time,
):
    # Step 6: Evaluate each retriever

    retrievers = {
        "naive": naive_retriever,
        "bm25": bm25_retriever,
        "compression": compression_retriever,
        "multi_query": multi_query_retriever,
        "parent_document": parent_document_retriever,
        "ensemble": ensemble_retriever,
    }

    results_dict = {}

    for _name, _retriever in retrievers.items():
        print(f"Evaluating {_name} retriever...")
        start_time = time.time()

        all_results = []
        for _q in test_questions:
            _result = run_retriever_query(_retriever, _q)
            all_results.append(_result)

        elapsed = time.time() - start_time

        # Calculate metrics: precision based on retrieved context coverage
        total_contexts = sum(r["num_docs"] for r in all_results)

        results_dict[_name] = {
            "total_retrieved": total_contexts,
            "avg_latency": elapsed / len(test_questions),
        }

        print(
            f"  Retrieved {total_contexts} docs in {elapsed:.2f}s ({elapsed / len(test_questions):.2f}s/query)"
        )

    print("\n--- All retrievers evaluated ---")

    # Semantic chunking comparison
    print("Evaluating semantic retriever (chunking ON)...")
    start_time = time.time()

    # Run queries separately to avoid double execution
    semantic_results = []
    for _q in test_questions:
        _result = run_retriever_query(semantic_retriever, _q)
        semantic_results.append(_result["num_docs"])
    elapsed = time.time() - start_time

    results_dict["semantic_chunking"] = {
        "total_retrieved": sum(semantic_results),
        "avg_latency": elapsed / len(test_questions),
    }
    return results_dict, retrievers


@app.cell
def _(testset):
    # Step 6b: Extract ground truth from testset for Ragas evaluation
    # Get ground truth (reference answers) and reference contexts from the generated testset

    ground_truths = [sample.eval_sample.reference for sample in testset.samples]
    reference_contexts_list = [
        sample.eval_sample.reference_contexts for sample in testset.samples
    ]

    print(f"Extracted {len(ground_truths)} ground truth answers for evaluation")
    return (ground_truths,)


@app.cell
def _(
    ChatOpenAI,
    LangchainEmbeddingsWrapper,
    LangchainLLMWrapper,
    OpenAIEmbeddings,
    ground_truths,
    retrievers,
    semantic_retriever,
    test_questions,
):
    # Step 6c: Qualitative Assessment with Ragas (context_precision and context_recall)
    from ragas import evaluate, RunConfig
    from ragas.metrics.collections import ContextPrecision, ContextRecall
    from datasets import Dataset

    # Set up evaluator LLM and embeddings using self-hosted endpoint
    eval_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="minimax-m2.5-mlx@4bit",
            base_url="http://192.168.1.79:8080/v1",
        )
    )
    eval_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-qwen3-embedding-4b",
            base_url="http://192.168.1.79:8080/v1",
            check_embedding_ctx_length=False,
        )
    )

    # Add the semantic retriever to our evaluators
    all_retrievers = dict(retrievers)
    all_retrievers["semantic_chunking"] = semantic_retriever

    ragas_results = {}

    for _name, _retriever in all_retrievers.items():
        print(f"Running Ragas evaluation for {_name} retriever...")

        # Get retrieved contexts for each question
        eval_data = []
        for i, q in enumerate(test_questions):
            docs = _retriever.invoke(q)
            retrieved_contexts = [doc.page_content for doc in docs]

            eval_data.append(
                {
                    "question": q,
                    "retrieved_contexts": retrieved_contexts,
                    "ground_truth": ground_truths[i],
                }
            )

        # Create dataset for Ragas
        ds = Dataset.from_list(eval_data)

        # getting rid of timeout errors
        run_config = RunConfig(timeout=120000, log_tenacity=True)

        # Run evaluation with context_precision and context_recall (as initialized objects)
        result = evaluate(
            ds,
            metrics=[ContextPrecision(), ContextRecall()],
            llm=eval_llm,
            embeddings=eval_embeddings,
            run_config=run_config,
        )

        # Extract average scores - handle both old and new Ragas APIs
        try:
            scores_df = result.to_pandas()  # type: ignore[attr-defined]
        except AttributeError:
            scores_df = result.scores  # type: ignore[attr-defined]

        avg_precision = float(scores_df["context_precision"].mean())  # type: ignore[union-attr]
        avg_recall = float(scores_df["context_recall"].mean())  # type: ignore[union-attr]

        ragas_results[_name] = {
            "context_precision": avg_precision,
            "context_recall": avg_recall,
        }

        print(
            f"  Context Precision: {avg_precision:.3f}, Context Recall: {avg_recall:.3f}"
        )

    print("\n--- Ragas evaluation complete ---")
    return (ragas_results,)


@app.cell
def _(mo, ragas_results, results_dict):
    # Step 7: Compile Results and Write Analysis
    import pandas as pd

    # Combine basic metrics with Ragas quality metrics
    combined_results = {}
    for name, metrics in results_dict.items():
        combined_results[name] = {
            "total_retrieved": metrics["total_retrieved"],
            "avg_latency": metrics["avg_latency"],
        }

    # Add Ragas results
    for name, ragas_metrics in ragas_results.items():
        if name in combined_results:
            combined_results[name]["context_precision"] = ragas_metrics[
                "context_precision"
            ]
            combined_results[name]["context_recall"] = ragas_metrics[
                "context_recall"
            ]

    df = pd.DataFrame(combined_results).T
    df.columns = [
        "Total Retrieved",
        "Avg Latency (s)",
        "Context Precision",
        "Context Recall",
    ]

    print("\n" + "=" * 60)
    print("RETRIEVER EVALUATION RESULTS")
    print("=" * 60)
    print(df.to_string())

    most_docs = df["Total Retrieved"].idxmax()
    fastest = df["Avg Latency (s)"].idxmin()

    # Find best quality metrics
    best_precision = df["Context Precision"].idxmax()
    best_recall = df["Context Recall"].idxmax()

    print(f"\nMost Documents Retrieved: {most_docs}")
    print(f"Fastest: {fastest}")
    print(f"Best Context Precision: {best_precision}")
    print(f"Best Context Recall: {best_recall}")

    analysis = f"""
    ### Analysis Summary

    Based on the evaluation results:

    - **Most Documents Retrieved**: {most_docs} - This retriever retrieves the most context chunks
    - **Fastest**: {fastest} - This retriever has the lowest average latency
    - **Best Context Precision**: {best_precision} - Highest context precision score (relevant contexts ranked higher)
    - **Best Context Recall**: {best_recall} - Highest context recall score (retrieved more relevant content)

    For this Health and Wellness knowledge base, the ensemble approach typically performs best as it combines multiple retrieval strategies (BM25 for keyword matching, dense embeddings for semantic similarity, and reranking for precision).
    """

    mo.md(analysis)
    return


if __name__ == "__main__":
    app.run()
