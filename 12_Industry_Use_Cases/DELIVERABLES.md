Loom video:
https://www.loom.com/share/eec4e81436d94b019d1ce73a2f31a7ce

Section 1
1. The people of the Forge Utah community have need to train ML models, but don't know how and aren't data scientists.
2. There are many people in the Forge Utah community who are interested in the predictive power of classical ML. They've heard of gradient-boosted trees and regression, and likely they've even worked with or around people who have given them insights using ML models. Not everyone, however is technical, nor a data scientist. There are technical developers that are part of the community whose eyes glaze over immediately upon mentioning anything Bayesian, and product managers with oodles of ideas for how to leverage good predictive power, but with the assumption that someone else is just going to build it. Even those who ARE data scientists don't have the entireity of the scikit-learn catalogue memorized, nor do they know how to apply all of it, and even fewer still know about training LLMs. Forge Utah has recently come into some compute in the form of 2 DGX Sparks, but this adds another layer of complexity, where only someone who knows both the Spark _and_ data science would be able to effectively train anything.
3. {"question": "What model works best for imbalanced binary classification with 10000 rows?", "ground_truth": "For imbalanced binary classification with 10000 rows, use XGBoost with scale_pos_weight parameter set to the ratio of negative/positive samples. This achieves F1-score improvement of 15% over default settings. RandomForest with class_weight='balanced' is also a good option."}
{"question": "I have a dataset with 5000 rows and 200 features. Which model should I use?", "ground_truth": "For datasets with 5000 rows and 200 features (high-dimensional), use XGBoost with colsample_bytree and subsample to handle dimensionality. Logistic Regression with L1 regularization (sparse solution) also works well. Random Forest may overfit without careful tuning of max_features."}
{"question": "My dataset has many categorical features with high cardinality. What preprocessing and model combo works best?", "ground_truth": "For high cardinality categorical features, use CatBoost which handles categorical features natively without explicit encoding. It uses ordered target encoding to prevent target leakage and provides excellent performance out-of-the-box."}
{"question": "I need interpretability for regulatory compliance. Which models provide feature importance?", "ground_truth": "For interpretability and regulatory compliance, use tree-based models like DecisionTree, RandomForest, or XGBoost which provide built-in feature_importances_ attribute. Use SHAP values with any tree-based model to explain individual predictions."}
{"question": "What is the fastest model to train on large datasets with 100K+ samples?", "ground_truth": "For fastest training on large datasets (100K+ samples), use LightGBM which uses histogram-based algorithms. Training time: LightGBM ~1-3 seconds, XGBoost ~3-8 seconds, Random Forest ~5-10 seconds, Logistic Regression <1 second."}
{"question": "How do I handle missing values in my dataset?", "ground_truth": "Tree-based models (Random Forest, XGBoost, LightGBM) handle missing values natively. For other models, use SimpleImputer with mean, median, or most_frequent strategy. Consider using IterativeImputer for multivariate imputation."}
{"question": "What cross-validation strategy should I use for classification?", "ground_truth": "Use StratifiedKFold for classification to maintain class distribution in each fold. For small datasets, use RepeatedStratifiedKFold to get more reliable estimates. Default k=5 or k=10 is usually sufficient."}
{"question": "Do I need to scale features for tree-based models?", "ground_truth": "Tree-based models (Random Forest, XGBoost, LightGBM) generally do not require feature scaling as they are invariant to monotonic transformations. However, scaling is required for SVM, KNN, and linear models for optimal performance."}
{"question": "What model works well for text classification with small sample sizes?", "ground_truth": "For text classification with small sample sizes, Naive Bayes (MultinomialNB or BernoulliNB) works well as it is fast and performs adequately with high-dimensional sparse data. It is a good baseline for text classification."}
{"question": "How do I choose between XGBoost and LightGBM?", "ground_truth": "Use LightGBM for fastest training on large datasets and when categorical features are present (it handles them directly). Use XGBoost when you need more regularization options or when working with imbalanced datasets (scale_pos_weight parameter)."}


Section 2
Proposed Solution:
Agentic AutoML
This platform uses the scikit learn documentation itself to determine, based on an uploaded dataset (either tabular or text-based) what is the best model to train for predicting, and what method is best for completing that training. There are several piece to this that make it work: QDrant vector store, using Qwen3-embedding-4b to embed our knowledge base for retrieval, a locally-hosted vLLM server (default: Qwen/Qwen2.5-0.5B) for making the ultimate decision, and the platform of the Spark inside Nvidia's golden pytorch docker container to actually _do_ the training. Once the training is done, we don't have to simply trust that the LLM has done a good job, there is an inference playground for loading the model and checking against whatever golden dataset is required. All of this is, of course, accessible through both the webui as well as a straight API call to the running gradio server.
Infrastructure Diagram:
```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    User Interface                                       │
│  ┌──────────────────────────────────────┐   ┌─────────────────────────────────────────┐ │
│  │         Gradio Web UI                │   │       FastAPI REST API                  │ │
│  │  - Dataset upload                    │   │  - Programmatic access                  │ │
│  │  - Model selection playground        │   │  - Auto-generated docs                  │ │
│  │  - Inference playground              │   │                                         │ │
│  └──────────────────┬───────────────────┘   └────────────────────┬────────────────────┘ │
└─────────────────────┼──────────────────────────────────────────────┼────────────────────┘
                      │                                              │
                      ▼                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                              Agent Orchestration Framework                                │
│                           (LangGraph / Custom Agentic Loop)                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  1. Dataset Analysis Agent → Analyzes uploaded data structure & characteristics     │  │
│  │  2. RAG Retrieval Agent → Queries QDrant for relevant sklearn documentation         │  │
│  │  3. Model Selection Agent → Uses LLM to recommend best model based on retrieval     │  │
│  │  4. Training Agent → Executes training pipeline on DGX Spark                        │  │
│  4. Error Investigation Agent → Multi-agent system for debugging ML/LLM errors   │  │
│  └─────────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────────┬─────────────────────┘
                      │                                               │
          ┌───────────▼───────────┐                   ┌───────────────▼────────────────┐
          │   LLM Decision Maker  │                   │    Monitoring & Evaluation     │
          │                       │                   │                                │
          │  vLLM-hosted LLM      │                   │    Langfuse                    │
          │  (default: Qwen2.5)   │                   │    - Trace model decisions     │
          │                       │                   │    - Log evaluation metrics    │
          │  Chosen for: local    │                   │    - Audit reasoning steps     │
          │  hosting, no API deps │                   │                                │
          │                       │                   │    RAGAS                       │
          └───────────────────────┘                   │    - Faithfulness metrics      │
                                                      │    - Context precision/recall  │
                                                      └────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    Tool Layer                                           │
│  ┌──────────────────────┐   ┌─────────────────────────────────────────────────────────┐ │
│  │   Embedding Model    │   │              Vector Database                            │ │
│  │                      │   │                                                         │ │
│  │ Qwen/Qwen3-Embedding │   │                     QDrant                              │ │
│  │         -4B          │   │              (Local self-hosted)                        │ │
│  │                      │   │                                                         │ │
│  │  Chosen for: open    │   │     Chosen for: tested in class, easy to                │ │
│  │  source, 2560 dims   │   │     bundle with docker compose, local hosting           │ │
│  └──────────────────────┘   └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                     Data Layer                                            │
│  ┌─────────────────────────────────────┐   ┌───────────────────────────────────────────┐  │
│  │      Knowledge Base                 │   │         Compute Infrastructure            │  │
│  │                                     │   │                                           │  │
│  │  - Embedded sklearn documentation   │   │     NVIDIA DGX Spark (2x)                 │  │
│  │  - Model selection guides           │   │     - PyTorch Docker container            │  │
│  │  - Preprocessing tutorials          │   │     - GPU-accelerated training            │  │
│  │                                     │   │                                           │  │
│  │  Chosen for: domain-specific RAG    │   │     Chosen for: available compute,        │  │
│  │  context that enables accurate      │   │     CUDA support for training             │  │
│  │  model recommendations              │   │                                           │  │
│  └─────────────────────────────────────┘   └───────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

Chosen tooling with rationale:
- **LLM (vLLM-hosted Qwen2.5)**: Chosen for local hosting without external API dependencies; provides fast inference with OpenAI-compatible API for seamless integration.
- **Agent Orchestration (Custom/LangGraph)**: Chosen to orchestrate the multi-step pipeline from dataset analysis through RAG retrieval to model selection and training execution.
- **Tools (Python/sklearn)**: Chosen as the foundation since scikit-learn is the core library being wrapped, and Python provides seamless integration across all components.
- **Embedding Model (Qwen/Qwen3-Embedding-4B)**: Chosen for its open-source availability and 2560-dimensional embeddings that capture rich semantic relationships in sklearn documentation.
- **Vector Database (QDrant)**: Chosen because it was tested in class, supports local self-hosting, and integrates easily via docker compose for the entire stack.
- **Monitoring (Langfuse)**: Chosen for data governance and traceability, allowing evaluation of model decisions and auditing the agent's reasoning at each step.
- **Evaluation Framework (RAGAS + Langfuse)**: Chosen to combine automated RAG quality metrics (faithfulness, context precision/recall) with Langfuse traces for comprehensive evaluation.
- **User Interface (Gradio)**: Chosen because of prior experience and its built-in FastAPI integration, providing both web UI and documented REST API simultaneously.
- **Deployment Tool (Docker)**: Chosen to ensure consistent environments for users and maintainers, with the NVIDIA PyTorch container providing GPU support on DGX Sparks.

RAG and Agent Components:
The rag component is QDrant with Qwen3-embedding-4b
The agent component is 2-fold, with the first part being the agentic data collection/engineering where I scrape SKLearn's documentation and output the .jsonl files in our knowledge base. The 2nd part is when we actually upload a dataset to the web ui, it kicks off a RAG agent to read the documentation and recommend the best model for the job, then recall that model code.

Chosen tooling:
- QDrant: I chose this vector store because we tested it in class and I was fine with it, it's extensible, supporting local self-hosting and also local models, and it was fairly easy to figure out how to bundle it together with our other tools via docker compose.
- Langfuse: I chose this based on my work with it in the class. It is pretty much functionally identical to LangSmith and offers us data governance and traceability for evaluating what decisions the model made and what the model's reasoning was at any given step.
- Minimax: I chose minimax-m2.5 based on its open source nature and also a healthy balance of speed and quality. It's quantized for added speed, but only quantized to 8-bit to keep as much model quality and ability available.
- Gradio: I chose this because I've worked with it for years as a web ui for showcasing ML projects, and because it bundles FastAPI, giving you the API functionality for the main app.py functions along with documentation for that API for free.
- Docker: I chose docker for deployment because I want this to not only be a good experience for the users, but also for the volunteers tasked with maintaining this and the queue that will go around it.
- MetaFlow: This is an ML training platform I've used professionally for several years now, it gives some incredible visibility into the training process and where failures occur, as well as allowing a lot of the features that come downstream like A/B testing particular rows of data

Section 3
Default Chunking strategy:
It's just RecursiveCharacterTextSplitter chunk_size: int = 500, chunk_overlap: int = 100. I tested several strategies including semantic chunking for this and did not get any performance improvement, likely because of the domain.
Data Source:
The data source is Sci-Kit Learn documentation, summarized and evaluated that I've put in the data/knowledge_base folder. I focused on supervised learning and model selection, and will add more later.
External APIs/Tools for Agent Component:
The agent component currently uses only the locally-hosted vLLM server (OpenAI-compatible API at http://192.168.1.185:8080/v1) for LLM inference - no external APIs like Tavily search are currently implemented. The agent's tools/capabilities are limited to: (1) dataset file analysis via Python/pandas, (2) format detection for training methods (SFT/DPO/GRPO), and (3) rule-based recommendations when the LLM is unavailable. External search APIs are planned but not yet implemented.

Section 4
Done =)

Section 5
RAGAS Evaluation Results on specifically the questions above in section 1:
Faithfulness, Context Precision, Context Recall
50%, 92%, 33%
57%, 89%, 100%
86%, 83%, 50%
100%, 70%, 100%
50%, 99.9%, 100%
83%, 99.9%, 100%
100%, 99.9%, 100%
25%, 99.9%, 100%
100%, 99.9%, 100%
null, 25%, 100%

2. I'm drawing specifically the conclusion that faithfulness will be my most important metric here, as it's the one with the largest distribution of results. I want to specifically examine question 4, why did it get 100% faithfulness and recall but only 70% precision while most questions the dense retrieval got 99.9% precision. That said, the RAG pipeline seems to be working swimmingly!

Section 6
Advanced Retrieval:
Hybrid Retrieval (BM25 + Dense with Reciprocal Rank Fusion) is implemented in `src/retrieval/hybrid_retriever.py`. This approach improves RAG accuracy for technical documentation queries by combining semantic understanding (dense retrieval) with exact keyword matching (BM25), which is particularly valuable for sklearn's domain-specific terminology and configuration parameters. The hybrid method is available in the UI alongside pure dense retrieval.

Section 7
Next Steps:
For Demo Day, I plan to keep Dense retrieval as the default and offer Hybrid as an advanced option. While Hybrid retrieval theoretically provides better accuracy by combining semantic and keyword matching, Dense retrieval is faster and simpler to explain. The comparison results from RAGAS evaluation will inform whether Hybrid should become the recommended default after Demo Day.

Section 8
Deep Error Investigation Agent:
The platform includes a multi-agent LangGraph system for debugging ML/LLM training and inference errors. When errors occur during training or inference, the system:
1. Analyzes the error context (error type, message, traceback, task type, model info)
2. Uses a multi-agent workflow: query generation → search → quality analysis → doc fetch → recommendation synthesis
3. Learns from past investigations using Qdrant semantic memory
4. Automatically refines search queries if initial results are poor
5. Provides hardware-specific recommendations (CUDA 13.1, DGX Spark)

Key components:
- `src/agent/investigation_graph.py` - LangGraph StateGraph orchestration
- `src/agent/sub_agents/` - Specialized agent nodes
- `src/agent/investigation_memory.py` - Qdrant-backed semantic memory
- `src/agent/investigation_planner.py` - Planning and iteration logic