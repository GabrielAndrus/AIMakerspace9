Section 1
1. The people of the Forge Utah community have need to train ML models, but don't know how and aren't data scientists.
2. There are many people in the Forge Utah community who are interested in the predictive power of classical ML. They've heard of gradient-boosted trees and regression, and likely they've even worked with or around people who have given them insights using ML models. Not everyone, however is technical, nor a data scientist. There are technical developers that are part of the community whose eyes glaze over immediately upon mentioning anything Bayesian, and product managers with oodles of ideas for how to leverage good predictive power, but with the assumption that someone else is just going to build it. Even those who ARE data scientists don't have the entireity of the scikit-learn catalogue memorized, nor do they know how to apply all of it, and even fewer still know about training LLMs. Forge Utah has recently come into some compute in the form of 2 DGX Sparks, but this adds another layer of complexity, where only someone who knows both the Spark _and_ data science would be able to effectively train anything.
3. [
    {"question": "I have a CSV file with customer purchase amounts and want to predict future spending. What model should I use?", "answer": "For predicting continuous numerical values like purchase amounts, the system will recommend a regression model such as Linear Regression, Ridge Regression, or Gradient Boosting Regressor depending on the data characteristics."},
    {"question": "I want to predict whether a loan will be approved or denied based on applicant data. What model type do I need?", "answer": "This is a binary classification problem. The system will recommend classifiers like Logistic Regression, Random Forest Classifier, or Gradient Boosting Classifier based on your data structure."},
    {"question": "I have a dataset with 10,000 rows and 50 features. Will the system still work?", "answer": "Yes, the agent will analyze your dataset dimensions and recommend appropriate models. For higher-dimensional data, it may suggest dimensionality reduction techniques (PCA) or regularization-heavy models to prevent overfitting."},
    {"question": "I don't know anything about machine learning. Can I still use this platform?", "answer": "Yes, the entire purpose of this platform is to serve non-technical users. You simply upload your dataset, and the agent walks you through model selection and training without requiring ML expertise."},
    {"question": "My dataset has missing values. How does the system handle this?", "answer": "The RAG agent will retrieve documentation on imputation strategies from scikit-learn and recommend appropriate handling (mean/median imputation, KNN imputation, or dropping features) based on your specific data."},
    {"question": "How do I verify the model actually works after training?", "answer": "The platform includes an inference playground where you can load your trained model and test it against a validation dataset or input new predictions to verify accuracy."},
    {"question": "Can I access this through an API instead of the web UI?", "answer": "Yes, Gradio bundles FastAPI automatically, so you have both a web interface and a documented REST API for programmatic access."},
    {"question": "I have text data I want to classify into categories. What will the system recommend?", "answer": "For text classification, the agent will recommend models like Naive Bayes (MultinomialNB), LinearSVC, or ensemble methods after potentially suggesting TF-IDF vectorization or embedding approaches."},
    {"question": "The system recommended a model I don't recognize. Where can I learn more?", "answer": "The system provides links to the scikit-learn documentation for each recommended model, allowing you to understand what the algorithm does and why it was selected."},
    {"question": "I have a small dataset with only 100 rows. Is that enough data?", "answer": "The agent will assess your dataset size and may recommend simpler models (like Logistic Regression or Decision Trees) that are less prone to overfitting on small datasets, rather than complex deep learning approaches."}
]

Section 2
Proposed Solution:
Agentic AutoML
This platform uses the scikit learn documentation itself to determine, based on an uploaded dataset (either tabular or text-based) what is the best model to train for predicting, and what method is best for completing that training. There are several piece to this that make it work: QDrant vector store, using Qwen3-embedding-4b to embed our knowledge base for retrieval, minimax-m2.5-mlx@8bit for making the ultimate decision, and the platform of the Spark inside Nvidia's golden pytorch docker container to actually _do_ the training. Once the training is done, we don't have to simply trust that the LLM has done a good job, there is an inference playground for loading the model and checking against whatever golden dataset is required. All of this is, of course, accessible through both the webui as well as a straight API call to the running gradio server.
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
│  └─────────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────────┬─────────────────────┘
                      │                                               │
          ┌───────────▼───────────┐                   ┌───────────────▼────────────────┐
          │   LLM Decision Maker  │                   │    Monitoring & Evaluation     │
          │                       │                   │                                │
          │  minimax-m2.5-mlx     │                   │    Langfuse                    │
          │  (8-bit quantized)    │                   │    - Trace model decisions     │
          │                       │                   │    - Log evaluation metrics    │
          │  Chosen for: balance  │                   │    - Audit reasoning steps     │
          │  of speed & quality   │                   └────────────────────────────────┘
          └───────────────────────┘
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
- **LLM (minimax-m2.5-mlx@8bit)**: Chosen for its open-source nature and balanced speed/quality; 8-bit quantization provides fast inference while preserving model quality for decision-making.
- **Agent Orchestration (Custom/LangGraph)**: Chosen to orchestrate the multi-step pipeline from dataset analysis through RAG retrieval to model selection and training execution.
- **Tools (Python/sklearn)**: Chosen as the foundation since scikit-learn is the core library being wrapped, and Python provides seamless integration across all components.
- **Embedding Model (Qwen/Qwen3-Embedding-4B)**: Chosen for its open-source availability and 2560-dimensional embeddings that capture rich semantic relationships in sklearn documentation.
- **Vector Database (QDrant)**: Chosen because it was tested in class, supports local self-hosting, and integrates easily via docker compose for the entire stack.
- **Monitoring (Langfuse)**: Chosen for data governance and traceability, allowing evaluation of model decisions and auditing the agent's reasoning at each step.
- **Evaluation Framework (Built-in + Langfuse)**: Chosen to combine built-in inference playground for manual testing with Langfuse traces for automated evaluation.
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

Section 4
Done =)