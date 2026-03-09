<p align = "center" draggable="false" ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719"
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 16: LLM Servers</h1>

| 📰 Session Sheet                                  | ⏺️ Recording                           | 🖼️ Slides                                   | 👨‍💻 Repo       | 📝 Homework                                              | 📁 Feedback                        |
| ------------------------------------------------- | -------------------------------------- | ------------------------------------------- | ------------- | -------------------------------------------------------- | ---------------------------------- |
| [LLM Servers](../00_Docs/Session_Sheets/16_LLM_Servers) |[Recording!](https://us02web.zoom.us/rec/share/HDunij9p7eCXeP_OgsRDRjTdWUqiEhDBGWrFJEn1bwWR1wz1jKX6EHXSOM45d0sC.rHiyo_znZ-R8Jh6S) <br> passcode: `D80X^YjL`| [Session 16 Slides](https://www.canva.com/design/DAG-EBu7B5A/POcowC5rDLENSPcSVpbf8g/edit?utm_content=DAG-EBu7B5A&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 16 Assignment: LLM Servers](https://forms.gle/Riqvwf6KrZcCRKV86) <br><br> [Demo Day Submission (3/12)](https://forms.gle/7xyuBUn69GX4v6K98)  | [Feedback 3/5](https://forms.gle/W28QFWJXpSS4ZAR6A) |

**⚠️!!! PLEASE BE SURE TO SHUTDOWN YOUR DEDICATED ENDPOINT ON FIREWORKS AI WHEN YOU'RE FINISHED YOUR ASSIGNMENT !!!⚠️**

# Build 🏗️

In today's assignment, we'll be creating Fireworks AI endpoints, and then building a RAG application.

- 🤝 Breakout Room #1
  - Set-up Open Source Endpoint (Instructions [here](./ENDPOINT_SETUP.md)) ((This process may take 15-20min.))
  - Test Endpoint and Embeddings with the `endpoint_slammer.ipynb` notebook.

- 🤝 Breakout Room #2
  - Use the Open Source Endpoints to build a RAG LangGraph application

# Ship 🚢

The completed notebook and your RAG app/notebook!

### Deliverables

- A short Loom of either:
  - the notebook and the RAG application you built for the Main Homework Assignment; or
  - the notebook you created for the Advanced Build

# Share 🚀

Make a social media post about your final application!

### Deliverables

- Make a post on any social media platform about what you built!

Here's a template to get you started:

```
🚀 Exciting News! 🚀

I am thrilled to announce that I have just built and shipped a RAG application powered by open-source endpoints! 🎉🤖

🔍 Three Key Takeaways:
1️⃣
2️⃣
3️⃣

Let's continue pushing the boundaries of what's possible in the world of AI and question-answering. Here's to many more innovations! 🚀
Shout out to @AIMakerspace !

#LangChain #QuestionAnswering #RetrievalAugmented #Innovation #AI #TechMilestone

Feel free to reach out if you're curious or would like to collaborate on similar projects! 🤝🔥
```

# Submitting You Homework [OPTIONAL]

## Main Homework Assignment

Follow these steps to prepare and submit your homework assignment:

1. Follow the instructions in `ENDPOINT_SETUP.md`
2. Replace both `model` values in `endpoint_slammer.ipynb` with the `gpt-oss` endpoint you created in Step 1
3. Run the code cells in `endpoint_slammer.ipynb`
4. Respond to the questions in the section below
5. Build a sample RAG
6. Record a Loom video reviewing what you have learned from this session

**⚠️!!! PLEASE BE SURE TO SHUTDOWN YOUR DEDICATED ENDPOINT ON FIREWORKS AI WHEN YOU HAVE FINISHED YOUR ASSIGNMENT !!!⚠️**

## Questions

### ❓ Question #1:

What is the difference between serverless and dedicated endpoints?

#### ✅ Answer:

Serverless endpoints share infrastructure across multiple users with pay-per-use pricing but may have variable latency due to resource contention. Dedicated endpoints reserve exclusive compute resources for consistent performance and guaranteed capacity at a higher fixed cost.

### ❓ Question #2:

Why is it important to consider token throughput and latency when choosing an LLM for user-facing applications?

#### ✅ Answer:

Token throughput determines how many requests you can handle concurrently, directly impacting scalability and user wait times. Latency affects the real-time responsiveness users experience - high latency makes applications feel sluggish and degrades user engagement.

## Activity 1: RAGAS Evaluation with Cost Analysis

Use RAGAS to evaluate your open-source Fireworks AI powered RAG app against an OpenAI `gpt-4.1-mini` powered equivalent. Compare retrieval quality, answer faithfulness, and end-to-end accuracy across both providers.

Additionally, instrument both pipelines with **LangSmith** to capture token usage and cost per query. Use LangSmith's tracing and cost dashboards to compare the total cost of running each provider at scale. Include your evaluation results, cost breakdown, and analysis in your Loom video.

## Advanced Activity: Local Models

Swap out the Fireworks AI endpoints for **locally-running open-source models** using [Ollama](https://ollama.com/) or another local inference server of your choice. Run both your embedding model and your chat model locally, and rebuild the RAG pipeline on top of them.

- Compare quality and latency between the local setup and your Fireworks AI hosted endpoint.
- Reflect: what are the trade-offs of local models vs. managed endpoints in a production setting?

Include your findings and a demo in your Loom video.

---

### Activity Results: Fireworks API vs Local Model (glm-5)

**Test Setup:**
- Fireworks API
- Local Model: `unsloth/glm-5`
- RAG context: Cat health guide PDF (cat-health-guide.pdf)
- Embeddings: `text-embedding-nomic-embed-text-v1.5`

**Latency Results (MEASURED):**

| Query | Fireworks API | Local Model (glm-5) |
|-------|---------------|---------------------|
| Q1: Life stages | 14.22s | 164.60s |
| Q2: Vaccination | 20.30s | 204.73s |
| Q3: Senior diet | 33.01s | 202.44s |
| Q4: Parasites | 31.57s | 203.53s |
| Q5: Behavioral signs | 41.27s | 224.22s |
| **AVG** | **28.07s** | **199.90s** |

- **Performance gap**: Fireworks API is ~7x faster (612% faster)

**Quality Comparison Analysis:**

#### Query 1: "What are the key life stages for cats?"

**Fireworks API Response:**
> Based on the provided context, the key life stages for cats are:
> 1. Kitten: From birth up to 1 year.
> 2. Young Adult: From 1 year through 6 years.
> 3. Mature Adult: From 7 to 10 years.
> 4. Senior: Aged over 10 years.
> 5. End-of-life: Can occur at any age.

**Local Model glm-5 Response:**
> Based on the provided documents, the key life stages for cats are defined by the 2021 AAHA/AAFP Feline Life Stage Guidelines as a five-stage grouping...
> [Same 5 stages listed]

**Analysis:** Both accurate. Local model added source attribution ("2021 AAHA/AAFP Feline Life Stage Guidelines") showing better context awareness.

---

#### Query 2: "How often should kittens be vaccinated?"

**Fireworks API Response:**
> Based on the provided context, kittens should be revaccinated against FPV, FHV-1, and FCV at 6 months of age. For FeLV, it is recommended to revaccinate 12 months after the last dose in the kitten series.

**Local Model glm-5 Response:**
> ...revaccination aims to reduce the window of susceptibility in kittens with maternally derived antibodies following the end of the initial kitten series (which typically ends around 16–18 weeks).
> The context does not specify the exact frequency or intervals for the initial doses within the "kitten series" itself.

**Analysis:** Local model BETTER - it explained WHY the 6-month revaccination matters and honestly acknowledged what the context DOESN'T contain. Fireworks API gave a more direct answer but lacked explanatory depth.

---

#### Query 3: "What is the recommended diet for senior cats?"

**Fireworks API Response:** (33.01s, ~900 chars)
- Listed energy requirements, protein levels, therapeutic diets
- Included kidney disease specifics and feeding strategy

**Local Model glm-5 Response:** (202.44s, ~850 chars)
- Added explicit context: "Senior cats (over 10 years of age) often require increased caloric intake because they may experience a reduction in digestive capabilities"
- Explained the REASONING behind recommendations

**Analysis:** Comparable quality. Local model provided better explanatory reasoning for WHY senior cats need more calories.

---

#### Query 4: "How can I prevent parasites in my cat?"

**Fireworks API Response:**
> ...Using effective flea prevention: This lowers the risk of cutaneous and systemic diseases and can reduce the risk of zoonotic diseases like cat scratch fever.

**Local Model glm-5 Response:**
> Regular flea prevention can also reduce the risk of zoonotic diseases like cat scratch fever (*Bartonella henselae*).

**Analysis:** Local model included scientific name (*Bartonella henselae*) showing better medical terminology retention. Both models covered same core points.

---

#### Query 5: "What behavioral changes indicate illness in cats?"

**Fireworks API Response:** (~700 chars)
- Covered grooming habits, activity/demeanor, elimination
- Missed gastrointestinal signs

**Local Model glm-5 Response:** (~1000 chars)
- Included all Fireworks points PLUS:
  - "Vomiting (including hairballs) and diarrhea can indicate early stages of disease"
  - Cardiac conditions (hypertrophic cardiomyopathy) linked to decreased activity
- More comprehensive coverage

**Analysis:** Local model MORE COMPLETE - captured additional behavioral indicators that Fireworks missed.

---

### Key Quality Differences:

**Fireworks API (glm-4.7-flash):**
- ✅ ~7x faster latency
- ✅ Concise, direct answers
- ✅ Good accuracy for straightforward queries
- ❌ Sometimes misses nuanced details
- ❌ Less explanatory reasoning

**Local Model (glm-5):**
- ✅ More comprehensive responses
- ✅ Better source attribution and context awareness
- ✅ Includes scientific/medical terminology
- ✅ Acknowledges information gaps honestly
- ❌ Much slower (~200s per query)

### Hallucination Check:
- **Neither model hallucinated** significantly beyond the provided RAG context
- Local model correctly stated "The context does not specify..." when info was missing (Q2)
- Both stayed grounded in retrieved chunks

### Recommendations:

- **Production/chat apps**: Use Fireworks API for speed
- **Medical/veterinary queries**: Consider Local Model for thoroughness
- **Hybrid approach**: Route simple factual queries to Fireworks, complex diagnostic questions to Local Model

### Trade-offs: Local Models vs Managed Endpoints

**Local Models (glm-5):**
- ✅ **Privacy**: Data never leaves your infrastructure
- ✅ **No rate limits**: Unlimited queries at no marginal cost
- ✅ **Customization**: Full control over model parameters and fine-tuning
- ❌ **Hardware costs**: Requires GPU/CPU investment and maintenance
- ❌ **Latency**: Often slower due to local hardware limitations
- ❌ **Scalability**: Limited by local resources, harder to scale under load

**Managed Endpoints (Fireworks API):**
- ✅ **Performance**: Optimized infrastructure with low latency (~7x faster in this test)
- ✅ **Scalability**: Auto-scales with demand without hardware management
- ✅ **Reliability**: Managed uptime, updates, and failover
- ❌ **Cost**: Pay-per-use can be expensive at scale
- ❌ **Data privacy**: Requests sent to external servers
- ❌ **Rate limits**: API throttling under heavy use

**Production Decision Framework:**
- Use **managed endpoints (Fireworks API)** for: production apps with variable load, when latency matters most, when you need reliability SLAs
- Use **local models (glm-5)** for: sensitive data workloads, development/testing, fixed predictable workloads, when privacy is paramount or response thoroughness is critical
