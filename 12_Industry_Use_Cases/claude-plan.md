# Comprehensive Code Review & Fix Plan

## Context

This Agentic AutoML Platform serves non-technical users (Forge Utah community) who need to train ML models without data science expertise. Two core goals:

1. **Absolute visibility** - Users who make mistakes are guided by the Error Investigator agent to understand and fix issues. Non-data-scientists should be taught how to use the interface correctly via clear error explanations.
2. **Zero ML knowledge required** - RAG from sklearn docs powers automatic model selection and training. Users upload data, the agent determines the right model, and the system trains it automatically.

The codebase has accumulated bugs, inconsistencies, incomplete Langfuse integration, and missing UI features that undermine both goals. This plan addresses every identified issue systematically.

---

## Assumptions

1. **The DGX Sparks are at 192.168.1.79 and 192.168.1.36** - The LLM inference server (vLLM serving minimax-m2.5-mlx@8bit) is at `192.168.1.79:8080` and searxng is at `192.168.1.36:4000`. The old IP `192.168.1.185` in config.py is stale.
2. **The model name is `minimax-m2.5-mlx@8bit`** - This is the correct model name for all LLM calls. The `openai/gpt-oss-120b` strings scattered across sub-agents are wrong.
3. **Docker environment** - The app runs inside an NVIDIA PyTorch container (`nvcr.io/nvidia/pytorch:26.01-py3`) with CUDA 13.1. Quantization via bitsandbytes is NOT compatible.
4. **Langfuse is deployed** - Langfuse services (web, worker, postgres, clickhouse, minio, redis) are defined in docker-compose.yml and expected to be running at `http://localhost:3000`.
5. **Searxng is deployed** - The searxng instance is running at `http://192.168.1.36:4000` for web search during error investigation.
6. **All user interactions happen through the Gradio web UI** - Terminal/stdout output is NOT visible to end users. Any information that should reach the user MUST be rendered in the Gradio HTML interface.
7. **Metaflow is configured with local metadata** - `METAFLOW_DEFAULT_METADATA=local` is set at module import time in runner.py.
8. **The `peft` library is a required dependency** - It's listed in pyproject.toml, so the fallback import from transformers is unnecessary.

---

## Reasoning

### Why these issues matter for the two goals:

**Goal 1 (Visibility/Error Guidance):**
- The Error Investigator agent runs and produces recommendations, but prints them to stdout. The Gradio UI user never sees them. This completely defeats the purpose.
- Langfuse is barely implemented (only `create_score`), so there's no observability into what the system is doing, which agents fired, what the LLM reasoned about, or where things went wrong.
- Several bugs (search_executor parsing, streaming generation) will cause silent failures that confuse users.

**Goal 2 (Zero ML Knowledge):**
- The LLM calls that power model selection and dataset analysis all use the wrong model name and URL, meaning they'll fail to connect to the actual inference server.
- The LLM Inference tab (where users test their trained models) was built but never added to the UI - users can't actually use their trained LoRA adapters.
- The save_model step in LLM training tries to load a LoRA adapter as a full model, which will crash.

---

## Phase 1: Critical Bug Fixes

### 1.1 Fix config.py - Wrong LLM server settings
**File:** `src/config.py`

**Problem:** `LLM_INFERENCE_URL` defaults to `http://192.168.1.185:8080/v1` (stale IP). No config for model name or searxng URL.

**Fix:**
- Change `LLM_INFERENCE_URL` default to `http://192.168.1.79:8080/v1`
- Add `LLM_MODEL_NAME: str = os.environ.get("LLM_MODEL_NAME", "minimax-m2.5-mlx@8bit")`
- Add `SEARXNG_URL: str = os.environ.get("SEARXNG_URL", "http://192.168.1.36:4000")`

**Reasoning:** Centralizing all service URLs in config.py means a single place to update when IPs change. Environment variable overrides allow docker-compose to set them without code changes.

### 1.2 Propagate correct LLM model name everywhere
**Problem:** 6 files hardcode `"openai/gpt-oss-120b"` as the model name for `init_chat_model`. The actual model is `minimax-m2.5-mlx@8bit`.

**Files to fix:**
- `src/agent/error_investigator.py:563` - `ErrorInvestigatorAgent.__init__` default param
- `src/agent/sub_agents/query_generator.py:71` - `init_chat_model("openai/gpt-oss-120b", ...)`
- `src/agent/sub_agents/search_analyzer.py:61` - same
- `src/agent/sub_agents/synthesizer.py:72` - same
- `src/agent/dataset_analyzer.py:33,46` - `__init__` default and `_setup_model`
- `src/agent/model_selector.py:32` - already has correct name but should use config

**Fix:** Replace all hardcoded model names with `settings.LLM_MODEL_NAME`. Also need to verify `init_chat_model` config parameter works correctly for setting base_url/api_key with the openai provider - if not, switch to direct `ChatOpenAI(base_url=..., api_key=..., model=...)` instantiation.

**Reasoning:** `init_chat_model` is a LangChain utility that may pass config differently than expected. The `model_provider="openai"` + `config={"base_url": ...}` pattern needs verification. If it doesn't work, `ChatOpenAI` from `langchain_openai` is the reliable alternative.

### 1.3 Move SEARXNG_URL to config
**File:** `src/agent/error_investigator.py`

**Problem:** Line 21 hardcodes `SEARXNG_URL = "http://192.168.1.36:4000"`. This is used by `search_searxng`, `_search_sync`, and the fallback investigation.

**Fix:** Replace with `from src.config import settings` and `SEARXNG_URL = settings.SEARXNG_URL`

### 1.4 Fix inference_server.py hardcoded values
**File:** `src/llm/inference_server.py`

**Problem:** Default `base_url` is `http://192.168.1.79:8080/v1` (correct but hardcoded). Default model is `minimax-m2.5-mlx@8bit` (correct but hardcoded in 3 functions).

**Fix:** Import settings and use `settings.LLM_INFERENCE_URL`, `settings.LLM_MODEL_NAME` as defaults.

### 1.5 Fix streaming generation bug (NameError)
**File:** `src/llm/local_inference.py:251-266`

**Problem:** At line 255, `"streamer": streamer` references `streamer` before it's defined (defined at line 264). Additionally, lines 251-259 create a `generation_kwargs` dict that's immediately overwritten at lines 281-286 (dead code).

**Fix:**
1. Delete lines 251-259 (the first `generation_kwargs` block - it's dead code)
2. Move the `TextIteratorStreamer` creation (lines 261-266) to before the second `generation_kwargs` block (line 281)

**Reasoning:** This is clearly a copy-paste error. The code was refactored but the old block wasn't removed, and the variable ordering got scrambled.

### 1.6 Fix search_executor parsing bug
**File:** `src/agent/sub_agents/search_executor.py:44`

**Problem:** `search_results.count()` is called without an argument. `list.count(x)` counts occurrences of `x` in the list - calling it with no args raises TypeError. Should be `len(search_results)`.

**Additionally:** The line `line.startswith(f"{search_results.count() + 1}. ")` is trying to match numbered results like "1. Title". The logic should be `line.startswith(f"{len(search_results) + 1}. ")`.

**Fix:** Replace `search_results.count()` with `len(search_results)`.

**Impact:** Without this fix, the entire search execution pipeline crashes, meaning the Error Investigator (Goal 1) can never find solutions.

### 1.7 Fix PeftModel import fallback
**File:** `src/llm/local_inference.py:73-80`

**Problem:** The except block tries `from transformers import PeftModel` which will always fail because PeftModel is only in the `peft` library.

**Fix:** Remove the try/except entirely and just do `from peft import PeftModel`. If peft isn't installed, let the error propagate clearly.

**Reasoning:** `peft` is already in pyproject.toml as a dependency. A silent fallback to a non-existent import just creates a confusing error.

### 1.8 Fix LLM training save_model step
**File:** `src/flows/llm_training_flow.py:271-276`

**Problem:** `AutoModelForCausalLM.from_pretrained(self.adapter_path)` tries to load the LoRA adapter directory as if it were a complete model. LoRA adapters only contain adapter weights, not the full model. This will fail.

**Fix:** The trainer already saves the adapter. Instead of reloading the model, just save metadata alongside the existing adapter:
```python
# Instead of reloading, use save_lora_adapter_package with paths only
self.model_path = save_lora_adapter_package(
    model=None,  # Don't reload
    tokenizer=None,
    output_dir=settings.MODEL_DIR,
    model_name=f"llm_{self.training_method.lower()}",
    version=str(current.run_id),
    base_model=self.base_model,
    adapter_path=self.adapter_path,  # Just reference existing adapter
    ...
)
```
OR load the base model first, apply the adapter, then save:
```python
model = AutoModelForCausalLM.from_pretrained(self.base_model, ...)
model = PeftModel.from_pretrained(model, self.adapter_path)
```

**Reasoning:** Need to check what `save_lora_adapter_package` actually requires to determine the best approach. The simplest fix may be to just copy the adapter directory and write a metadata.json file.

### 1.9 Fix GRPO reward template auto-detection
**File:** `src/flows/llm_training_flow.py:210-217`

**Problem:** If the first dataset example has neither `ground_truth` nor `pattern`, `detected_template` stays `None`, which will likely cause `train_grpo` to fail with an unclear error.

**Fix:** Add a clear error message:
```python
if detected_template is None:
    raise ValueError(
        "GRPO training requires a reward template. Either:\n"
        "1. Select 'math' or 'format_check' in the UI\n"
        "2. Include 'ground_truth' field in your JSONL for math rewards\n"
        "3. Include 'pattern' field in your JSONL for format checking"
    )
```

**Reasoning:** This aligns with Goal 1 (visibility) - non-technical users need clear guidance on what went wrong and how to fix it.

---

## Phase 2: Missing UI Features

### 2.1 Add LLM Inference tab to main app
**File:** `src/app.py:1368-1375`

**Problem:** `create_llm_inference_tab()` is defined (lines 1242-1365) but never called in the `gr.Blocks` context. The current UI only has 4 tabs: Tabular ML, LLM Fine-tuning, Inference Playground (tabular only), RAGAS Evaluation.

**Fix:** Add `create_llm_inference_tab()` after `create_ragas_evaluation_tab()` in the Blocks context:
```python
with gr.Blocks(...) as demo:
    gr.Markdown("# Agentic AutoML Platform")
    create_tabular_ml_tab()
    create_llm_finetuning_tab()
    create_inference_playground_tab()
    create_llm_inference_tab()       # <-- ADD THIS
    create_ragas_evaluation_tab()
```

**Reasoning:** Users who fine-tune LLM models currently have no way to test them in the UI. This is a critical missing feature for Goal 2.

### 2.2 Verify _investigate_training_error reference in app.py
**File:** `src/app.py:648`

**Problem:** `_investigate_training_error` is called in `train_llm_model`'s except block, but this function is defined in `src/flows/runner.py`, not in app.py. The app.py file has `_investigate_inference_error` (line 22) but not the training version.

**Fix:** Either import it from runner.py or define a similar wrapper in app.py. Since `_investigate_training_error` in runner.py takes different parameters (includes `flow_name`), the call at line 648 needs to match that signature.

---

## Phase 3: Langfuse Full Observability

### 3.1 Rewrite langfuse_client.py with proper tracing
**File:** `src/utils/langfuse_client.py`

**Problem:** Current implementation only uses `create_score()` to log numeric scores with comments. This is NOT tracing - it doesn't capture inputs/outputs, doesn't create trace hierarchies, and doesn't record LLM generations.

**Fix:** Replace with proper Langfuse SDK usage:

```python
from langfuse import Langfuse
from contextlib import contextmanager

_langfuse_client = None

def get_langfuse_client():
    """Get or create Langfuse client singleton."""
    global _langfuse_client
    if _langfuse_client is None:
        try:
            from src.config import settings
            _langfuse_client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
        except Exception as e:
            print(f"Warning: Langfuse init failed: {e}")
    return _langfuse_client

@contextmanager
def langfuse_trace(name, metadata=None, user_id=None):
    """Context manager for wrapping operations in Langfuse traces."""
    client = get_langfuse_client()
    if client is None:
        yield None
        return
    trace = client.trace(name=name, metadata=metadata or {}, user_id=user_id)
    try:
        yield trace
    finally:
        client.flush()

def trace_llm_call(trace, name, input_text, output_text, model, metadata=None):
    """Record an LLM generation within a trace."""
    if trace is None:
        return
    trace.generation(
        name=name,
        input=input_text,
        output=output_text,
        model=model,
        metadata=metadata or {},
    )

def trace_retrieval(trace, name, query, results, metadata=None):
    """Record a retrieval operation within a trace."""
    if trace is None:
        return
    trace.span(
        name=name,
        input={"query": query},
        output={"results": results[:5] if isinstance(results, list) else str(results)[:500]},
        metadata=metadata or {},
    )
```

**Reasoning:** The Langfuse Python SDK provides `trace()`, `generation()`, and `span()` methods that create proper observability. Using a context manager pattern means traces are always flushed even if exceptions occur. The `if trace is None` guards ensure graceful degradation when Langfuse is unavailable.

### 3.2 Instrument all LLM calls with Langfuse

Every file that calls an LLM needs to be wrapped with tracing. The pattern is:

```python
from src.utils.langfuse_client import langfuse_trace, trace_llm_call

with langfuse_trace("operation_name", metadata={...}) as trace:
    # ... prepare prompt ...
    response = model.invoke(prompt)
    trace_llm_call(trace, "step_name", prompt, response.content, settings.LLM_MODEL_NAME)
```

**Files to instrument:**

| File | Function | What to trace |
|------|----------|---------------|
| `src/agent/error_investigator.py` | `ErrorInvestigatorAgent.investigate()` | Wrap full investigation in a trace, add generation spans for query gen and recommendation |
| `src/agent/sub_agents/query_generator.py` | `query_generator_node()` | Generation span for query generation LLM call |
| `src/agent/sub_agents/search_analyzer.py` | `search_analyzer_node()` | Generation span for quality evaluation LLM call |
| `src/agent/sub_agents/synthesizer.py` | `synthesizer_node()` | Generation span for synthesis LLM call |
| `src/agent/dataset_analyzer.py` | `DatasetAnalyzer._run_agent_analysis()` | Generation span for dataset analysis LLM call (replace current broken Langfuse callback) |
| `src/agent/model_selector.py` | `ModelSelector.select_model()` | Trace wrapping the full selection + generation span |
| `src/ml/data_validator.py` | `detect_task_type_and_recommend_model()` | Trace + retrieval span + generation span |
| `src/llm/inference_server.py` | `LLMInferenceServer.generate()` | Generation span for user-facing inference |
| `src/evaluation/ragas_evaluator.py` | `evaluate_dataset()` | Trace wrapping the full evaluation |

### 3.3 Instrument retrieval operations
**Files:**
- `src/retrieval/qdrant_client.py` - Add `trace_retrieval()` calls to search operations
- `src/retrieval/hybrid_retriever.py` - Add spans showing dense vs sparse vs fusion results

### 3.4 Instrument training flows
**Files:**
- `src/flows/runner.py` - Wrap `run_ml_training_flow()` and `run_llm_training_flow()` with traces including metadata (data_path, training_method, model, epochs, etc.)

### 3.5 Fix broken Langfuse setup in dataset_analyzer.py
**File:** `src/agent/dataset_analyzer.py:59-66`

**Problem:** Creates a `Langfuse()` instance that's never stored. Creates `LangfuseCallbackHandler()` which requires the global Langfuse client to be configured correctly.

**Fix:** Remove the custom Langfuse setup. Use the centralized `langfuse_client.py` utilities instead. Replace the callback handler pattern with explicit `trace_llm_call()`.

---

## Phase 4: Metaflow Pipeline Coverage

### 4.1 Current state of Metaflow integration

**Currently pipelined via Metaflow:**
- ML training: `train_tabular_model()` -> `run_ml_training_flow()` -> `MLTrainingFlow` (7 steps)
- LLM training: `train_llm_model()` -> `run_llm_training_flow()` -> `LLMTrainingFlow` (7 steps)

**NOT pipelined (and assessment):**
- ML inference (`predict_with_model`) - Direct joblib load. **Keep as-is** - inference is instant, no pipeline overhead needed
- LLM inference (`generate_with_lora`) - Direct model call. **Keep as-is** - interactive use requires low latency
- RAGAS evaluation - Direct evaluation. **Consider wrapping** in a flow for visibility into evaluation runs
- Knowledge base indexing - Not exposed in UI. **Keep as-is** - one-time setup operation

### 4.2 Improve Metaflow error reporting to UI
**File:** `src/flows/runner.py`

**Problem:** When a flow fails, `_investigate_training_error()` prints investigation results to stdout (lines 29-46). The Gradio user never sees this because they only see the HTML in the web interface.

**Fix:** Modify `_investigate_training_error()` to return the investigation text instead of (or in addition to) printing it. Then the calling code in `run_ml_training_flow()` and `run_llm_training_flow()` can include it in the RuntimeError message, which app.py then renders in the error HTML.

---

## Phase 5: Code Cleanup

### 5.1 Remove debug print statements
**File:** `src/app.py`

Lines 256-257, 268, 274, 284, 295-296, 378-379 all have `print(f"[DEBUG] ...")` statements that should be removed for production.

### 5.2 Remove dead code in local_inference.py
**File:** `src/llm/local_inference.py:251-259`

The first `generation_kwargs` block is dead code (immediately overwritten at line 281). Remove it as part of the streaming fix (Phase 1.5).

### 5.3 Clean up redundant config code
**File:** `src/config.py:9-10`

```python
if token := os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = token  # Redundant - already in env
```
This reads HF_TOKEN from env and writes it back to env. Remove.

### 5.4 Fix model_selector.py to use config
**File:** `src/agent/model_selector.py`

Uses `os.getenv()` directly (lines 12-13) instead of `settings`. Import and use `settings.LLM_INFERENCE_URL`, `settings.LLM_INFERENCE_KEY`, `settings.LLM_MODEL_NAME`.

### 5.5 Consolidate duplicate report generation
**File:** `src/agent/dataset_analyzer.py`

`generate_report()` method (line 308) and `get_training_recommendation()` function (line 339) both build identical HTML reports. Refactor `get_training_recommendation()` to call `analyzer.generate_report(analysis)`.

### 5.6 Evaluate files for removal
- `HYBRID_FIX_SUMMARY.md` - Development artifact, not needed for production
- `error-data-flow.md` - Development artifact
- `1772919487420021.zip` - 3.4MB zip file in repo root - investigate and remove if not needed

### 5.7 Assess ErrorInvestigatorAgent class usage
**File:** `src/agent/error_investigator.py:560-890`

The `ErrorInvestigatorAgent` class is defined but the actual `investigate_error()` function (line 892) uses the LangGraph-based `run_investigation()` from `investigation_graph.py` instead. Need to verify the class is truly unused before removing. The class's tools (`search_searxng`, `fetch_documentation`, etc.) ARE used by `search_executor_node` (imported at line 3 of search_executor.py), so those must be preserved.

---

## Phase 6: UX Improvements for Non-Technical Users

### 6.1 Surface error investigation results in the Gradio UI
**Problem:** This is THE most critical UX issue. When training fails, the Error Investigator runs, produces actionable recommendations... and prints them to the terminal that users can't see.

**Fix - 3 parts:**

**Part A: Make investigation return results (not just print)**
`src/flows/runner.py` - `_investigate_training_error()`:
- Change from printing to collecting results
- Return the investigation text so callers can use it

**Part B: Include investigation in error HTML**
`src/app.py` - `train_tabular_model()` and `train_llm_model()` error handlers:
- Call the investigation function
- Include the recommendation in the error HTML displayed to the user
- Format it in a user-friendly expandable section

**Part C: Make investigation results accessible from investigation_graph**
`src/agent/investigation_graph.py` - `run_investigation()` already returns a dict with `recommendation` and `confidence_level`. This data needs to flow back through runner.py to app.py.

### 6.2 Use existing error formatting utilities
**File:** `src/utils/error_handling.py`

This file already has `format_exception_for_user()` which maps technical exceptions to user-friendly messages. It's imported in app.py but only used in the import - the actual error handlers in `train_tabular_model()` and `train_llm_model()` build their own HTML instead.

**Fix:** Use `format_exception_for_user()` consistently in error handlers to provide the "What went wrong (simple)" explanation, then include the investigation results as the "Technical details" section.

---

## Execution Order

1. **Phase 1** (Bug Fixes) - Fix all broken code first, so everything can at least run
2. **Phase 2** (Missing UI) - Add the LLM Inference tab
3. **Phase 5** (Cleanup) - Remove dead code to simplify the codebase before adding new code
4. **Phase 3** (Langfuse) - Add full observability
5. **Phase 4** (Metaflow) - Ensure pipeline coverage and error surfacing
6. **Phase 6** (UX) - Surface investigation results to users

---

## Verification

### After each phase:

**Phase 1 (Bug Fixes):**
- Run `pytest tests/ test_*.py` - existing tests should still pass
- `grep -r "192.168.1.185" src/` should return nothing
- `grep -r "gpt-oss-120b" src/` should return nothing
- `grep -r "192.168.1.36" src/` should return nothing (moved to config)

**Phase 2 (Missing UI):**
- Launch `python -m src.app` and verify 5 tabs appear: Tabular ML, LLM Fine-tuning, Inference Playground, LLM Inference, RAGAS Evaluation

**Phase 3 (Langfuse):**
- Start Langfuse (`docker-compose up langfuse-web`)
- Trigger any training job or dataset analysis
- Verify traces appear in Langfuse dashboard at http://localhost:3000
- Traces should show: trace name, input/output, model used, duration

**Phase 4 (Metaflow):**
- Trigger a training job and verify Metaflow cards are generated
- Check metaflow dashboard at http://localhost:3001

**Phase 5 (Cleanup):**
- `grep -r "\[DEBUG\]" src/app.py` should return nothing
- Code should still pass all tests

**Phase 6 (UX):**
- Upload an invalid CSV file -> verify error explanation appears in web UI (not just terminal)
- Upload a valid dataset with wrong training method -> verify the recommendation appears
- Trigger a training error -> verify investigation results appear in the error HTML

---

## Critical Files to Modify

| File | Phase | Changes |
|------|-------|---------|
| `src/config.py` | 1.1 | Fix LLM URL, add model name and searxng URL |
| `src/app.py` | 2.1, 5.1, 6.1 | Add LLM tab, remove debug prints, surface errors |
| `src/utils/langfuse_client.py` | 3.1 | Complete rewrite with proper tracing |
| `src/agent/error_investigator.py` | 1.2, 1.3, 3.2 | Fix config, add tracing |
| `src/agent/sub_agents/search_executor.py` | 1.6 | Fix len() bug |
| `src/agent/sub_agents/query_generator.py` | 1.2, 3.2 | Fix model name, add tracing |
| `src/agent/sub_agents/search_analyzer.py` | 1.2, 3.2 | Fix model name, add tracing |
| `src/agent/sub_agents/synthesizer.py` | 1.2, 3.2 | Fix model name, add tracing |
| `src/agent/dataset_analyzer.py` | 1.2, 3.5, 5.5 | Fix model, fix Langfuse, dedupe report |
| `src/agent/model_selector.py` | 1.2, 5.4 | Use config settings |
| `src/llm/local_inference.py` | 1.5, 1.7 | Fix streaming, fix PeftModel import |
| `src/llm/inference_server.py` | 1.4, 3.2 | Use config, add tracing |
| `src/flows/runner.py` | 3.4, 4.2, 6.1 | Add tracing, return investigation results |
| `src/flows/llm_training_flow.py` | 1.8, 1.9 | Fix save_model, fix GRPO detection |
| `src/retrieval/qdrant_client.py` | 3.3 | Add retrieval tracing |
| `src/retrieval/hybrid_retriever.py` | 3.3 | Add retrieval tracing |
