# Error Investigation Data Flow

This document traces the exact data flow of the LLM-based error investigation system from the codebase.

---

## Overview

When an error occurs during training or inference, the system:
1. Catches the exception and captures the traceback
2. Gathers context (task type, flow args, model paths)
3. **Uses an LLM agent** to:
   - Generate search queries
   - Search searxng using a tool
   - Fetch relevant documentation using a tool
   - Analyze results and generate recommendations
4. Prints the LLM's analysis and recommendations to the terminal

---

## LLM Configuration

The error investigator uses:
- **LLM**: `openai/gpt-oss-120b` (configurable)
- **Endpoint**: `http://192.168.1.185:8080/v1` (from `settings.LLM_INFERENCE_URL`)
- **API Key**: `not-needed` (from `settings.LLM_INFERENCE_KEY`)

---

## Tools Used by the Agent

### 1. `search_searxng` Tool
- **Purpose**: Search for solutions using searxng
- **Engines**: github, stackoverflow, huggingface, reddit, duckduckgo
- **Implementation**: Direct `requests.get()` to searxng API

### 2. `fetch_documentation` Tool
- **Purpose**: Fetch and extract content from URLs
- **Implementation**: Direct `requests.get()` + HTML stripping with regex

---

## 1. ML Training Error Flow (runner.py)

```
User clicks "Train Model" in Gradio
        ↓
train_tabular_model() in app.py calls run_ml_training_flow()
        ↓
run_ml_training_flow() line 40-57:
  - Metaflow Runner executes MLTrainingFlow
  - If flow fails: running.status == "failed"
        ↓
  Line 87: tb_str = f"MLTrainingFlow failed: {error_details}"
        ↓
  Line 88-94: _investigate_training_error() called with:
    - error = RuntimeError(error_details)
    - tb_str = "MLTrainingFlow failed: {error_details}"
    - task_type = "ml_training"
    - flow_name = "MLTrainingFlow"
    - flow_args = {"data_path": "...", "target_column": "..."}
        ↓
  _investigate_training_error() line 33-43:
    - Extracts data_path, training_method, base_model from flow_args
    - Calls investigate_error() passing:
      - error, tb_str, task_type, flow_name, flow_args
      - data_path=flow_args.get("data_path")
      - training_method=None (not in ML flow_args)
      - base_model=None (not in ML flow_args)
        ↓
  investigate_error() creates ErrorContext → ErrorInvestigatorAgent.investigate()
        ↓
  ErrorInvestigatorAgent:
    1. Initializes LLM via init_chat_model()
    2. Calls LLM to generate search query
    3. Invokes search_searxng tool
    4. Invokes fetch_documentation tool on top results
    5. Calls LLM to analyze and generate recommendation
        ↓
  Prints LLM-generated investigation to TERMINAL
```

---

## 2. LLM Training Error Flow (runner.py)

```
User clicks "Start Training" in LLM Fine-tuning tab
        ↓
train_llm_model() in app.py calls run_llm_training_flow()
        ↓
run_llm_training_flow() line 176-186:
  - Metaflow Runner executes LLMTrainingFlow
  - If flow fails: running.status == "failed"
        ↓
  Line 177: tb_str = f"LLMTrainingFlow failed: {error_details}"
        ↓
  Line 178-185: _investigate_training_error() called with:
    - error = RuntimeError(error_details)
    - tb_str = "LLMTrainingFlow failed: {error_details}"
    - task_type = "llm_training"
    - flow_name = "LLMTrainingFlow"
    - flow_args = {"data_path": "...", "training_method": "SFT", "base_model": "Qwen/...", "epochs": 3, ...}
        ↓
  Same flow as above → LLM agent investigates → prints to TERMINAL
```

---

## 3. Inference Error Flow (app.py)

For `load_model_for_inference()`, `predict_with_model()`, `load_lora_adapter()`, `generate_with_lora()`, etc.:

```
User loads model or runs inference in Playground
        ↓
Exception caught in try/except block
        ↓
Line 959 (example for load_model_for_inference):
  tb_str = traceback.format_exc()
        ↓
Line 960: _investigate_inference_error() called with:
  - error = e (the exception)
  - tb_str = traceback string
  - task_type = "ml_inference" or "llm_inference"
  - model_path = path to model file
        ↓
_investigate_inference_error() line 37-43:
  - Calls investigate_error() with:
    - error, tb_str, task_type
    - model_path (passed explicitly)
        ↓
  Same flow as above → LLM agent investigates → prints to TERMINAL
```

---

## 4. LLM Agent Investigation Processing (error_investigator.py)

```
investigate_error() receives parameters:
  - error: Exception
  - traceback_str: str
  - task_type: "ml_training" | "llm_training" | "ml_inference" | "llm_inference"
  - flow_name: str | None
  - flow_args: dict | None
  - training_method: str | None (SFT/DPO/GRPO)
  - base_model: str | None
  - data_path: str | None
  - model_path: str | None
        ↓
Creates ErrorContext dataclass
        ↓
ErrorInvestigatorAgent.investigate() processes:
  
  Step 1: Initialize LLM
    - init_chat_model() with settings.LLM_INFERENCE_URL
    - If fails, set llm_available = False (fallback mode)
  
  Step 2: Generate search query with LLM
    - Prompt LLM with error context
    - Get search query like "CUDA out of memory Qwen2.5 fine-tuning"
  
  Step 3: Invoke search_searxng tool
    - requests.get(SEARXNG_URL + "/search?q={query}&engines=...")
    - Returns formatted results with titles, URLs, snippets
  
  Step 4: Extract URLs from search results
    - Parse URLs from formatted results
    - Select top 2-3 most relevant
  
  Step 5: Invoke fetch_documentation tool
    - requests.get(url) for each URL
    - Strip HTML with regex
    - Return cleaned text content
  
  Step 6: Generate recommendation with LLM
    - Prompt LLM with:
      * Error context
      * Search results
      * Fetched documentation
    - LLM analyzes and provides:
      * Likely cause
      * Specific fix steps
      * Configuration changes
  
  Step 7: Yield results → printed to terminal
```

---

## Key Data Points Passed

| Error Source | task_type | Passed Data |
|-------------|-----------|-------------|
| `run_ml_training_flow()` | `"ml_training"` | flow_args (data_path, target_column) |
| `run_llm_training_flow()` | `"llm_training"` | flow_args (data_path, training_method, base_model, epochs, learning_rate) |
| `load_model_for_inference()` | `"ml_inference"` | model_path |
| `load_lora_adapter()` | `"llm_inference"` | model_path |
| `generate_with_lora()` | `"llm_inference"` | (no model_path passed) |
| `predict_with_model()` | `"ml_inference"` | model_path |

---

## Integration Points

### runner.py

| Function | Location | Calls |
|----------|----------|-------|
| `run_ml_training_flow()` | Line 49-120 | `_investigate_training_error()` on failure |
| `run_llm_training_flow()` | Line 123-205 | `_investigate_training_error()` on failure |

### app.py

| Function | Location | Calls |
|----------|----------|-------|
| `train_llm_model()` | Line 636-693 | `_investigate_training_error()` in except block |
| `load_model_for_inference()` | Line 432-456 | `_investigate_inference_error()` in except block |
| `predict_with_model()` | Line 458-489 | `_investigate_inference_error()` in except block |
| `load_lora_adapter()` | Line 855-909 | `_investigate_inference_error()` in except block |
| `generate_with_lora()` | Line 912-961 | `_investigate_inference_error()` in except block |
| `generate_with_lora_streaming()` | Line 964-1024 | `_investigate_inference_error()` in except block |

---

## LLM Prompt Engineering

The error investigator uses two LLM calls:

### 1. Search Query Generation
```
Generate a concise search query to find solutions for this error.

Error Type: {ctx.error_type}
Error Message: {ctx.error_message}
Task Type: {ctx.task_type or 'unknown'}
Training Method: {ctx.training_method or 'N/A'}
Base Model: {ctx.base_model or 'N/A'}

Provide ONLY the search query, nothing else.
```

### 2. Recommendation Generation
```
You are an expert ML engineer helping debug errors.

## ERROR CONTEXT
Error Type: ...
Error Message: ...
Task Type: ...
...

## SEARCH RESULTS
[Results from searxng]

## FETCHED DOCUMENTATION
[Content from top URLs]

## YOUR TASK
Provide a clear, actionable recommendation that includes:
1. What likely caused this error
2. Specific steps to fix it (numbered list)
3. Any relevant configuration changes
```

---

## Fallback Mode

If the LLM is not available (`llm_available = False`), the system falls back to:
1. Simple search query generation (non-LLM)
2. Direct search to searxng
3. Basic pattern-matching recommendations based on error type
