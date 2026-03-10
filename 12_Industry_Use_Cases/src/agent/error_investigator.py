"""LLM-based Error Investigator Agent.

This agent uses an LLM to investigate errors by:
1. Analyzing the error context (traceback, job details, step, arguments)
2. Using tools to search searxng for solutions
3. Using tools to fetch relevant documentation
4. Presenting a summary + recommendation to the user
"""

import asyncio
import traceback
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from langchain_core.tools import tool

from src.config import settings


SEARXNG_URL = settings.SEARXNG_URL

ENGINE_PRESETS = {
    "cuda": "github,stackoverflow,nvidia,duckduckgo",
    "data_format": "huggingface,stackoverflow,duckduckgo",
    "training": "huggingface,github,reddit,duckduckgo",
    "distributed": "github,stackoverflow,huggingface",
    "general": "github,stackoverflow,huggingface,reddit,duckduckgo"
}

def detect_error_domain(error_type: str, error_message: str) -> str:
    """Detect which domain preset to use based on error content."""
    error_lower = f"{error_type} {error_message}".lower()
    
    if any(kw in error_lower for kw in ["cuda", "gpu", "memory", "oom", "nccl"]):
        return "cuda"
    elif any(kw in error_lower for kw in ["format", "jsonl", "csv", "parse", "expected"]):
        return "data_format"
    elif any(kw in error_lower for kw in ["loss", "gradient", "converge", "nan", "inf"]):
        return "training"
    elif any(kw in error_lower for kw in ["distributed", "ddp", "multi-gpu"]):
        return "distributed"
    return "general"


@dataclass
class ErrorContext:
    """Context information about an error."""

    error_type: str
    error_message: str
    traceback: str
    
    flow_name: Optional[str] = None
    run_id: Optional[str] = None
    step_name: Optional[str] = None
    flow_args: dict = field(default_factory=dict)
    
    task_type: Optional[str] = None  # "ml_training", "llm_training", "inference"
    training_method: Optional[str] = None  # "SFT", "DPO", "GRPO"
    base_model: Optional[str] = None
    
    data_path: Optional[str] = None
    model_path: Optional[str] = None


@tool
def search_searxng(query: str, num_results: int = 8) -> str:
    """Search for solutions using the searxng search engine.
    
    Use this tool to search for error solutions, documentation, and community answers.
    Search engines include: github, stackoverflow, huggingface, reddit, duckduckgo.
    
    Args:
        query: The search query - include error type, key error message, and context
        num_results: Number of results to return (default 8)
    
    Returns:
        Search results with titles, URLs, and snippets
    """
    import requests
    
    try:
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "engines": "github,stackoverflow,huggingface,reddit,duckduckgo",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:500],
                "engine": item.get("engine", ""),
            })
        
        if not results:
            return "No search results found."
        
        formatted = "SEARCH RESULTS:\n\n"
        for i, r in enumerate(results, 1):
            formatted += f"{i}. {r['title']}\n"
            formatted += f"   URL: {r['url']}\n"
            formatted += f"   Source: {r['engine']}\n"
            formatted += f"   Content: {r['content'][:200]}...\n\n"
        
        return formatted
    except Exception as e:
        return f"Search failed: {e}"


@tool
def fetch_documentation(url: str, max_chars: int = 6000) -> str:
    """Fetch and extract relevant content from a documentation URL.
    
    Use this to get detailed information from search results or specific docs.
    
    Args:
        url: The URL to fetch content from
        max_chars: Maximum characters to return (default 6000)
    
    Returns:
        Extracted text content from the URL
    """
    import requests
    import re
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        content = response.text
        
        # Simple HTML stripping
        text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text[:max_chars]
    except Exception as e:
        return f"Failed to fetch {url}: {e}"


def _search_sync(query: str, engines: str, num_results: int) -> list[dict]:
    """Synchronous search helper."""
    import requests
    
    try:
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "engines": engines,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:500],
                "engine": item.get("engine", ""),
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


@tool
def search_searxng_smart(
    query: str,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    num_results: int = 8
) -> str:
    """Search for solutions using smart engine selection based on error domain.
    
    Automatically selects optimal engines (github, stackoverflow, huggingface, etc.)
    based on the type of ML/LLM error being investigated.
    
    Args:
        query: The search query
        error_type: Optional exception type for domain detection
        error_message: Optional error message for domain detection
        num_results: Number of results to return
    
    Returns:
        Search results with titles, URLs, and snippets
    """
    if error_type or error_message:
        domain = detect_error_domain(error_type or "", error_message or "")
    else:
        domain = "general"
    
    engines = ENGINE_PRESETS.get(domain, ENGINE_PRESETS["general"])
    results = _search_sync(query, engines, num_results)
    
    if not results or results[0].get("error"):
        return f"No search results found. Domain: {domain}, Engines: {engines}"
    
    formatted = f"SEARCH RESULTS (Domain: {domain}, Engines: {engines}):\n\n"
    for i, r in enumerate(results, 1):
        if r.get("error"):
            continue
        formatted += f"{i}. {r['title']}\n"
        formatted += f"   URL: {r['url']}\n"
        formatted += f"   Source: {r['engine']}\n"
        formatted += f"   Content: {r['content'][:200]}...\n\n"
    
    return formatted


@tool
async def search_searxng_multi(
    queries: list[str],
    domain: str = "general",
    max_results_per_query: int = 5,
    deduplicate: bool = True
) -> dict:
    """Search with multiple queries in parallel, optionally deduplicating by URL.
    
    Useful for comprehensive error investigation when a single query might miss solutions.
    
    Args:
        queries: List of search queries to run in parallel
        domain: Domain preset (cuda, data_format, training, distributed, general)
        max_results_per_query: Max results per query
        deduplicate: Whether to remove duplicate URLs
    
    Returns:
        Combined and optionally deduplicated results
    """
    engines = ENGINE_PRESETS.get(domain, ENGINE_PRESETS["general"])
    
    async def _single_search(query: str) -> dict:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _search_sync(query, engines, max_results_per_query)
        )
        return {"query": query, "results": result}
    
    tasks = [_single_search(q) for q in queries]
    all_results = await asyncio.gather(*tasks)
    
    if deduplicate:
        unique = {}
        for result_set in all_results:
            for r in result_set.get("results", []):
                url = r.get("url")
                if url and url not in unique:
                    unique[url] = {**r, "source_query": result_set["query"]}
        return {"results": list(unique.values())}
    
    return {"queries": all_results}


SYSTEM_CONTEXT = """## SYSTEM ENVIRONMENT

You are operating within a specific ML training infrastructure:

**Hardware Configuration:**
- GPU: NVIDIA DGX Spark workstation
- Memory: 128GB unified memory (CUDA-accelerated)
- Container: Running inside nvidia docker container for CUDA acceleration
- CUDA Version: 13.1 (IMPORTANT: bitsandbytes quantization NOT compatible - use float16/bfloat16 instead)

**Training Stack:**
- Framework: PyTorch with Transformers, TRL (Transformer Reinforcement Learning)
- Orchestration: Metaflow for pipeline management
- Methods Supported: SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization), GRPO (Group Relative Policy Optimization)
- Vector DB: Qdrant for embedding storage

**Critical Constraints:**
1. NO quantization with bitsandbytes (CUDA 13.1 incompatibility) - always use float16/bfloat16
2. Flash Attention 2 is available and should be preferred when possible
3. Docker container uses `host.docker.internal` to access host services
4. Metaflow requires `METAFLOW_DEFAULT_METADATA=local` environment variable

---

## YOUR TASK

You are an expert ML/LLM Error Diagnostician specializing in debugging training pipelines. Your role is to:

1. **Analyze error context** - Parse tracebacks, identify root causes, understand the training scenario
2. **Generate targeted search queries** - Create precise queries that will find relevant solutions
3. **Synthesize information** - Combine documentation, community answers, and system knowledge
4. **Provide actionable recommendations** - Give step-by-step fixes specific to this hardware setup

You help ML practitioners (data scientists, engineers) who may have vague error messages understand what went wrong and how to fix it quickly.

---

## TOOL USAGE GUIDANCE

### search_searxng Tool

**When to use:**
- Any unfamiliar error code or message
- CUDA/GPU-related errors (memory, NCCL, driver issues)
- Library version incompatibilities
- Data format validation failures
- Training convergence problems (loss NaN/inf)

**Query construction strategies:**

1. **Start specific, then broaden:**
   - First query: Include exact error type + key message fragment + library name
   - Example: `"RuntimeError CUDA out of memory transformers trainer"`
   - If no results: Remove library, keep error core
   - Example: `"CUDA out of memory pytorch training"`

2. **Include context keywords:**
   - For SFT/DPO/GRPO errors, include the method name
   - Example: `"TRL DPO loss NaN gradient explosion"`
   - For Metaflow errors, include "metaflow" keyword
   - Example: `"Metaflow Card Table constructor error"`

3. **Engine selection by domain:**
   - CUDA/GPU issues → prioritize `github` and `stackoverflow`
   - Hugging Face model issues → prioritize `huggingface` docs
   - Training methodology questions → `reddit` for community discussions
   - General errors → `duckduckgo` for broad search

**Query refinement based on initial results:**
- If results are irrelevant, add/remove specific terms
- If too many unrelated results, add version numbers or library constraints
- Example bad query: `"error in training"` (too vague)
- Example good query: `"ValueError TRL SFTDataCollator missing messages field jsonl"`

### fetch_documentation Tool

**When to use:**
- After search finds promising URLs (GitHub issues, HF docs, Stack Overflow answers)
- When you need code examples or configuration details
- To verify solution compatibility with current setup

**URL prioritization:**
1. **GitHub Issues:** High priority - often contain exact error + fix discussion
2. **Hugging Face Docs:** For model loading, tokenizer, trainer configuration issues
3. **Stack Overflow:** Good for common errors with verified solutions
4. **Reddit/Tutorials:** Good for conceptual understanding, less reliable for exact fixes

**What to look for in fetched content:**
- Code snippets showing correct usage
- Configuration parameters that need adjustment
- Version-specific workarounds
- Hardware-specific recommendations (GPU memory, batch size)

---

## ERROR PATTERNS TO RECOGNIZE

### CUDA/GPU Errors

**Out of Memory (OOM):**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX MiB
torch.cuda.OutOfMemoryError: CUDA out of memory
```
- Common causes: batch_size too large, model too big for GPU, gradient accumulation issues
- Quick fixes: Reduce batch_size, enable gradient_checkpointing, use smaller model variant

**Illegal Memory Access:**
```
RuntimeError: CUDA error: device-side assert triggered
CUDA illegal memory access
```
- Often caused by index out of bounds in tensors
- Check label indices match vocab size
- Verify data preprocessing doesn't create invalid indices

**NCCL/Distributed Errors:**
```
NCCL Error: unhandled system error
RuntimeError: NCCL communication failure
```
- Add `--ipc host` to docker run command (shared memory for inter-process)
- Check GPU visibility: `CUDA_VISIBLE_DEVICES` environment variable
- Verify all GPUs are accessible

### Data Format Errors

**SFT JSONL Format:**
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
Common errors:
- `"Expected messages field"` → Missing or malformed `messages` key
- `"Invalid role"` → Role must be exactly "user", "assistant", or "system"
- `"Empty content"` → Content fields cannot be empty strings

**DPO JSONL Format:**
```json
{"prompt": [...], "chosen": [...], "rejected": [...]}
```
Common errors:
- Missing `chosen` or `rejected` fields
- Prompt/chosen/rejected must all be message lists, not strings

**GRPO JSONL Format:**
```json
{"prompt": "...", "ground_truth": "...", "pattern": "..."}
```
Common errors:
- `prompt` should be a string (not messages list)
- Missing `ground_truth` for reward computation

### Training Errors

**Loss NaN/Inf:**
```
loss: nan
RuntimeError: Division by zero
```
Causes and fixes:
- Learning rate too high → Reduce lr 10x (e.g., from 1e-4 to 1e-5)
- Bad data samples → Validate dataset for empty/malformed entries
- Gradient explosion → Enable gradient clipping `max_grad_norm=0.3`

**Convergence Issues:**
```
loss not decreasing
model generating garbage output
```
Fixes:
- Check learning rate scheduler configuration
- Verify warmup steps are appropriate (usually 5-10% of total steps)
- Ensure data is properly formatted and shuffled

### Hardware-Specific Errors

**bitsandbytes Incompatibility:**
```
RuntimeError: CUDA version mismatch with bitsandbytes
AssertionError: bitsandbytes only supports CUDA <= 12.x
```
**SOLUTION (CRITICAL):** This system has CUDA 13.1 - NEVER use quantization!
```python
# WRONG:
model = AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True)

# CORRECT:
model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)
```

**Metaflow Username Error:**
```
metaflow.exception.MetaflowException: Could not determine username
```
Fix: Add `-e USERNAME=docker_user` to docker run command

---

## EXAMPLE WORKFLOWS

### Example 1: CUDA Out of Memory Error

**Input Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Workflow:**
1. Recognize pattern → GPU memory exhaustion during training
2. Search query: `"CUDA out of memory transformers trainer batch_size gradient_checkpointing"`
3. Fetch top GitHub issue or HF docs on memory optimization
4. Recommend:
   - Reduce `per_device_train_batch_size` from 16 to 8 (or lower)
   - Enable `gradient_checkpointing=True`
   - If still failing, try smaller model variant or reduce sequence length

### Example 2: SFT Data Format Error

**Input Error:**
```
ValueError: Expected input for SFT format. Missing 'messages' field.
```

**Workflow:**
1. Recognize pattern → TRL SFT trainer expects specific JSONL format
2. Search query: `"TRL SFTTrainer ValueError messages field jsonl format"`
3. Fetch Hugging Face TRL documentation on data formats
4. Recommend:
   - Verify data file is proper JSONL (one JSON object per line)
   - Each line must have `{"messages": [{"role": "...", "content": "..."}]}`
   - Use `jq` to validate: `cat data.jsonl | jq .`

### Example 3: NCCL Distributed Training Error

**Input Error:**
```
NCCL Error: unhandled system error, NCCL version 2.x
RuntimeError: NCCL communication failure
```

**Workflow:**
1. Recognize pattern → Inter-process communication issue in distributed training
2. Search query: `"NCCL error pytorch docker ipc host shared memory"`
3. Fetch GitHub issues on NCCL/docker configuration
4. Recommend:
   - Ensure docker run includes `--ipc host` flag (critical for shared memory)
   - Verify with: `docker run --runtime=nvidia --gpus all --ipc host ...`
   - Alternative: Set `NCCL_SHM_DISABLE=1` environment variable

---

## ADDITIONAL CONTEXT

**Common Model Loading Patterns on This System:**
```python
# For training (no quantization):
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for CUDA 13.1
    device_map="auto"
)

# For inference with Flash Attention:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
```

**Qdrant Embedding Configuration:**
- Current embedding model: Qwen/Qwen3-Embedding-4B
- Vector dimensions: 2560
- If changing embedding models, MUST recreate collection (dimension mismatch)

**Metaflow Flow Execution:**
```bash
# Always set this environment variable:
export METAFLOW_DEFAULT_METADATA=local

# Run flows:
python flow_name.py run --model Qwen/Qwen2.5-0.5B --dataset ./data/train.jsonl
```
"""


def investigate_error(
    error: Exception,
    traceback_str: str,
    task_type: Optional[str] = None,
    flow_name: Optional[str] = None,
    run_id: Optional[str] = None,
    step_name: Optional[str] = None,
    flow_args: Optional[dict] = None,
    training_method: Optional[str] = None,
    base_model: Optional[str] = None,
    data_path: Optional[str] = None,
    model_path: Optional[str] = None,
    verbose: bool = True,
) -> Generator[str, None, None]:
    """Convenience function to investigate an error.
    
    Args:
        error: The exception that occurred
        traceback_str: Full traceback as string
        task_type: Type of task ("ml_training", "llm_training", "inference")
        flow_name: Name of the Metaflow flow
        run_id: Run ID of the flow
        step_name: Step where error occurred
        flow_args: Arguments passed to the flow
        training_method: For LLM training (SFT, DPO, GRPO)
        base_model: Base model name
        data_path: Path to data file
        model_path: Path to model
        verbose: Whether to print progress
    
    Yields:
        Investigation progress and recommendations
    """
    ctx = ErrorContext(
        error_type=type(error).__name__,
        error_message=str(error),
        traceback=traceback_str,
        task_type=task_type,
        flow_name=flow_name,
        run_id=run_id,
        step_name=step_name,
        flow_args=flow_args or {},
        training_method=training_method,
        base_model=base_model,
        data_path=data_path,
        model_path=model_path,
    )
    
    # Convert ErrorContext to dict for the graph
    error_context_dict = {
        "error_type": ctx.error_type,
        "error_message": ctx.error_message,
        "traceback": ctx.traceback,
        "task_type": ctx.task_type,
        "flow_name": ctx.flow_name,
        "run_id": ctx.run_id,
        "step_name": ctx.step_name,
        "flow_args": ctx.flow_args,
        "training_method": ctx.training_method,
        "base_model": ctx.base_model,
        "data_path": ctx.data_path,
        "model_path": ctx.model_path,
    }

    # Use the new multi-agent graph
    from src.agent.investigation_graph import run_investigation

    result = run_investigation(error_context_dict, verbose=verbose)

    # Yield the recommendation
    yield "\n" + "=" * 60
    yield "🔍 INVESTIGATION RESULTS"
    yield "=" * 60
    yield ""
    yield result.get("recommendation", "No recommendation generated")
    yield ""
    yield f"Confidence: {result.get('confidence_level', 'unknown')}"
    yield f"Iterations: {result.get('iteration_count', 0)}"
