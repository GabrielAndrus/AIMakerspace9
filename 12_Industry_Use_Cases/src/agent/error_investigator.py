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

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from src.config import settings


SEARXNG_URL = "http://192.168.1.36:4000"

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


class ErrorInvestigatorAgent:
    """LLM-based agent that investigates errors and provides recommendations."""

    def __init__(self, model_name: str = "openai/gpt-oss-120b", verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        self.model = None
        self.llm_available = False
        self._setup_model()

    def _setup_model(self):
        try:
            base_url = settings.LLM_INFERENCE_URL
            api_key = settings.LLM_INFERENCE_KEY
            
            self.model = init_chat_model(
                self.model_name,
                model_provider="openai",
                config={
                    "base_url": base_url,
                    "api_key": api_key,
                    "temperature": 0.2,
                },
            )
            self.llm_available = True
            if self.verbose:
                print(f"[ErrorInvestigator] LLM initialized: {self.model_name}")
        except Exception as e:
            self.llm_available = False
            if self.verbose:
                print(f"[ErrorInvestigator] LLM not available: {e}")

    def _print(self, message: str):
        """Print message if verbose is enabled."""
        if self.verbose:
            print(message)

    def investigate(self, ctx: ErrorContext) -> Generator[str, None, None]:
        """Investigate an error and yield recommendations.
        
        Yields:
            Strings containing investigation progress and final recommendation.
        """
        self._print("\n" + "=" * 60)
        self._print("🔍 ERROR INVESTIGATOR - Analyzing Error")
        self._print("=" * 60)
        
        # Step 1: Summarize the error
        yield "\n📋 ERROR SUMMARY\n"
        yield f"  Type: {ctx.error_type}"
        yield f"  Message: {ctx.error_message[:200]}"
        
        if ctx.task_type:
            yield f"  Task: {ctx.task_type}"
        if ctx.flow_name:
            yield f"  Flow: {ctx.flow_name}"
        if ctx.step_name:
            yield f"  Failed Step: {ctx.step_name}"
        if ctx.run_id:
            yield f"  Run ID: {ctx.run_id}"
        if ctx.training_method:
            yield f"  Training Method: {ctx.training_method}"
        if ctx.base_model:
            yield f"  Base Model: {ctx.base_model}"
        if ctx.data_path:
            yield f"  Data Path: {ctx.data_path}"
        if ctx.model_path:
            yield f"  Model Path: {ctx.model_path}"
            
        yield "\n📌 KEY TRACEBACK INFO"
        key_info = self._extract_key_info(ctx)
        yield f"  {key_info}"
        
        # Step 2: Use LLM to generate search queries and analyze
        if not self.llm_available:
            yield "\n⚠️ LLM not available - using fallback investigation"
            yield from self._fallback_investigation(ctx)
            return
        
        yield "\n🔎 SEARCHING FOR SOLUTIONS..."
        
        # Build context for the LLM
        context = self._build_error_context(ctx)
        
        # Generate search query using LLM
        search_query = self._generate_search_query_with_llm(ctx)
        yield f"  Generated query: {search_query}"
        
        # Step 3: Use search tool
        search_results = search_searxng.invoke(search_query)
        yield f"\n  Found relevant results"
        
        # Step 4: Fetch most helpful URLs
        yield "\n📖 FETCHING HELPFUL DOCUMENTATION..."
        
        urls_to_fetch = self._extract_urls(search_results)
        
        fetched_contents = []
        for url in urls_to_fetch[:2]:
            yield f"  Fetching: {url[:60]}..."
            content = fetch_documentation.invoke(url)
            if content and len(content) > 100:
                fetched_contents.append({"url": url, "content": content[:3000]})
        
        # Step 5: Use LLM to analyze and generate recommendation
        yield "\n🤔 ANALYZING ERROR AND GENERATING RECOMMENDATION..."
        
        recommendation = self._generate_recommendation_with_llm(
            ctx, search_results, fetched_contents
        )
        
        yield "\n💡 RECOMMENDATION\n"
        yield recommendation
        
        yield "\n" + "=" * 60
        yield "✅ Investigation complete"
        yield "=" * 60 + "\n"

    def _extract_key_info(self, ctx: ErrorContext) -> str:
        """Extract key information from traceback for context."""
        lines = ctx.traceback.split("\n")
        
        relevant_lines = []
        for line in lines[-10:]:
            if line.strip() and not line.startswith("Traceback"):
                relevant_lines.append(line.strip())
        
        return " | ".join(relevant_lines[:3])

    def _build_error_context(self, ctx: ErrorContext) -> str:
        """Build a detailed context string for the LLM."""
        context = f"""
ERROR TYPE: {ctx.error_type}
ERROR MESSAGE: {ctx.error_message}

TASK TYPE: {ctx.task_type or 'unknown'}
FLOW NAME: {ctx.flow_name or 'N/A'}
"""
        
        if ctx.flow_args:
            context += f"\nFLOW ARGUMENTS:\n"
            for k, v in ctx.flow_args.items():
                context += f"  {k}: {v}\n"
        
        context += f"\nTRACEBACK:\n{ctx.traceback}"
        
        return context

    def _generate_search_query_with_llm(self, ctx: ErrorContext) -> str:
        """Use LLM to generate a good search query."""
        prompt = f"""Generate a concise search query to find solutions for this error.

{SYSTEM_CONTEXT}

Error Type: {ctx.error_type}
Error Message: {ctx.error_message}
Task Type: {ctx.task_type or 'unknown'}
Training Method: {ctx.training_method or 'N/A'}
Base Model: {ctx.base_model or 'N/A'}

Provide ONLY the search query, nothing else. Make it specific and include key error terms.
"""
        try:
            response = self.model.invoke(prompt)
            query = response.content.strip()
            # Limit query length
            if len(query) > 150:
                query = query[:150]
            return query
        except Exception as e:
            # Fallback to simple query
            return f"{ctx.error_type} {ctx.error_message} {ctx.task_type or ''}"

    def _extract_urls(self, search_results: str) -> list[str]:
        """Extract URLs from search results."""
        import re
        urls = re.findall(r'https?://[^\s\)]+', search_results)
        # Clean up URLs
        cleaned = []
        for url in urls:
            url = url.rstrip('.,;:)')
            if url not in cleaned and len(url) > 20:
                cleaned.append(url)
        return cleaned[:3]

    def _generate_recommendation_with_llm(
        self, 
        ctx: ErrorContext, 
        search_results: str, 
        fetched_contents: list[dict]
    ) -> str:
        """Use LLM to analyze and generate a recommendation."""
        
        prompt = f"""You are an expert ML engineer helping debug errors on a specific hardware setup.
Analyze this error and provide a clear, actionable recommendation.

{SYSTEM_CONTEXT}

## ERROR CONTEXT
Error Type: {ctx.error_type}
Error Message: {ctx.error_message}
Task Type: {ctx.task_type or 'unknown'}
Training Method: {ctx.training_method or 'N/A'}
Base Model: {ctx.base_model or 'N/A'}
Data Path: {ctx.data_path or 'N/A'}
Model Path: {ctx.model_path or 'N/A'}

## SEARCH RESULTS
{search_results}

"""
        
        if fetched_contents:
            prompt += "## FETCHED DOCUMENTATION\n\n"
            for item in fetched_contents:
                prompt += f"From: {item['url']}\n\n"
                prompt += f"Content:\n{item['content'][:1500]}\n\n"
        
        prompt += """
## YOUR TASK
Provide a clear, actionable recommendation that includes:
1. What likely caused this error
2. Specific steps to fix it (numbered list)
3. Any relevant configuration changes

Be specific to the ML/LLM context. Use bullet points.
"""
        
        try:
            response = self.model.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Failed to generate recommendation: {e}"

    def _fallback_investigation(self, ctx: ErrorContext) -> Generator[str, None, None]:
        """Enhanced fallback investigation using error context for smarter analysis."""
        import requests
        
        query_parts = [ctx.error_type, ctx.error_message[:100]]
        
        if ctx.task_type:
            task_query_map = {
                "ml_training": "scikit-learn auto-sklearn automl",
                "llm_training": f"{ctx.training_method or 'fine-tuning'} huggingface trainer",
                "ml_inference": "model inference prediction sklearn joblib",
                "llm_inference": "transformers model loading inference"
            }
            query_parts.append(task_query_map.get(ctx.task_type, ""))
        
        if ctx.base_model:
            query_parts.append(ctx.base_model.split("/")[-1])
        
        query = " ".join(filter(None, query_parts))
        
        domain = detect_error_domain(ctx.error_type, ctx.error_message)
        engines = ENGINE_PRESETS.get(domain, ENGINE_PRESETS["general"])
        
        yield f"\n🔍 FALLBACK INVESTIGATION"
        yield f"  Domain: {domain}"
        yield f"  Query: {query[:150]}"
        
        try:
            response = requests.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json", "engines": engines},
                timeout=30,
            )
            data = response.json()
            results = data.get("results", [])[:5]
            
            yield f"\n  Found {len(results)} results"
            for r in results:
                yield f"    - {r.get('title', 'N/A')}"
                yield f"      {r.get('url', '')}\n"
                
        except Exception as e:
            yield f"  Search failed: {e}"
        
        yield "\n💡 CONTEXT-AWARE RECOMMENDATIONS"
        
        error_lower = ctx.error_message.lower()
        
        if "cuda" in error_lower or "gpu" in error_lower:
            yield "  🔥 CUDA/GPU Error detected on DGX Spark:"
            yield "     - CUDA 13.1 installed (bitsandbytes NOT compatible)"
            yield "     - Use float16/bfloat16 instead of quantization"
            if "memory" in error_lower or "oom" in error_lower:
                yield "     - Reduce batch_size in training args"
                yield "     - Enable gradient_checkpointing=True"
                yield "     - Consider DeepSpeed ZeRO-3 for large models"
        elif "nccl" in error_lower or "distributed" in error_lower:
            yield "  🔗 NCCL/Distributed Error:"
            yield "     - Ensure docker run includes --ipc host flag"
            yield "     - Verify: docker run --runtime=nvidia --gpus all --ipc host ..."
            yield "     - Alternative: Set NCCL_SHM_DISABLE=1 environment variable"
        elif ctx.task_type == "llm_training":
            if "format" in error_lower or "jsonl" in error_lower:
                yield f"  📝 Data Format Error ({ctx.training_method}):"
                training_formats = {
                    "SFT": '{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}',
                    "DPO": '{"prompt": [...], "chosen": [...], "rejected": [...]}',
                    "GRPO": '{"prompt": "...", "ground_truth": "..."}'
                }
                yield f"     Expected format: {training_formats.get(ctx.training_method, 'Check HuggingFace docs')}"
            elif ctx.base_model and "qwen" in ctx.base_model.lower():
                yield f"  🤗 Qwen Model ({ctx.base_model}):"
                yield "     - Ensure attn_implementation='flash_attention_2' if available"
                yield "     - Check torch_dtype=torch.bfloat16 compatibility"
            elif "loss" in error_lower or "nan" in error_lower:
                yield "  📉 Training Loss Error:"
                yield "     - Reduce learning_rate (try 1e-5 to 5e-5)"
                yield "     - Enable gradient clipping: max_grad_norm=1.0"
                yield "     - Check data for NaN/Inf values"
        elif ctx.task_type == "ml_training":
            yield "  📊 ML Training Error:"
            yield "     - Verify data_path exists and is readable"
            yield "     - Check target_column matches a column in the dataset"
            if "metric" in error_lower:
                yield "     - Validate metric name (accuracy, f1, r2, etc.)"
        elif "not found" in error_lower or "no such file" in error_lower:
            yield "  📁 File Not Found Error:"
            yield "     - Verify file paths are correct"
            yield "     - Check file permissions (ls -la)"
            if ctx.data_path:
                yield f"     - Data path specified: {ctx.data_path}"
        else:
            yield "  ⚠️ General troubleshooting:"
            yield "     - Check error message for specific hints"
            yield "     - Verify environment dependencies"
            yield "     - Review hardware compatibility (CUDA 13.1)"


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
