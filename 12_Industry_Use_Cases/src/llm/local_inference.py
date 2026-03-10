"""Local LLM Inference with LoRA Adapter Support.

This module enables users to test their trained LoRA adapters by loading
them locally with the base model and running inference.
"""

from pathlib import Path
from typing import Generator, Optional

import torch

from src.utils.langfuse_client import langfuse_trace, trace_llm_call


class LocalLLMInference:
    """Load and run inference on locally-trained LoRA adapters."""

    def __init__(
        self,
        max_seq_length: int = 2048,
        load_in_4bit: bool = False,
    ):
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.base_model_name: Optional[str] = None

    def load_trained_lora(
        self,
        lora_path: str,
        base_model: Optional[str] = None,
    ) -> bool:
        """Load a trained LoRA adapter with its base model.

        Args:
            lora_path: Path to the saved LoRA adapter directory
            base_model: Base model name (auto-detected from metadata if not provided)

        Returns:
            True if loading successful

        Raises:
            FileNotFoundError: If LoRA adapter files missing
            ValueError: If base model cannot be determined
        """
        lora_path = Path(lora_path)

        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA adapter directory not found: {lora_path}")

        required_files = ["adapter_config.json"]
        missing = [f for f in required_files if not (lora_path / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"LoRA adapter incomplete. Missing: {missing}\n"
                "Expected files in adapter directory:\n"
                "  - adapter_config.json (PEFT config)\n"
                "  - adapter_model.safetensors or adapter_model.bin (weights)"
            )

        if base_model is None:
            base_model = self._detect_base_model(lora_path)

        if not base_model:
            raise ValueError(
                f"Could not determine base model for LoRA at {lora_path}.\n"
                "Please provide base_model parameter or ensure metadata.json "
                "contains 'base_model' field."
            )

        self.base_model_name = base_model

        with langfuse_trace(
            "load_lora_adapter",
            input={"lora_path": str(lora_path), "base_model": base_model},
        ) as trace:
            self.model, self.tokenizer = self._load_with_unsloth(base_model)

            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(lora_path))

            if trace:
                trace.update(output={"status": "loaded", "base_model": base_model, "lora_path": str(lora_path)})

        return True

    def _detect_base_model(self, lora_path: Path) -> Optional[str]:
        """Auto-detect base model from adapter metadata."""
        import json

        metadata_path = lora_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                return metadata.get("base_model")

        adapter_config_path = lora_path / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path) as f:
                config = json.load(f)
                return config.get("base_model_name_or_path")

        return None

    def _load_with_unsloth(self, model_name: str):
        """Load base model using Unsloth for efficient inference."""
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=self.max_seq_length,
                dtype=torch.float16 if not self.load_in_4bit else None,
                load_in_4bit=self.load_in_4bit,
            )

            FastLanguageModel.for_inference(model)

            return model, tokenizer
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
            }

            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            return model, tokenizer

    def _prepare_inputs(self, prompt: str, system_prompt: Optional[str], messages: list) -> dict:
        """Tokenize and prepare model inputs as a dict with input_ids on the correct device."""
        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                # apply_chat_template may return a tensor or a dict
                if isinstance(input_ids, dict):
                    return {k: v.to(self.model.device) for k, v in input_ids.items()}
                return {"input_ids": input_ids.to(self.model.device)}
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                return {k: v.to(self.model.device) for k, v in inputs.items()}
        except Exception:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            return {k: v.to(self.model.device) for k, v in inputs.items()}

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate response using loaded LoRA adapter.

        Args:
            prompt: User input
            system_prompt: System instruction (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Generated response text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded. Call load_trained_lora() first.")

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        inputs = self._prepare_inputs(prompt, system_prompt, messages)

        with langfuse_trace(
            "local_lora_inference",
            input={"prompt": prompt, "system_prompt": system_prompt, "model": self.base_model_name,
                   "max_new_tokens": max_new_tokens, "temperature": temperature},
        ) as trace:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            trace_llm_call(trace, "lora_generate", prompt, response, self.base_model_name or "unknown")
            if trace:
                trace.update(output={"response": response, "tokens_generated": len(generated_tokens)})

        return response

    def generate_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Generate response with streaming (for real-time UI updates).

        Args:
            prompt: User input
            system_prompt: System instruction (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Generated text chunks as they are produced
        """
        from threading import Thread

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded. Call load_trained_lora() first.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        inputs = self._prepare_inputs(prompt, system_prompt, messages)

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        with langfuse_trace(
            "local_lora_inference_streaming",
            input={"prompt": prompt, "system_prompt": system_prompt, "model": self.base_model_name,
                   "max_new_tokens": max_new_tokens, "temperature": temperature},
        ) as trace:
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            full_response = []
            for text in streamer:
                full_response.append(text)
                yield text

            thread.join()

            response_text = "".join(full_response)
            trace_llm_call(trace, "lora_generate_streaming", prompt, response_text, self.base_model_name or "unknown")
            if trace:
                trace.update(output={"response": response_text, "chunks": len(full_response)})


def merge_lora_to_base(
    lora_path: str,
    output_dir: str,
    base_model: Optional[str] = None,
) -> str:
    """Merge LoRA adapter into base model and save as standalone.

    Use this for deployment to systems that don't support PEFT/LoRA loading,
    or when you want a single model file for easier distribution.

    Args:
        lora_path: Path to LoRA adapter
        output_dir: Where to save merged model
        base_model: Base model name (auto-detected if not provided)

    Returns:
        Path to merged model directory
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lora_path = Path(lora_path)

    if base_model is None:
        metadata_path = lora_path / "metadata.json"
        if metadata_path.exists():
            import json

            with open(metadata_path) as f:
                base_model = json.load(f).get("base_model")

    if not base_model:
        raise ValueError("Could not determine base model. Please provide base_model parameter.")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = PeftModel.from_pretrained(model, str(lora_path))
    model = model.merge_and_unload()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    return str(output_path)


def validate_lora_adapter(lora_path: str) -> dict:
    """Validate a LoRA adapter package for completeness.

    Returns:
        {
            "valid": bool,
            "issues": list[str],
            "warnings": list[str],
            "base_model": str | None,
        }
    """
    import json

    path = Path(lora_path)
    result = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "base_model": None,
    }

    if not path.exists():
        result["valid"] = False
        result["issues"].append(f"Directory not found: {lora_path}")
        return result

    required = ["adapter_config.json"]

    has_weights = (path / "adapter_model.safetensors").exists()
    has_bin = (path / "adapter_model.bin").exists()

    if not has_weights and not has_bin:
        required.append("adapter_model.safetensors (or adapter_model.bin)")

    for f in required:
        if not (path / f).exists():
            result["valid"] = False
            result["issues"].append(f"Missing required file: {f}")

    if (path / "adapter_config.json").exists():
        try:
            with open(path / "adapter_config.json") as f:
                config = json.load(f)

            result["base_model"] = config.get("base_model_name_or_path")

            if not result["base_model"]:
                result["warnings"].append(
                    "adapter_config.json missing base_model_name_or_path. "
                    "You'll need to specify base model manually."
                )
        except json.JSONDecodeError:
            result["valid"] = False
            result["issues"].append("adapter_config.json is not valid JSON")

    if (path / "metadata.json").exists():
        try:
            with open(path / "metadata.json") as f:
                metadata = json.load(f)

            if result["base_model"] is None:
                result["base_model"] = metadata.get("base_model")
        except json.JSONDecodeError:
            result["warnings"].append("metadata.json is not valid JSON (non-critical)")

    return result


_local_inference: Optional[LocalLLMInference] = None


def get_local_inference() -> LocalLLMInference:
    """Get or create the global local inference instance."""
    global _local_inference
    if _local_inference is None:
        _local_inference = LocalLLMInference()
    return _local_inference
