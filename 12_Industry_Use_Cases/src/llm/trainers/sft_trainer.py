from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def train_sft(
    dataset,
    base_model: str = "Qwen/Qwen2.5-0.5B",
    output_dir: str = "outputs",
    epochs: int = 3,
    learning_rate: float = 2e-4,
    use_quantization: bool = True,
) -> str:
    """
    Train a model using Supervised Fine-Tuning (SFT) with TRL.

    Args:
        dataset: Dataset in SFT format with 'messages' field containing
                 conversational data (list of dicts with 'role' and 'content').
        base_model: HuggingFace model identifier for the base model.
        output_dir: Directory to save the trained model and tokenizer.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        use_quantization: Whether to use 4-bit quantization. Set to False if
                          bitsandbytes is not available or incompatible.

    Returns:
        Path to the output directory containing the trained model.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    
    # Check for Flash Attention availability
    try:
        from flash_attn import flash_attn_func
        flash_attention = True
        print("✓ Flash Attention enabled")
    except ImportError:
        flash_attention = False
        print("⚠ Flash Attention not available, using standard attention")

    model_kwargs = {
        "device_map": "auto" if use_gpu else "cpu",
    }
    
    # Handle quantization with fallback for incompatible CUDA versions
    if use_quantization:
        try:
            import bitsandbytes
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
        except (ImportError, RuntimeError) as e:
            print(f"Warning: bitsandbytes not available ({e}), using float16 instead")
            use_quantization = False
    
    if not use_quantization:
        model_kwargs["torch_dtype"] = torch.float16
    
    # Enable Flash Attention 2 if available
    if flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=2 if use_gpu else 1,
        gradient_accumulation_steps=4 if use_gpu else 1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=use_gpu,
        fp16=False,
        use_cpu=not use_gpu,
        optim="adamw_torch",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print(f"\n{'='*60}")
    print("Training Configuration:")
    print(f"  Model: {base_model}")
    print(f"  GPU: {'✓' if use_gpu else '✗'}")
    print(f"  Flash Attention: {'✓' if flash_attention else '✗'}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {learning_rate:.2e}")
    print(f"{'='*60}\n")

    trainer.train()

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    return str(output_path)
