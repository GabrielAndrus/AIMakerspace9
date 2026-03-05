from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def train_dpo(
    dataset,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "outputs",
    beta: float = 0.1,
    epochs: int = 1,
    use_quantization: bool = True,
) -> str:
    """
    Train a model using Direct Preference Optimization (DPO) with TRL.

    Args:
        dataset: Dataset in DPO format with 'prompt', 'chosen', and 'rejected'
                 fields. Each field contains conversational data as a list of
                 dicts with 'role' and 'content'.
        base_model: HuggingFace model identifier for the base model.
                    Should be an instruct-tuned model for DPO.
        output_dir: Directory to save the trained model and tokenizer.
        beta: KL penalty coefficient controlling deviation from reference model.
        epochs: Number of training epochs.
        use_quantization: Whether to use 4-bit quantization. Set to False if
                          bitsandbytes is not available or incompatible.

    Returns:
        Path to the output directory containing the trained model.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_kwargs = {"device_map": "auto"}
    
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

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
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

    training_args = DPOConfig(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        beta=beta,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    return str(output_path)
