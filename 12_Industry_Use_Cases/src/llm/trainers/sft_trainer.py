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

    Returns:
        Path to the output directory containing the trained model.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
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
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    trainer.train()

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    return str(output_path)
