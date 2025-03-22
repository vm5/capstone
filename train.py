from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, GPT2LMHeadModel, TrainerCallback, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# Debug: Print initial message
print("✅ Starting the fine-tuning script...")

# Load and split the dataset
try:
    print("✅ Loading dataset...")
    dataset = load_dataset("json", data_files="data.json")
    train_test = dataset["train"].train_test_split(test_size=0.2, seed=42)
    print(f"✅ Dataset loaded successfully. Training examples: {len(train_test['train'])}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# Load a larger model
try:
    print("✅ Loading model and tokenizer...")
    model_name = "distilgpt2"  # Changed from large to medium for memory efficiency
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add padding token and configure model settings
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    print("✅ Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or tokenizer: {e}")
    exit(1)

def preprocess_function(examples):
    """Enhanced preprocessing with better context handling"""
    full_texts = []
    for input_text, output_text in zip(examples["input_text"], examples["output_text"]):
        # Split output into MCQ and analytical parts
        mcq_part = output_text.split("Analytical Questions:")[0].strip()
        analytical_part = output_text.split("Analytical Questions:")[1].strip()
        
        formatted_text = f"""Context: {input_text}

Technical Multiple Choice Question:
{mcq_part}

Key Points:
- Question directly relates to technical concepts
- Each option has specific implementation details
- Correct answer includes technical justification
- Distractors are plausible but technically flawed

Analytical Questions:
{analytical_part}

END
"""
        full_texts.append(formatted_text)
    
    return tokenizer(
        full_texts,
        truncation=True,
        max_length=1024,
        return_special_tokens_mask=True
    )

# Preprocess the dataset
try:
    print("✅ Preprocessing dataset...")
    tokenized_train = train_test["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=train_test["train"].column_names
    )
    tokenized_val = train_test["test"].map(
        preprocess_function,
        batched=True,
        remove_columns=train_test["test"].column_names
    )
    print("✅ Dataset preprocessed successfully.")
except Exception as e:
    print(f"❌ Error preprocessing dataset: {e}")
    exit(1)

# Custom callback
class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            print(f"Step {state.global_step}: Loss = {logs.get('loss', 'N/A')}")

# Training arguments with better parameters
training_args = TrainingArguments(
    output_dir="./fine_tuned_model2025",
    num_train_epochs=20,  # Increased for better learning
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=3e-5,  # Slightly increased
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine_with_restarts",
    save_strategy="steps",
    save_steps=20,
    evaluation_strategy="steps",
    eval_steps=20,
    logging_steps=5,
    load_best_model_at_end=True,
    save_total_limit=3,
    fp16=False,
    no_cuda=True,
    seed=42,
    remove_unused_columns=False,
    prediction_loss_only=True,
    label_smoothing_factor=0.05,  # Reduced for more precise learning
    max_grad_norm=1.0
)

# Improved data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    return_tensors="pt",
    pad_to_multiple_of=8
)

# Initialize the Trainer
try:
    print("✅ Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=[CustomCallback()]
    )
    print("✅ Trainer initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing Trainer: {e}")
    exit(1)

# Train the model
try:
    print("✅ Starting training...")
    trainer.train()
    print("✅ Training completed successfully.")
except Exception as e:
    print(f"❌ Error during training: {e}")
    exit(1)

# Save the fine-tuned model
try:
    print("✅ Saving model...")
    trainer.save_model("./fine_tuned_model2025")
    tokenizer.save_pretrained("./fine_tuned_model2025")
    print("✅ Model saved successfully.")
except Exception as e:
    print(f"❌ Error saving model: {e}")
    exit(1)