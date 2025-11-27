from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, GPT2LMHeadModel, TrainerCallback, DataCollatorForLanguageModeling, pipeline
from datasets import Dataset
import torch
from torch.nn import CrossEntropyLoss
import os
import json
import psutil
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure directories exist
os.makedirs('summaries', exist_ok=True)
os.makedirs('C:/Users/Admin/Music/dashboard/dashboard-app/ml/fine_tuned_distilgpt2', exist_ok=True)

# Utility functions
def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# File processing functions
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def extract_text_from_image(file_path):
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Error reading image {file_path}: {e}")
        return ""

def extract_text_from_file(file_path):
    if file_path.endswith(('.pdf', '.pptx')):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(('.png', '.jpg', '.jpeg')):
        return extract_text_from_image(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def generate_summary(text):
    if not text.strip():
        return ""
    
    try:
        summarizer = pipeline(
            "summarization", 
            model="sshleifer/distilbart-cnn-12-6",
            device=0 if torch.cuda.is_available() else -1
        )
        summary = summarizer(
            text,
            max_length=min(100, len(text.split())),
            min_length=30,
            do_sample=False,
            truncation=True
        )
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Summarization error: {e}")
        return text[:300] + "..." if len(text) > 300 else text

def store_summary(summary, file_name):
    try:
        with open(f"summaries/{os.path.basename(file_name)}.txt", "w", encoding="utf-8") as f:
            f.write(summary)
    except Exception as e:
        print(f"Error saving summary for {file_name}: {e}")

def process_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return []
    
    processed_data = []
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    print(f"\nProcessing {len(files)} files...")
    for filename in tqdm(files, desc="Processing files"):
        file_path = os.path.join(folder_path, filename)
        
        # Skip existing data.json to avoid duplicates
        if filename == "data.json":
            continue
            
        try:
            # Handle different file types
            if filename.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        processed_data.extend(file_data)
                    elif isinstance(file_data, dict):
                        processed_data.append(file_data)
            else:
                text = extract_text_from_file(file_path)
                if text:
                    summary = generate_summary(text)
                    if summary:
                        processed_data.append({
                            "input_text": text[:1000],  # Take first 1000 chars as input
                            "output_text": summary
                        })
                        store_summary(summary, filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return processed_data

# Dataset functions
def validate_dataset(dataset):
    print("Validating dataset structure...")
    for idx, example in enumerate(dataset):
        if not isinstance(example['input_text'], str) or not isinstance(example['output_text'], str):
            print(f"❌ Invalid data format at index {idx}")
            return False
        if len(example['input_text']) == 0 or len(example['output_text']) == 0:
            print(f"❌ Empty text at index {idx}")
            return False
    print("✅ Dataset validation successful")
    return True

# Training components
class TrainingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("\n🚀 Training started!")
        print(f"Training on {len(train_test['train'])} examples")
        print(f"Validating on {len(train_test['test'])} examples")
        print_memory_usage()
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            if hasattr(state, 'log_history') and state.log_history:
                current_loss = state.log_history[-1].get('loss', 'N/A')
                if isinstance(current_loss, (int, float)):
                    print(f"Step {state.global_step}: Loss = {current_loss:.4f}")
                else:
                    print(f"Step {state.global_step}: Loss = {current_loss}")
            
    def on_epoch_end(self, args, state, control, **kwargs):
        if hasattr(state, 'log_history') and len(state.log_history) >= 2:
            epoch = state.epoch
            train_loss = state.log_history[-2].get('loss', 'N/A')
            eval_loss = state.log_history[-1].get('eval_loss', 'N/A')
            
            print(f"\nEpoch {epoch:.2f} results:")
            if isinstance(train_loss, (int, float)):
                print(f"Training loss: {train_loss:.4f}")
            else:
                print(f"Training loss: {train_loss}")
                
            if isinstance(eval_loss, (int, float)):
                print(f"Validation loss: {eval_loss:.4f}")
            else:
                print(f"Validation loss: {eval_loss}")
            
            print_memory_usage()

def preprocess_function(examples):
    texts = [
        f"Input: {inp}\nOutput: {out}\n<|endoftext|>"
        for inp, out in zip(examples["input_text"], examples["output_text"])
    ]
    return tokenizer(
        texts,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

# Main execution
if __name__ == "__main__":
    print("✅ Starting the fine-tuning script...")
    print_memory_usage()

    # Process files and create dataset
    dataset_folder = "C:/Users/Admin/Music/dashboard/dashboard-app/ml/Datasets"
    
    try:
        print("✅ Processing all files and creating dataset...")
        processed_data = process_files_in_folder(dataset_folder)
        
        # Filter valid data
        valid_data = [
            item for item in processed_data 
            if isinstance(item.get('input_text', ''), str) and 
               isinstance(item.get('output_text', ''), str) and
               len(item.get('input_text', '')) > 0 and
               len(item.get('output_text', '')) > 0
        ]
        
        if not valid_data:
            raise ValueError("No valid data found after processing files")
        
        # Save the combined dataset
        output_path = os.path.join(dataset_folder, "data.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(valid_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created dataset with {len(valid_data)} examples")
        
        # Create the dataset for training
        dataset = Dataset.from_dict({
            "input_text": [item["input_text"] for item in valid_data],
            "output_text": [item["output_text"] for item in valid_data]
        })
        
        train_test = dataset.train_test_split(test_size=0.2, seed=42)
        print(f"✅ Dataset split into {len(train_test['train'])} training and {len(train_test['test'])} validation examples")

    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        exit(1)

    # Load model
    try:
        print("✅ Loading model and tokenizer...")
        model_path = "C:/Users/Admin/Music/dashboard/dashboard-app/ml/fine_tuned_distilgpt2"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        try:
            print("⌛ Loading existing fine-tuned model...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
            print("✅ Successfully loaded existing fine-tuned model")
        except Exception as e:
            print(f"⚠️ Could not load fine-tuned model: {e}")
            print("⌛ Loading base model instead...")
            tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
            print("✅ Loaded base model successfully")
        
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        
        print("✅ Model and tokenizer configured successfully")
        print(f"Model device: {next(model.parameters()).device}")

    except Exception as e:
        print(f"❌ Error loading model or tokenizer: {e}")
        exit(1)

    # Preprocess data
    try:
        print("🔄 Preprocessing data...")
        train_test = train_test.map(
            preprocess_function,
            batched=True,
            remove_columns=["input_text", "output_text"],
            desc="Processing dataset"
        )
        
        print("✅ Preprocessing completed")

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        exit(1)

    # Training setup
    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=5,
        per_device_train_batch_size=1,  # Reduced batch size
        per_device_eval_batch_size=1,   # Reduced batch size
        gradient_accumulation_steps=4,   # Increased gradient accumulation
        learning_rate=1e-5,             # Reduced learning rate
        weight_decay=0.01,
        warmup_steps=50,                # Reduced warmup steps
        logging_steps=1,                # Log every step
        save_strategy="epoch",
        eval_strategy="epoch",          # Fixed parameter name
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,                     # Disabled fp16 since using CPU
        report_to="none",
        remove_unused_columns=False
    )

    # Add dataset size check
    if len(train_test["train"]) < 1 or len(train_test["test"]) < 1:
        print("❌ Dataset is too small. Please ensure you have enough training data.")
        exit(1)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize and run trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test["train"],
        eval_dataset=train_test["test"],
        data_collator=data_collator,
        callbacks=[TrainingCallback()]
    )

    print("\nStarting training...")
    try:
        train_result = trainer.train()
        
        # Try to save with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"\nSaving model (attempt {attempt + 1}/{max_retries})...")
                # Force garbage collection before saving
                import gc
                gc.collect()
                
                # Save to a temporary directory first
                temp_dir = os.path.join(os.path.dirname(model_path), "temp_model")
                os.makedirs(temp_dir, exist_ok=True)
                trainer.save_model(temp_dir)
                
                # Move files from temp to final location
                import shutil
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
                shutil.move(temp_dir, model_path)
                print("✅ Model saved successfully!")
                break
            except Exception as save_error:
                print(f"Warning: Save attempt {attempt + 1} failed: {save_error}")
                if attempt == max_retries - 1:
                    print("❌ Failed to save model after all attempts")
                else:
                    print("Retrying after a short delay...")
                    import time
                    time.sleep(5)  # Wait 5 seconds before retry
        
        # Save metrics
        metrics = train_result.metrics
        try:
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
            
            print("\n✅ Training completed successfully!")
            print(f"Final training loss: {metrics['train_loss']:.4f}")
            print(f"Final evaluation loss: {eval_metrics['eval_loss']:.4f}")
        except Exception as metric_error:
            print(f"\n⚠️ Warning: Could not save metrics: {metric_error}")
            print("But the training itself completed successfully!")
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        exit(1)