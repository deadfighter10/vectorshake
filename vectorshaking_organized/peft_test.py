import datetime
import random

import numpy as np
from datasets import load_dataset, DownloadConfig, Dataset, Features, Value, ClassLabel # Import necessary types
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertConfig
from peft import get_peft_model, LoraConfig, TaskType
import torch
# Assuming 'rich' library is used for [bold] markup, else remove or replace
try:
    from rich import print
except ImportError:
    pass # Use standard print if rich is not installed

def robust_load_dataset(dataset_name):
    download_config = DownloadConfig(
        num_proc=2,
        max_retries=5,
        resume_download=True
    )

    try:
        return load_dataset(
            dataset_name,
            download_config=download_config
        )
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        raise


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class Config:
    SEED = 42
    NUM_EPOCHS = 30 # Tuned down for quicker example, adjust as needed
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    OPTIMIZATION_TRIALS = 500 # Not used in this snippet
    LOG_FILE = f"logs/training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json" # Not used in this snippet
    RESULTS_FILE = "final_results.json" # Not used in this snippet
    OPTIMIZATION_FILE = "best_params.json" # Not used in this snippet
    STUIES_FILE = "studies.json" # Not used in this snippet
    DATASET_NAME = "ag_news"


    DATASET_CONFIGS = {
        "ag_news": {
            "name": "AG News",
            "loader": lambda: robust_load_dataset("ag_news"),
            "train_split": "train",
            "test_split": "test",
            "text_key": "text",
            "label_key": "label",
            "train_percent": 0.01 # Using 1% for AG News as per original code
        },
        "trec": {
            "name": "TREC-6",
            "loader": lambda: load_dataset("trec",  trust_remote_code=True),
            "train_split": "train",
            "test_split": "test",
            "text_key": "text",
            "label_key": "coarse_label",
            "train_percent": 1 # Using 100% for TREC
        },
        "semeval": {
            "name": "SemEval-2018",
            "loader": lambda: load_dataset("sem_eval_2018_task_1", "subtask5.english", trust_remote_code=True),
            "train_split": "train",
            "test_split": "test",
            "text_key": "Tweet",
            # Handle label processing directly in load_data for semeval
            "label_key": "label", # We will create this column
            "train_percent": 1 # Using 100% for SemEval
        }
    }

# --- Configuration ---
model_name = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(model_name) # Load tokenizer once

def load_data():
    config = Config.DATASET_CONFIGS[Config.DATASET_NAME]
    dataset = config["loader"]()

    train_ds_raw = dataset[config["train_split"]]
    test_ds_raw = dataset[config["test_split"]]

    # Special handling for SemEval labels
    if Config.DATASET_NAME == "semeval":
        emotion_cols = ["anger", "anticipation", "disgust", "fear", "joy",
                        "love", "optimism", "pessimism", "sadness", "surprise", "trust"]
        def get_label(example):
            labels = [example[e] for e in emotion_cols]
            return np.argmax(labels) # Return index directly

        train_ds_raw = train_ds_raw.map(lambda ex: {"label": get_label(ex)})
        test_ds_raw = test_ds_raw.map(lambda ex: {"label": get_label(ex)})
        # Create label names list based on the order
        label_names = emotion_cols
    else:
        # For other datasets, get unique labels to create names
        all_labels = train_ds_raw[config["label_key"]] + test_ds_raw[config["label_key"]]
        unique_labels = sorted(list(set(all_labels)))
        # If labels are integers, create string names, otherwise use them directly
        if isinstance(unique_labels[0], int):
             label_names = [str(lbl) for lbl in unique_labels]
        else:
             label_names = unique_labels # Assuming they are already strings


    # Define Features including ClassLabel
    features = Features({
        config["text_key"]: Value('string'),
        config["label_key"]: ClassLabel(names=label_names) # Use ClassLabel
    })

    # Apply features (especially ClassLabel which converts labels to integers)
    # Need to handle potential renaming if label_key is not 'label'
    if config["label_key"] != 'label':
        train_ds_raw = train_ds_raw.rename_column(config["label_key"], "label")
        test_ds_raw = test_ds_raw.rename_column(config["label_key"], "label")

    train_ds_raw = train_ds_raw.cast(features)
    test_ds_raw = test_ds_raw.cast(features)


    # Subsampling (if train_percent < 100)
    if config["train_percent"] < 100:
        # Use dataset's train_test_split for stratified sampling
        train_ds_subset = train_ds_raw.train_test_split(
            test_size=(100 - config["train_percent"]) / 100.0,
            seed=Config.SEED,
            stratify_by_column="label" # Use the standardized 'label' column
        )['train'] # Keep only the 'train' part of the split
        print(f"[bold]Subsampled train data:[/bold] {len(train_ds_subset)} examples")
    else:
        train_ds_subset = train_ds_raw
        print(f"[bold]Using full train data:[/bold] {len(train_ds_subset)} examples")


    print(f"[bold]Loaded {config['name']} dataset[/]\nTrain: {len(train_ds_subset)} | Test: {len(test_ds_raw)}")
    # num_labels is derived from the features
    num_labels = train_ds_subset.features['label'].num_classes
    print(f"Number of labels: {num_labels}")
    print(f"Label names: {train_ds_subset.features['label'].names}")

    return train_ds_subset, test_ds_raw, num_labels


# --- Tokenization Function ---
def preprocess_function(examples):
    # The keys in `examples` are the column names ('text', 'label' or 'Tweet', 'label')
    text_key = Config.DATASET_CONFIGS[Config.DATASET_NAME]["text_key"]
    # Tokenize the texts
    batch = tokenizer(examples[text_key], padding="max_length", truncation=True, max_length=Config.MAX_LENGTH)
    # The labels are already integers thanks to ClassLabel, just pass them through.
    # The map function handles batching, so 'label' will be a list of integers.
    batch['label'] = examples['label']
    return batch

# --- Data Loading and Preprocessing ---
train_dataset_raw, test_dataset_raw, num_labels = load_data()

# Apply tokenization using map (batched for efficiency)
train_dataset = train_dataset_raw.map(preprocess_function, batched=True)
test_dataset = test_dataset_raw.map(preprocess_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# --- Create DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE) # No shuffle for test/eval


# --- Define LoRA Config ---
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, # Sequence Classification
    inference_mode=False,
    r=8,                      # Rank of the update matrices
    lora_alpha=16,            # Scaling factor
    lora_dropout=0.1,         # Dropout probability
    target_modules=["query", "value"] # Target modules
)

print(f"LoRA Config: {peft_config}")

# --- Model Loading ---
config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
model = BertForSequenceClassification.from_pretrained(model_name, config=config)

# --- Apply LoRA to the model ---
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.to(device)


# --- Optimizer ---
optimizer = AdamW(model.parameters(), lr=6e-3) # Use parameters() for PEFT model

# --- Evaluation Function ---
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            # Move batch to device - already done by set_format? Double check needed.
            # Safest to move explicitly if unsure.
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label # Pass labels to compute loss during eval if needed
            )
            loss = outputs.loss
            if loss is not None: # Handle cases where labels might not be passed or model doesn't return loss
                 total_loss += loss.item()
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels.extend(label.cpu().numpy()) # Use the label moved to device

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# --- Training Loop ---
print(f"Starting training on {device}...")
set_seed(Config.SEED)
for epoch in range(Config.NUM_EPOCHS):
    model.train()
    total_train_loss = 0
    batch_count = 0
    for batch in train_loader:
        # Data is already on the correct device due to set_format or move explicitly
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        batch_count += 1
        # Optional: Print batch loss less frequently
        # if batch_count % 10 == 0:
        #     print(f"Epoch {epoch+1}, Batch {batch_count}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} Avg Train Loss: {avg_train_loss:.4f}")

    # --- Evaluation ---
    metrics = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1} Val Metrics: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

print("Training finished.")

# Final evaluation after training
final_metrics = evaluate(model, test_loader, device)
print("\n--- Final Test Metrics ---")
for key, value in final_metrics.items():
    print(f"{key.capitalize()}: {value:.4f}")