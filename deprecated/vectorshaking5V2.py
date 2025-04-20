import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For deterministic CuBLAS behavior

import warnings

warnings.filterwarnings("ignore")

import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, logging
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset

import optuna
from optuna import Trial
from rich.console import Console
from rich.progress import track

import nltk

nltk.download("punkt")

logging.set_verbosity_error()  # Suppress extra transformer logs
console = Console()

# Enable deterministic algorithms in PyTorch
torch.use_deterministic_algorithms(True)

# ------------------------------------------------------------------------------
# Base Variables and Configurations
# ------------------------------------------------------------------------------
NUM_EPOCHS = 40  # Number of epochs for both optimization and final training
USE_OPTIMIZATION = False  # Set to False to load hyperparameters from JSON
OPTIMIZATION_FILE = "best_params.json"

# Dataset configuration dictionary allows easy swapping between datasets.
DATASET_CONFIGS = {
    "semeval": {
        "name": "SemEval-2018 Task 1 (Subtask5.english)",
        "loader": lambda: load_dataset("sem_eval_2018_task_1", "subtask5.english", trust_remote_code=True),
        "train_percent": 10
    },
    "go_emotions": {
        "name": "GoEmotions",
        "loader": lambda: load_dataset("go_emotions", "simplified", trust_remote_code=True),
        "train_percent": 10
    }
}
# Choose dataset here:
DATASET_NAME = "go_emotions"  # Change to "go_emotions" to switch


# ------------------------------------------------------------------------------
# Deterministic Settings and Random Seed
# ------------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ------------------------------------------------------------------------------
# 1. Load and Process Dataset (SemEval-2018 Task 1)
# ------------------------------------------------------------------------------
def process_semeval(split, dataset):
    texts = []
    labels = []
    emotion_list = ["anger", "anticipation", "disgust", "fear", "joy",
                    "love", "optimism", "pessimism", "sadness", "surprise", "trust"]
    for ex in dataset[split]:
        texts.append(ex["Tweet"])
        scores = [ex[emotion] for emotion in emotion_list]
        label = emotion_list[np.argmax(scores)]
        labels.append(label)
    return texts, labels


ds_config = DATASET_CONFIGS[DATASET_NAME]
dataset = ds_config["loader"]()
if DATASET_NAME == "semeval":
    train_texts_all, train_labels_all = process_semeval("train", dataset)
    test_texts, test_labels = process_semeval("test", dataset)
elif DATASET_NAME == "go_emotions":
    train_texts_all = dataset["train"]["text"]
    train_labels_all = dataset["train"]["labels"]
    test_texts = dataset["validation"]["text"]
    test_labels = dataset["validation"]["labels"]
    test_labels = [int(label[0]) for label in test_labels]
    train_labels_all = [int(label[0]) for label in train_labels_all]
else:
    raise ValueError("Invalid dataset name specified.")

# Build mapping from unique labels (from both splits) to integers.
unique_labels = sorted(list(set(train_labels_all + test_labels)))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
num_labels = len(label2id)
console.print(f"[bold]Found {num_labels} unique labels in {ds_config['name']}.[/bold]")

# Stratified sampling: use the specified percentage of the training set.
train_texts_sample, _, train_labels_sample, _ = train_test_split(
    train_texts_all, train_labels_all,
    test_size=(100 - ds_config["train_percent"]) / 100.0,
    random_state=42, stratify=train_labels_all
)

dataset_info = {
    "dataset_name": ds_config["name"],
    "train_percent": ds_config["train_percent"],
    "total_train_samples": len(train_texts_all),
    "used_train_samples": len(train_texts_sample),
    "total_test_samples": len(test_texts)
}


# ------------------------------------------------------------------------------
# 2. Dataset Class
# ------------------------------------------------------------------------------
class SentenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_length=512):
        self.texts = texts
        self.labels = labels  # labels as strings
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_str = self.labels[idx]
        label = self.label2id[label_str]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = SentenceDataset(train_texts_sample, train_labels_sample, tokenizer, label2id, max_length=512)
val_dataset = SentenceDataset(test_texts, test_labels, tokenizer, label2id, max_length=512)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# ------------------------------------------------------------------------------
# 3. Baseline Model Training Function (Standard Fine-Tuning)
# ------------------------------------------------------------------------------
def train_baseline(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        set_seed(42)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


# ------------------------------------------------------------------------------
# 4. FGSM Adversarial Perturbation Function
# ------------------------------------------------------------------------------
def fgsm_adversarial_perturbation(latent, labels, model, epsilon):
    latent = latent.detach().clone().requires_grad_(True)
    logits = model.classifier(latent)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    delta = epsilon * latent.grad.sign()
    return (latent + delta).detach()


# ------------------------------------------------------------------------------
# 5. Custom Model with Reconstruction Head
# ------------------------------------------------------------------------------
class BertForSequenceClassificationWithReconstruction(BertForSequenceClassification):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.reconstruct = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.xavier_uniform_(self.reconstruct.weight)


# ------------------------------------------------------------------------------
# 6. Improved Training Function: Combined FGSM, Mixup, Contrastive & Consistency Loss
# ------------------------------------------------------------------------------
def train_improved(model, data_loader, optimizer, device, lambda_mix, lambda_contrast, lambda_consistency, beta_a,
                   beta_b, epsilon):
    model.train()
    total_loss = 0.0
    # (Do not reset seed every batch to allow natural stochasticity.)
    for batch in data_loader:
        # Removed: set_seed(42)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        # Clean forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        clean_latent = outputs.hidden_states[-1][:, 0, :]  # Clean [CLS] latent
        clean_logits = outputs.logits
        loss_ce = F.cross_entropy(clean_logits, labels)

        # Detach clean predictions to serve as the target anchor
        clean_probs = F.softmax(clean_logits.detach(), dim=-1)

        # Generate two adversarial views using FGSM (FGSM takes one gradient step)
        adv_view1 = fgsm_adversarial_perturbation(clean_latent, labels, model, epsilon)
        adv_view2 = fgsm_adversarial_perturbation(clean_latent, labels, model, epsilon)

        # Contrastive loss: compare each adversarial view to the clean latent.
        cos_sim1 = F.cosine_similarity(adv_view1, clean_latent, dim=1)
        cos_sim2 = F.cosine_similarity(adv_view2, clean_latent, dim=1)
        loss_contrast = (torch.mean(1 - cos_sim1) + torch.mean(1 - cos_sim2)) / 2

        # Latent manifold mixup on adv_view1:
        num_classes = model.config.num_labels
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        mix_alpha = torch.distributions.Beta(beta_a, beta_b).sample((adv_view1.size(0),)).to(device)
        mix_alpha = mix_alpha.unsqueeze(1)
        indices = torch.randperm(adv_view1.size(0))
        adv_view1_shuffled = adv_view1[indices]
        # We mix the latent representations
        mixed_latent = mix_alpha * adv_view1 + (1 - mix_alpha) * adv_view1_shuffled
        logits_mix = model.classifier(mixed_latent)
        loss_mix = F.kl_div(F.log_softmax(logits_mix, dim=-1), clean_probs, reduction="batchmean")

        # Consistency loss: compare the prediction on adv_view1 to the clean prediction.
        logits_adv = model.classifier(adv_view1)
        loss_consistency = F.kl_div(F.log_softmax(logits_adv, dim=-1), clean_probs, reduction="batchmean")

        # Combine all loss terms
        total_batch_loss = loss_ce + lambda_mix * loss_mix + lambda_contrast * loss_contrast + lambda_consistency * loss_consistency

        total_batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += total_batch_loss.item()
    return total_loss / len(data_loader)


# ------------------------------------------------------------------------------
# 7. Evaluation Function (with Additional Metrics)
# ------------------------------------------------------------------------------
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, accuracy, precision, recall, f1


# ------------------------------------------------------------------------------
# 8. Bayesian Optimization Objective for Hyperparameters
# ------------------------------------------------------------------------------
def objective(trial: Trial):
    set_seed(42)

    lambda_mix = trial.suggest_float("lambda_mix", 0.5, 5.0)
    lambda_contrast = trial.suggest_float("lambda_contrast", 0.3, 5.0)
    lambda_consistency = trial.suggest_float("lambda_consistency", 0.2, 5.0)
    beta_a = trial.suggest_float("beta_a", 0.3, 5.0)
    beta_b = trial.suggest_float("beta_b", 0.5, 6.0)
    epsilon = trial.suggest_float("epsilon", 0.005, 0.1)
    lr = trial.suggest_float("lr", 1e-5, 9e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)

    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
    config.output_hidden_states = True
    model = BertForSequenceClassificationWithReconstruction.from_pretrained("bert-base-uncased", config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_epochs_opt = 20
    for epoch in range(1, num_epochs_opt + 1):
        train_loss = train_improved(model, train_loader, optimizer, device,
                                    lambda_mix=lambda_mix, lambda_contrast=lambda_contrast,
                                    lambda_consistency=lambda_consistency,
                                    beta_a=beta_a, beta_b=beta_b, epsilon=epsilon)
    _, val_acc, _, _, _ = evaluate(model, val_loader, device)
    #trial.set_user_attr("lambda_mix", lambda_mix)
    trial.set_user_attr("lambda_contrast", lambda_contrast)
    trial.set_user_attr("lambda_consistency", lambda_consistency)
    trial.set_user_attr("beta_a", beta_a)
    trial.set_user_attr("beta_b", beta_b)
    trial.set_user_attr("epsilon", epsilon)
    trial.set_user_attr("lr", lr)
    trial.set_user_attr("weight_decay", weight_decay)
    return val_acc


def logging_callback(study, trial):
    console.print(f"Trial {trial.number} finished: Val Accuracy = {trial.value:.4f}, Params: {trial.params}")


# ------------------------------------------------------------------------------
# 9. Main Experiment: Switchable Bayesian Optimization and Final Training with Baseline
# ------------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = NUM_EPOCHS

    # Save experiment configuration details
    experiment_info = {
        "dataset": DATASET_CONFIGS[DATASET_NAME]["name"],
        "train_percent": DATASET_CONFIGS[DATASET_NAME]["train_percent"],
        "total_train_samples": len(train_texts_all),
        "used_train_samples": len(train_texts_sample),
        "total_test_samples": len(test_texts)
    }

    # Train Baseline Model first
    console.rule("[bold blue]Final Baseline Model Training")
    baseline_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    baseline_model.to(device)
    # For baseline, we use lr and weight_decay from the Bayesian study if available;
    # if not, we set default values.
    default_lr = 2e-5
    default_weight_decay = 0.0
    if USE_OPTIMIZATION:
        # Run a quick baseline training with default values
        optimizer_baseline = AdamW(baseline_model.parameters(), lr=default_lr, weight_decay=default_weight_decay)
    else:
        optimizer_baseline = AdamW(baseline_model.parameters(), lr=default_lr, weight_decay=default_weight_decay)
    baseline_acc_history = []
    for epoch in track(range(1, num_epochs + 1), description="Baseline Training"):
        train_loss = train_baseline(baseline_model, train_loader, optimizer_baseline, device)
        _, val_acc, _, _, _ = evaluate(baseline_model, val_loader, device)
        baseline_acc_history.append(val_acc)
        console.log(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Accuracy = {val_acc:.4f}%")
    base_loss, base_acc, base_prec, base_rec, base_f1 = evaluate(baseline_model, val_loader, device)
    console.print(
        f"\n[bold]Final Baseline Model -- Loss: {base_loss:.4f}, Accuracy: {base_acc:.4f}, Precision: {base_prec:.4f}, Recall: {base_rec:.4f}, F1: {base_f1:.4f}[/bold]\n")

    # Now run Bayesian Optimization
    if USE_OPTIMIZATION:
        console.rule("[bold blue]Bayesian Optimization for Hyperparameters")
        storage_name = "sqlite:///optuna_study_5V2.db"
        study = optuna.create_study(direction="maximize", storage=storage_name, study_name="GoEmotions_Test1",
                                    load_if_exists=True)
        study.optimize(objective, n_trials=50, callbacks=[logging_callback])
        best_params = study.best_params
        best_val_acc = study.best_value
        console.print(f"\n[bold]Best Hyperparameters: {best_params} with Val Accuracy: {best_val_acc:.4f}[/bold]\n")
        with open(OPTIMIZATION_FILE, "w") as f:
            json.dump(best_params, f, indent=4)
    else:
        with open(OPTIMIZATION_FILE, "r") as f:
            best_params = json.load(f)
        best_val_acc = None
        console.print(f"\n[bold]Loaded Hyperparameters: {best_params}[/bold]\n")

    console.rule("[bold blue]Final Training: Improved Model with FGSM, Mixup, Contrastive & Consistency Loss")
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
    config.output_hidden_states = True
    improved_model = BertForSequenceClassificationWithReconstruction.from_pretrained("bert-base-uncased", config=config)
    improved_model.to(device)
    optimizer_improved = AdamW(improved_model.parameters(), lr=best_params.get("lr", default_lr),
                               weight_decay=best_params.get("weight_decay", default_weight_decay))

    improved_acc_history = []
    for epoch in track(range(1, num_epochs + 1), description="Improved Training"):
        train_loss = train_improved(improved_model, train_loader, optimizer_improved, device,
                                    lambda_mix=best_params["lambda_mix"],
                                    lambda_contrast=best_params["lambda_contrast"],
                                    lambda_consistency=best_params["lambda_consistency"],
                                    beta_a=best_params["beta_a"],
                                    beta_b=best_params["beta_b"],
                                    epsilon=best_params["epsilon"])
        _, val_acc, _, _, _ = evaluate(improved_model, val_loader, device)
        improved_acc_history.append(val_acc)
        console.log(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Accuracy = {val_acc:.4f}%")
    imp_loss, imp_acc, imp_prec, imp_rec, imp_f1 = evaluate(improved_model, val_loader, device)
    console.print(
        f"\n[bold]Final Improved Model -- Loss: {imp_loss:.4f}, Accuracy: {imp_acc:.4f}, Precision: {imp_prec:.4f}, Recall: {imp_rec:.4f}, F1: {imp_f1:.4f}[/bold]\n")

    # Save final experiment results in a JSON file.
    final_results = {
        "best_hyperparameters": best_params,
        "improved_model": {
            "loss": imp_loss,
            "accuracy": imp_acc,
            "precision": imp_prec,
            "recall": imp_rec,
            "f1": imp_f1
        },
        "baseline_model": {
            "loss": base_loss,
            "accuracy": base_acc,
            "precision": base_prec,
            "recall": base_rec,
            "f1": base_f1
        },
        "dataset_info": dataset_info,
        "num_epochs": NUM_EPOCHS,
        "optimizer": {
            "lr": best_params.get("lr", default_lr),
            "weight_decay": best_params.get("weight_decay", default_weight_decay)
        }
    }
    with open("final_results.json", "w") as f:
        json.dump(final_results, f, indent=4)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), improved_acc_history, marker="o", label="Improved")
    plt.plot(range(1, num_epochs + 1), baseline_acc_history, marker="o", label="Baseline")
    plt.title("Validation Accuracy Over Epochs (" + DATASET_CONFIGS[DATASET_NAME]["name"] + ")")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.legend()
    plt.show()

    return best_params, best_val_acc


if __name__ == "__main__":
    num_labels = len(unique_labels)  # Set dynamically based on dataset
    best_params, best_val_acc = main()
