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
from sklearn.metrics import accuracy_score
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

# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
USE_OPTIMIZATION = True  # Set to False to load hyperparameters from JSON
OPTIMIZATION_FILE = "../../best_params.json"
NUM_EPOCHS = 20  # Use 20 epochs everywhere


# ------------------------------------------------------------------------------
# 0. Fix Random Seeds for Reproducibility
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
# 1. Load and Process SemEval-2018 Task 1 (Subtask5.english) Dataset
# ------------------------------------------------------------------------------
dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english", trust_remote_code=True)

# The dataset contains columns: 'ID', 'Tweet', and one column per emotion.
emotion_list = ["anger", "anticipation", "disgust", "fear", "joy",
                "love", "optimism", "pessimism", "sadness", "surprise", "trust"]


def process_examples(split):
    texts = []
    labels = []
    for ex in dataset[split]:
        texts.append(ex["Tweet"])
        scores = [ex[emotion] for emotion in emotion_list]
        label = emotion_list[np.argmax(scores)]
        labels.append(label)
    return texts, labels


train_texts_all, train_labels_all = process_examples("train")
test_texts, test_labels = process_examples("test")

# Build a mapping from unique labels from both splits.
unique_labels = sorted(list(set(train_labels_all + test_labels)))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
num_labels = len(label2id)
console.print(f"[bold]Found {num_labels} unique labels in SemEval-2018 Task 1.[/bold]")

# Stratified sampling: use 10% of the training set.
train_texts_sample, _, train_labels_sample, _ = train_test_split(
    train_texts_all, train_labels_all, test_size=0.70, random_state=42,
    stratify=train_labels_all
)


# ------------------------------------------------------------------------------
# 2. Dataset Class for SemEval-2018
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
# 4. Adversarial Perturbation Function (applied to the [CLS] token)
# ------------------------------------------------------------------------------
def adversarial_perturbation(latent, labels, model, epsilon):
    latent = latent.detach().clone().requires_grad_()
    logits = model.classifier(latent)
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, latent, retain_graph=True)[0]
    delta = epsilon * grad.sign()
    latent_adv = latent + delta
    return latent_adv.detach()


# ------------------------------------------------------------------------------
# 5. Custom Model with Reconstruction Head
# ------------------------------------------------------------------------------
class BertForSequenceClassificationWithReconstruction(BertForSequenceClassification):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.reconstruct = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.xavier_uniform_(self.reconstruct.weight)


# ------------------------------------------------------------------------------
# 6. Improved Training Function: Combined Adversarial Perturbation, Mixup & Contrastive Loss
# ------------------------------------------------------------------------------
def train_improved(model, data_loader, optimizer, device, lambda_mix, lambda_contrast, beta_a, beta_b, epsilon):
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        set_seed(42)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        latent = outputs.hidden_states[-1][:, 0, :]
        logits_orig = outputs.logits
        loss_ce = F.cross_entropy(logits_orig, labels)

        # Generate two adversarial views
        adv_view1 = adversarial_perturbation(latent, labels, model, epsilon)
        adv_view2 = adversarial_perturbation(latent, labels, model, epsilon)

        # Contrastive loss: 1 - cosine similarity per sample.
        cos_sim = F.cosine_similarity(adv_view1, adv_view2, dim=1)
        loss_contrast = torch.mean(1 - cos_sim)

        # Latent manifold mixup on adv_view1
        num_classes = model.config.num_labels
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        mix_alpha = torch.distributions.Beta(beta_a, beta_b).sample((adv_view1.size(0),)).to(device)
        mix_alpha = mix_alpha.unsqueeze(1)
        indices = torch.randperm(adv_view1.size(0))
        adv_view1_shuffled = adv_view1[indices]
        labels_shuffled = labels_onehot[indices]
        mixed_latent = mix_alpha * adv_view1 + (1 - mix_alpha) * adv_view1_shuffled
        mixed_labels = mix_alpha * labels_onehot + (1 - mix_alpha) * labels_shuffled

        logits_mix = model.classifier(mixed_latent)
        log_probs = F.log_softmax(logits_mix, dim=-1)
        loss_mix = F.kl_div(log_probs, mixed_labels, reduction="batchmean")

        loss = loss_ce + lambda_mix * loss_mix + lambda_contrast * loss_contrast
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


# ------------------------------------------------------------------------------
# 7. Evaluation Function
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
    return avg_loss, accuracy


# ------------------------------------------------------------------------------
# 8. Bayesian Optimization Objective for Hyperparameters
# ------------------------------------------------------------------------------
def objective(trial: Trial):
    lambda_mix = trial.suggest_float("lambda_mix", 0.1, 2.0)
    lambda_contrast = trial.suggest_float("lambda_contrast", 0.1, 1.0)
    beta_a = trial.suggest_float("beta_a", 0.2, 1.0)
    beta_b = trial.suggest_float("beta_b", 0.2, 1.0)
    epsilon = trial.suggest_float("epsilon", 0.005, 0.05)

    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
    config.output_hidden_states = True
    model = BertForSequenceClassificationWithReconstruction.from_pretrained("bert-base-uncased", config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    num_epochs_opt = 20
    for epoch in range(1, num_epochs_opt + 1):
        train_loss = train_improved(model, train_loader, optimizer, device,
                                    lambda_mix=lambda_mix, lambda_contrast=lambda_contrast,
                                    beta_a=beta_a, beta_b=beta_b, epsilon=epsilon)
    _, val_acc = evaluate(model, val_loader, device)
    trial.set_user_attr("lambda_mix", lambda_mix)
    trial.set_user_attr("lambda_contrast", lambda_contrast)
    trial.set_user_attr("beta_a", beta_a)
    trial.set_user_attr("beta_b", beta_b)
    trial.set_user_attr("epsilon", epsilon)
    return val_acc


def logging_callback(study, trial):
    console.print(f"Trial {trial.number} finished: Val Accuracy = {trial.value:.4f}, Params: {trial.params}")


# ------------------------------------------------------------------------------
# 9. Main Experiment: Switchable Bayesian Optimization and Final Training with Baseline
# ------------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 20

    # Baseline reference accuracy (assumed value)
    baseline_reference = 52.9900
    console.rule("[bold blue]Baseline Reference")
    console.print(f"[bold]Baseline Accuracy: {baseline_reference:.4f}%[/bold]\n")

    # Switch for Bayesian Optimization
    USE_OPTIMIZATION = False

    if USE_OPTIMIZATION:
        console.rule("[bold blue]Bayesian Optimization for Latent Manifold Mixup Hyperparameters")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1, callbacks=[logging_callback])
        best_params = study.best_params
        best_val_acc = study.best_value
        console.print(f"\n[bold]Best Hyperparameters: {best_params} with Val Accuracy: {best_val_acc:.4f}[/bold]\n")
        # Save best hyperparameters to a JSON file
        with open("../../best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
    else:
        # Load hyperparameters from a JSON file
        with open("../../best_params.json", "r") as f:
            best_params = json.load(f)
        best_val_acc = None
        console.print(f"\n[bold]Loaded Hyperparameters: {best_params}[/bold]\n")

    console.rule("[bold blue]Final Training: Improved Model with Latent Mixup on Adversarial Perturbations")
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
    config.output_hidden_states = True
    improved_model = BertForSequenceClassificationWithReconstruction.from_pretrained("bert-base-uncased", config=config)
    improved_model.to(device)
    optimizer_improved = AdamW(improved_model.parameters(), lr=2e-5)

    improved_acc_history = []
    for epoch in track(range(1, num_epochs + 1), description="Improved Training"):
        train_loss = train_improved(improved_model, train_loader, optimizer_improved, device,
                                    lambda_mix=best_params["lambda_mix"],
                                    lambda_contrast=best_params["lambda_contrast"],
                                    beta_a=best_params["beta_a"],
                                    beta_b=best_params["beta_b"],
                                    epsilon=best_params["epsilon"])
        _, val_acc = evaluate(improved_model, val_loader, device)
        improved_acc_history.append(val_acc)
        console.log(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Accuracy = {val_acc:.4f}%")
    improved_final = evaluate(improved_model, val_loader, device)
    console.print(
        f"\n[bold]Final Improved Model -- Loss: {improved_final[0]:.4f}, Accuracy: {improved_final[1]:.4f}%[/bold]\n")

    console.rule("[bold blue]Final Baseline Model Training")
    baseline_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    baseline_model.to(device)
    optimizer_baseline = AdamW(baseline_model.parameters(), lr=2e-5)
    baseline_acc_history = []
    for epoch in track(range(1, num_epochs + 1), description="Baseline Training"):
        train_loss = train_baseline(baseline_model, train_loader, optimizer_baseline, device)
        _, val_acc = evaluate(baseline_model, val_loader, device)
        baseline_acc_history.append(val_acc)
        console.log(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Accuracy = {val_acc:.4f}%")
    baseline_final = evaluate(baseline_model, val_loader, device)
    console.print(
        f"\n[bold]Final Baseline Model -- Loss: {baseline_final[0]:.4f}, Accuracy: {baseline_final[1]:.4f}%[/bold]\n")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), improved_acc_history, marker="o", label="Improved")
    plt.plot(range(1, num_epochs + 1), baseline_acc_history, marker="o", label="Baseline")
    plt.title("Validation Accuracy Over Epochs (SemEval-2018 Task 1)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.legend()
    plt.show()

    return best_params, best_val_acc


if __name__ == "__main__":
    import json

    best_params, best_val_acc = main()
