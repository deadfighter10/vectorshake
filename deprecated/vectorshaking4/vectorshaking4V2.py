import warnings

warnings.filterwarnings("ignore")

import random
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

logging.set_verbosity_error()  # Suppress extra transformer logs
console = Console()


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
# 1. Dataset Class for GoEmotions (Simplified)
# ------------------------------------------------------------------------------
class SentenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        # Each label is a list (e.g., [3]); we take the first element.
        self.labels = [int(label[0]) for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
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


# ------------------------------------------------------------------------------
# 2. Load GoEmotions Dataset (Simplified) for 28 Classes
# ------------------------------------------------------------------------------
dataset = load_dataset("go_emotions", "simplified")

# Use 10% of the original training split (with stratification)
all_train_texts = dataset["train"]["text"]
all_train_labels = dataset["train"]["labels"]
train_texts, _, train_labels, _ = train_test_split(
    all_train_texts, all_train_labels, test_size=0.80, random_state=42,
    stratify=[label[0] for label in all_train_labels]
)

# Use the official validation split
val_texts = dataset["validation"]["text"]
val_labels = dataset["validation"]["labels"]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = SentenceDataset(train_texts, train_labels, tokenizer)
val_dataset = SentenceDataset(val_texts, val_labels, tokenizer)

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
    """
    Applies adversarial perturbation on the [CLS] token representation.
    latent: tensor of shape [batch, hidden_size]
    """
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
        # Reconstruction head: maps a token's hidden representation back to its original space.
        self.reconstruct = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.xavier_uniform_(self.reconstruct.weight)


# ------------------------------------------------------------------------------
# 6. Improved Training Function: Latent Manifold Mixup with Adversarial Perturbations
# ------------------------------------------------------------------------------
def train_mixup_adv(model, data_loader, optimizer, device, lambda_mix, beta_a, beta_b, epsilon):
    """
    For each batch:
      - Extract the [CLS] latent from the final hidden states.
      - Compute an adversarial perturbation on these latents.
      - Convert labels to one-hot.
      - Sample a mixup coefficient alpha from a Beta distribution.
      - Shuffle the batch and interpolate the adversarial latents and one-hot labels.
      - Compute a mixup loss (via KL divergence) between the mixed prediction and the mixed labels.
      - Combine with the original classification loss.
    """
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        set_seed(42)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        latent = outputs.hidden_states[-1][:, 0, :]  # [CLS] token
        logits_orig = outputs.logits
        loss_orig = F.cross_entropy(logits_orig, labels)

        # Compute adversarial perturbation on [CLS] latent
        adv_latent = adversarial_perturbation(latent, labels, model, epsilon)

        # One-hot encode labels
        num_classes = model.config.num_labels
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

        # Sample mixup coefficient alpha from Beta distribution
        alpha = torch.distributions.Beta(beta_a, beta_b).sample((adv_latent.size(0),)).to(device)
        alpha = alpha.unsqueeze(1)

        # Shuffle batch indices
        indices = torch.randperm(adv_latent.size(0))
        adv_latent_shuffled = adv_latent[indices]
        labels_shuffled = labels_onehot[indices]

        # Mixup in latent space
        mixed_latent = alpha * adv_latent + (1 - alpha) * adv_latent_shuffled
        mixed_labels = alpha * labels_onehot + (1 - alpha) * labels_shuffled

        # Forward pass on mixed latent
        logits_mix = model.classifier(mixed_latent)
        log_probs = F.log_softmax(logits_mix, dim=-1)
        loss_mix = F.kl_div(log_probs, mixed_labels, reduction="batchmean")

        loss = loss_orig + lambda_mix * loss_mix

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
# 8. Bayesian Optimization Objective for Latent Mixup Hyperparameters
# ------------------------------------------------------------------------------
def objective(trial: Trial):
    # Suggest hyperparameters
    lambda_mix = trial.suggest_float("lambda_mix", 0.05, 3)
    beta_a = trial.suggest_float("beta_a", 0.5, 10)
    beta_b = trial.suggest_float("beta_b", 0.2, 10)
    epsilon = trial.suggest_float("epsilon", 0.001, 0.5)

    # Create model and optimizer
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=28)
    config.output_hidden_states = True
    model = BertForSequenceClassificationWithReconstruction.from_pretrained("bert-base-uncased", config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    num_epochs_opt = 5
    for epoch in range(1, num_epochs_opt + 1):
        train_loss = train_mixup_adv(model, train_loader, optimizer, device,
                                     lambda_mix=lambda_mix, beta_a=beta_a, beta_b=beta_b, epsilon=epsilon)
    _, val_acc = evaluate(model, val_loader, device)
    trial.set_user_attr("lambda_mix", lambda_mix)
    trial.set_user_attr("beta_a", beta_a)
    trial.set_user_attr("beta_b", beta_b)
    trial.set_user_attr("epsilon", epsilon)
    return val_acc


# Callback for tracking optimization process
def logging_callback(study, trial):
    console.print(f"Trial {trial.number} finished: Val Accuracy = {trial.value:.4f}, Params: {trial.params}, Best: {study.best_value:.4f}")


# ------------------------------------------------------------------------------
# 9. Main Experiment: Bayesian Optimization and Final Training with Baseline
# ------------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 30

    console.rule("[bold blue]Bayesian Optimization for Latent Manifold Mixup Hyperparameters")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=300, callbacks=[logging_callback])
    best_params = study.best_params
    best_val_acc = study.best_value
    console.print(f"\n[bold]Best Hyperparameters: {best_params} with Val Accuracy: {best_val_acc:.4f}[/bold]\n")

    console.rule("[bold blue]Final Training: Improved Model with Latent Mixup on Adversarial Perturbations")
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=28)
    config.output_hidden_states = True
    improved_model = BertForSequenceClassificationWithReconstruction.from_pretrained("bert-base-uncased", config=config)
    improved_model.to(device)
    optimizer_improved = AdamW(improved_model.parameters(), lr=2e-5)

    improved_acc_history = []
    for epoch in track(range(1, num_epochs + 1), description="Improved Training"):
        train_loss = train_mixup_adv(improved_model, train_loader, optimizer_improved, device,
                                     lambda_mix=best_params["lambda_mix"],
                                     beta_a=best_params["beta_a"],
                                     beta_b=best_params["beta_b"],
                                     epsilon=best_params["epsilon"])
        _, val_acc = evaluate(improved_model, val_loader, device)
        improved_acc_history.append(val_acc)
        console.log(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Accuracy = {val_acc:.4f}%")
    improved_final = evaluate(improved_model, val_loader, device)
    console.print(
        f"\n[bold]Final Improved Model -- Loss: {improved_final[0]:.4f}, Accuracy: {improved_final[1]:.4f}%[/bold]\n")

    # Baseline training at the end for comparison
    console.rule("[bold blue]Final Baseline Model Training")
    baseline_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)
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

    # Plot the training curves for final improved model and baseline
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), improved_acc_history, marker="o", label="Improved")
    plt.plot(range(1, num_epochs + 1), baseline_acc_history, marker="o", label="Baseline")
    plt.title("Validation Accuracy Over Epochs (GoEmotions)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.legend()
    plt.show()

    return best_params, best_val_acc


if __name__ == "__main__":
    best_params, best_val_acc = main()
