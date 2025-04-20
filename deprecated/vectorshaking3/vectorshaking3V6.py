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

from rich.console import Console
from rich.progress import track

logging.set_verbosity_error()  # Suppress extra transformer logs
console = Console()

# ------------------------------
# 0. Fix Random Seeds
# ------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------------
# 1. Dataset Class for GoEmotions (Simplified)
# ------------------------------
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

# ------------------------------
# 2. Load GoEmotions Dataset (Simplified) for 28 Classes
# ------------------------------
dataset = load_dataset("go_emotions", "simplified")

# Use 10% of the original training split (with stratification)
all_train_texts = dataset["train"]["text"]
all_train_labels = dataset["train"]["labels"]
train_texts, _, train_labels, _ = train_test_split(
    all_train_texts, all_train_labels, test_size=0.90, random_state=42,
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

# ------------------------------
# 3. Baseline Model Training Function (Standard Fine-Tuning)
# ------------------------------
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
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# ------------------------------
# 4. Adversarial Perturbation Function
# ------------------------------
def adversarial_perturbation(latent, labels, model, epsilon):
    latent = latent.detach().clone().requires_grad_()
    logits = model.classifier(latent)
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, latent, retain_graph=True)[0]
    delta = epsilon * grad.sign()
    latent_adv = latent + delta
    return latent_adv.detach()

# ------------------------------
# 5. Custom Model with Reconstruction Head
# ------------------------------
class BertForSequenceClassificationWithReconstruction(BertForSequenceClassification):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Reconstruction head: maps perturbed latent back to original latent space.
        self.reconstruct = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.xavier_uniform_(self.reconstruct.weight)

# ------------------------------
# 6. Improved Training Function with Consistency & Reconstruction Loss and 10 Perturbations
# ------------------------------
def train_improved_multi(model, data_loader, optimizer, device, current_epoch, total_epochs,
                           lambda_consistency, lambda_recon, base_epsilon, num_variations):
    model.train()
    total_loss = 0.0
    total_orig_loss = 0.0
    total_adv_loss = 0.0
    total_consist_loss = 0.0
    total_recon_loss = 0.0

    # Curriculum: scale epsilon linearly with epoch
    current_epsilon = base_epsilon * ((current_epoch + 1) / total_epochs)

    for batch in data_loader:
        set_seed(42)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = outputs.logits
        loss_main_original = F.cross_entropy(logits, labels)
        hidden_states = outputs.hidden_states
        latent = hidden_states[-1][:, 0, :]  # [CLS] token

        adv_losses = []
        consist_losses = []
        recon_losses = []
        for _ in range(num_variations):
            latent_adv = adversarial_perturbation(latent, labels, model, epsilon=current_epsilon)
            logits_adv = model.classifier(latent_adv)
            loss_adv = F.cross_entropy(logits_adv, labels)
            loss_consist = F.kl_div(
                F.log_softmax(logits_adv, dim=-1),
                F.softmax(logits, dim=-1),
                reduction="batchmean"
            )
            reconstructed_latent = model.reconstruct(latent_adv)
            loss_recon = F.mse_loss(reconstructed_latent, latent)
            adv_losses.append(loss_adv)
            consist_losses.append(loss_consist)
            recon_losses.append(loss_recon)

        loss_adv_avg = sum(adv_losses) / num_variations
        loss_consist_avg = sum(consist_losses) / num_variations
        loss_recon_avg = sum(recon_losses) / num_variations

        loss = loss_main_original + loss_adv_avg + lambda_consistency * loss_consist_avg + lambda_recon * loss_recon_avg
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_orig_loss += loss_main_original.item()
        total_adv_loss += loss_adv_avg.item()
        total_consist_loss += loss_consist_avg.item()
        total_recon_loss += loss_recon_avg.item()

    n = len(data_loader)
    return {
        "avg_loss": total_loss / n,
        "avg_orig_loss": total_orig_loss / n,
        "avg_adv_loss": total_adv_loss / n,
        "avg_consist_loss": total_consist_loss / n,
        "avg_recon_loss": total_recon_loss / n,
        "current_epsilon": current_epsilon
    }

# ------------------------------
# 7. Evaluation Function
# ------------------------------
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
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

# ------------------------------
# 8. Grid Search Function for Improved Model Hyperparameters
# ------------------------------
def grid_search_improved(num_epochs_grid=5):
    # Define candidate values
    candidate_lambda_consistency = [0.5, 0.7, 0.9]
    candidate_lambda_recon = [0.3, 0.5, 0.7]
    candidate_base_epsilon = [0.005, 0.01, 0.02]
    candidate_num_variations = [5, 10]

    # Define device locally
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    grid_count = 0
    total_configs = len(candidate_lambda_consistency) * len(candidate_lambda_recon) * len(candidate_base_epsilon) * len(candidate_num_variations)

    for lam_cons in candidate_lambda_consistency:
        for lam_recon in candidate_lambda_recon:
            for eps in candidate_base_epsilon:
                for n_var in candidate_num_variations:
                    grid_count += 1
                    console.log(f"[yellow]Testing config {grid_count}/{total_configs}: λ_consistency={lam_cons}, λ_recon={lam_recon}, base_epsilon={eps}, num_variations={n_var}[/yellow]")
                    set_seed(42)
                    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=28)
                    config.output_hidden_states = True
                    model = BertForSequenceClassificationWithReconstruction.from_pretrained("bert-base-uncased", config=config)
                    model.to(device)
                    optimizer = AdamW(model.parameters(), lr=2e-5)
                    for epoch in range(1, num_epochs_grid+1):
                        train_improved_multi(model, train_loader, optimizer, device,
                                             current_epoch=epoch-1, total_epochs=num_epochs_grid,
                                             lambda_consistency=lam_cons,
                                             lambda_recon=lam_recon,
                                             base_epsilon=eps,
                                             num_variations=n_var)
                    _, val_acc = evaluate(model, val_loader, device)
                    config_key = (lam_cons, lam_recon, eps, n_var)
                    results[config_key] = val_acc
                    console.log(f"Config {config_key}: Final Val Accuracy = {val_acc:.4f}")
    return results

# ------------------------------
# 9. Main Experiment: Baseline vs. Improved and Grid Search
# ------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 20

    ########################
    # Experiment A: Baseline Model (Standard Fine-Tuning)
    ########################
    console.rule("[bold blue]Experiment A: Baseline Model (Standard Fine-Tuning)")
    baseline_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)
    baseline_model.to(device)
    optimizer_baseline = AdamW(baseline_model.parameters(), lr=2e-5)
    baseline_acc_history = []
    for epoch in track(range(1, num_epochs+1), description="Baseline Training"):
        train_loss = train_baseline(baseline_model, train_loader, optimizer_baseline, device)
        val_loss, val_acc = evaluate(baseline_model, val_loader, device)
        baseline_acc_history.append(val_acc)
        console.log(f"Baseline Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}")
    baseline_final = evaluate(baseline_model, val_loader, device)
    console.print(f"\n[bold]Final Baseline Model -- Loss: {baseline_final[0]:.4f}, Accuracy: {baseline_final[1]:.4f}[/bold]\n")

    ########################
    # Experiment B: Improved Model (Perturbation + Consistency + Reconstruction)
    ########################
    console.rule("[bold blue]Experiment B: Improved Model (Perturbation + Consistency + Reconstruction)")
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=28)
    config.output_hidden_states = True
    improved_model = BertForSequenceClassificationWithReconstruction.from_pretrained("bert-base-uncased", config=config)
    improved_model.to(device)
    optimizer_improved = AdamW(improved_model.parameters(), lr=2e-5)
    improved_acc_history = []
    # Fixed hyperparameters for this run:
    best_lambda_consistency = 0.7
    best_lambda_recon = 0.5
    best_base_epsilon = 0.01
    for epoch in track(range(1, num_epochs+1), description="Improved Training"):
        metrics = train_improved_multi(improved_model, train_loader, optimizer_improved, device,
                                       current_epoch=epoch-1, total_epochs=num_epochs,
                                       lambda_consistency=best_lambda_consistency,
                                       lambda_recon=best_lambda_recon,
                                       base_epsilon=best_base_epsilon,
                                       num_variations=10)
        val_loss, val_acc = evaluate(improved_model, val_loader, device)
        improved_acc_history.append(val_acc)
        console.log(
            f"Improved Epoch {epoch}: [ε = {metrics['current_epsilon']:.4f}] Total Loss = {metrics['avg_loss']:.4f}, "
            f"Orig = {metrics['avg_orig_loss']:.4f}, Adv = {metrics['avg_adv_loss']:.4f}, "
            f"Consist = {metrics['avg_consist_loss']:.4f}, Recon = {metrics['avg_recon_loss']:.4f}, "
            f"Val Accuracy = {val_acc:.4f}"
        )
    improved_final = evaluate(improved_model, val_loader, device)
    console.print(f"\n[bold]Final Improved Model -- Loss: {improved_final[0]:.4f}, Accuracy: {improved_final[1]:.4f}[/bold]\n")

    ########################
    # Grid Search on Improved Model Hyperparameters
    ########################
    grid_results = grid_search_improved(num_epochs_grid=5)
    best_config = None
    best_acc = 0.0
    for config_key, acc in grid_results.items():
        console.log(f"Config {config_key}: Val Accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_config = config_key
    console.print(f"\n[bold]Best Grid Search Config: λ_consistency={best_config[0]}, λ_recon={best_config[1]}, base_epsilon={best_config[2]}, num_variations={best_config[3]} with Val Accuracy = {best_acc:.4f}[/bold]\n")

    ########################
    # Plot Comparison of Validation Accuracy
    ########################
    plt.figure(figsize=(8,5))
    plt.plot(range(1, num_epochs+1), baseline_acc_history, marker="o", label="Baseline")
    plt.plot(range(1, num_epochs+1), improved_acc_history, marker="o", label="Improved")
    plt.title("Validation Accuracy Over Epochs (GoEmotions)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()

    return grid_results

if __name__ == "__main__":
    grid_results = main()
