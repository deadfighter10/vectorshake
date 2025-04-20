import warnings
warnings.filterwarnings("ignore")

import json
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

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, track

logging.set_verbosity_error()
console = Console()

# ------------------------------
# 0. Fix Random Seeds for Reproducibility
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
# 1. Data Preparation from data/data.json
# ------------------------------

class SentenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = [int(label) for label in labels]  # 29 classes expected
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
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    texts = [entry["query"] for entry in data]
    labels = [entry["label"] for entry in data]
    return texts, labels

# ------------------------------
# 2. Custom Model with Reconstruction Head
# ------------------------------

class BertForSequenceClassificationWithReconstruction(BertForSequenceClassification):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Add a reconstruction head to map perturbed latent back to original latent space
        self.reconstruct = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.xavier_uniform_(self.reconstruct.weight)

# ------------------------------
# 3. Adversarial Perturbation Function
# ------------------------------

def adversarial_perturbation(latent, labels, model, epsilon):
    # Create a leaf tensor from latent and set requires_grad
    latent = latent.detach().clone().requires_grad_()
    logits = model.classifier(latent)
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, latent, retain_graph=True)[0]
    delta = epsilon * grad.sign()
    latent_adv = latent + delta
    return latent_adv.detach()

# ------------------------------
# 4. Improved Training Loop with 10 Perturbed Variations
# ------------------------------

def train_improved_multi(model, data_loader, optimizer, device, current_epoch, total_epochs,
                           lambda_consistency=0.9, lambda_recon=0.7, base_epsilon=0.02, num_variations=10):
    model.train()
    total_loss = 0.0
    total_orig_loss = 0.0
    total_adv_loss = 0.0
    total_consist_loss = 0.0
    total_recon_loss = 0.0

    # Curriculum: epsilon increases linearly with epoch
    current_epsilon = base_epsilon * ((current_epoch + 1) / total_epochs)

    for batch in data_loader:
        set_seed(42)  # keep consistency for non-perturbed operations
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True)
        logits = outputs.logits
        loss_main_original = F.cross_entropy(logits, labels)

        hidden_states = outputs.hidden_states
        latent = hidden_states[-1][:, 0, :]  # [CLS] token

        adv_losses = []
        consist_losses = []
        recon_losses = []

        # Generate multiple perturbed variations
        for _ in range(num_variations):
            latent_adv = adversarial_perturbation(latent, labels, model, epsilon=current_epsilon)
            logits_adv = model.classifier(latent_adv)
            adv_loss = F.cross_entropy(logits_adv, labels)
            consist_loss = F.kl_div(F.log_softmax(logits_adv, dim=-1),
                                    F.softmax(logits, dim=-1),
                                    reduction='batchmean')
            reconstructed_latent = model.reconstruct(latent_adv)
            recon_loss = F.mse_loss(reconstructed_latent, latent)
            adv_losses.append(adv_loss)
            consist_losses.append(consist_loss)
            recon_losses.append(recon_loss)

        loss_main_adv_avg = sum(adv_losses) / num_variations
        loss_consist_avg = sum(consist_losses) / num_variations
        loss_recon_avg = sum(recon_losses) / num_variations

        loss = (loss_main_original +
                loss_main_adv_avg +
                lambda_consistency * loss_consist_avg +
                lambda_recon * loss_recon_avg)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_orig_loss += loss_main_original.item()
        total_adv_loss += loss_main_adv_avg.item()
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
# 5. Evaluation Function (returns loss and accuracy)
# ------------------------------

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True)
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
# 6. Main Experiment Function (Baseline vs Improved) with Single-Line Progress Updates
# ------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load data from data/data.json (29 classes expected)
    texts, labels = load_data("../../data/data.json")
    # Split data into 10% training and 90% evaluation with stratification
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.90, random_state=42, stratify=labels
    )

    train_dataset = SentenceDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentenceDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    num_epochs = 30

    ########################
    # Experiment A: Baseline Model (No Improvements)
    ########################
    console.rule("[bold blue]Experiment A: Baseline Model (No Improvements)")
    baseline_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=29)
    baseline_model.to(device)
    optimizer_baseline = AdamW(baseline_model.parameters(), lr=2e-5)

    baseline_epoch_info = []
    for epoch in track(range(1, num_epochs+1), description="Baseline Training"):
        baseline_model.train()
        total_loss = 0.0
        for batch in train_loader:
            set_seed(42)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            optimizer_baseline.zero_grad()
            outputs = baseline_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels_batch)
            loss.backward()
            optimizer_baseline.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(baseline_model, val_loader, device)
        baseline_epoch_info.append({
            "epoch": epoch,
            "avg_train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        #console.log(f"Baseline Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}")
    baseline_final = evaluate(baseline_model, val_loader, device)
    console.print(f"\n[bold]Final Baseline -- Loss: {baseline_final[0]:.4f}, Accuracy: {baseline_final[1]:.4f}[/bold]\n")

    ########################
    # Experiment B: Improved Model with 10 Perturbed Variations
    ########################
    console.rule("[bold blue]Experiment B: Improved Model")
    # Fixed hyperparameters: λ_consistency = 0.9, λ_recon = 0.7, base_epsilon = 0.02
    best_lambda_consistency = 0.9
    best_lambda_recon = 0.7
    best_base_epsilon = 0.02

    set_seed(42)
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=29)
    config.output_hidden_states = True
    improved_model = BertForSequenceClassificationWithReconstruction.from_pretrained("bert-base-uncased", config=config)
    improved_model.to(device)
    optimizer_improved = AdamW(improved_model.parameters(), lr=2e-5)

    improved_epoch_info = []
    for epoch in track(range(1, num_epochs+1), description="Improved Training"):
        metrics = train_improved_multi(improved_model, train_loader, optimizer_improved, device,
                                       current_epoch=epoch-1, total_epochs=num_epochs,
                                       lambda_consistency=best_lambda_consistency,
                                       lambda_recon=best_lambda_recon,
                                       base_epsilon=best_base_epsilon,
                                       num_variations=10)
        _, val_acc = evaluate(improved_model, val_loader, device)
        improved_epoch_info.append({
            "epoch": epoch,
            "avg_loss": metrics["avg_loss"],
            "avg_orig_loss": metrics["avg_orig_loss"],
            "avg_adv_loss": metrics["avg_adv_loss"],
            "avg_consist_loss": metrics["avg_consist_loss"],
            "avg_recon_loss": metrics["avg_recon_loss"],
            "epsilon": metrics["current_epsilon"],
            "val_acc": val_acc
        })

    improved_final = evaluate(improved_model, val_loader, device)
    console.print(f"\n[bold]Final Improved -- Loss: {improved_final[0]:.4f}, Accuracy: {improved_final[1]:.4f}[/bold]\n")

    ########################
    # Plot Training Curves
    ########################
    baseline_acc = [info["val_acc"] for info in baseline_epoch_info]
    improved_acc = [info["val_acc"] for info in improved_epoch_info]

    plt.figure(figsize=(8,5))
    plt.plot(range(1, num_epochs+1), baseline_acc, marker='o', label="Baseline")
    plt.plot(range(1, num_epochs+1), improved_acc, marker='o', label="Improved")
    plt.title("Validation Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
