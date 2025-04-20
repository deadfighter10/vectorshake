import warnings

warnings.filterwarnings("ignore")

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For progress bars
from transformers import logging
logging.set_verbosity_error()



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
    """
    Dataset for sentence classification.
    Each JSON entry must have a "query" and a "label".
    """

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
        self.reconstruct = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.xavier_uniform_(self.reconstruct.weight)


# ------------------------------
# 3. Adversarial Perturbation Function
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
# 4. Improved Training Loop with Curriculum, Adversarial Perturbation, and Reconstruction Loss
# ------------------------------

def train_improved(model, data_loader, optimizer, device, current_epoch, total_epochs,
                   lambda_consistency=0.5, lambda_recon=0.5, base_epsilon=0.01):
    model.train()
    total_loss = 0.0

    # Curriculum: scale epsilon linearly with epoch
    current_epsilon = base_epsilon * ((current_epoch + 1) / total_epochs)

    # Use tqdm for batch-level progress
    for batch in data_loader:
        set_seed(42)
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
        latent = hidden_states[-1][:, 0, :]

        latent_adv = adversarial_perturbation(latent, labels, model, epsilon=current_epsilon)
        logits_adv = model.classifier(latent_adv)
        loss_main_adv = F.cross_entropy(logits_adv, labels)

        loss_consist = F.kl_div(F.log_softmax(logits_adv, dim=-1),
                                F.softmax(logits, dim=-1),
                                reduction='batchmean')

        reconstructed_latent = model.reconstruct(latent_adv)
        loss_recon = F.mse_loss(reconstructed_latent, latent)

        loss = loss_main_original + loss_main_adv + lambda_consistency * loss_consist + lambda_recon * loss_recon

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


# ------------------------------
# 5. Standard Evaluation Function (returns loss and accuracy)
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
# 6. Main Experiment Function with Grid Search and Baseline Comparison with Progress Bars
# ------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Load dataset from data/data.json (29 classes expected)
    texts, labels = load_data('../../data/data.json')

    # Split into training and validation (30/70)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.90, random_state=42, stratify=labels
    )

    train_dataset = SentenceDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = SentenceDataset(val_texts, val_labels, tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    num_epochs = 20

    ###############################
    # Experiment A: Baseline Model (No Improvements)
    ###############################
    print("=== Experiment A: Baseline Model (No Improvements) ===")
    baseline_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=29)
    baseline_model.to(device)
    optimizer_baseline = AdamW(baseline_model.parameters(), lr=2e-5)

    baseline_epoch_acc = []
    for epoch in tqdm(range(num_epochs), desc="Baseline Training Epochs", leave=False):
        baseline_model.train()
        total_loss = 0.0
        for batch in train_loader:
            set_seed(42)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            optimizer_baseline.zero_grad()
            outputs = baseline_model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     output_hidden_states=True)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels_batch)
            loss.backward()
            optimizer_baseline.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(baseline_model, val_loader, device)
        baseline_epoch_acc.append(val_acc)
        #tqdm.write(
            #f"Baseline Epoch {epoch + 1}/{num_epochs}: Train Loss {avg_loss:.4f}, Val Loss {val_loss:.4f}, Accuracy {val_acc:.4f}", nolock=True)

    baseline_final = evaluate(baseline_model, val_loader, device)
    print("\nFinal Baseline Model -- Loss: {:.4f} | Accuracy: {:.4f}".format(*baseline_final))

    ###############################
    # Experiment B: Improved Model Grid Search
    ###############################
    candidate_lambda_consistency = [i/10 for i in range(0, 10, 1)]
    candidate_lambda_recon = [i/10 for i in range(0, 10, 1)]
    candidate_base_epsilon = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

    grid_results = {}  # Final accuracy for each combination
    grid_curves = {}  # Epoch-wise accuracy curves

    print("\n=== Experiment B: Grid Search for Improved Model ===")
    for epsilon_val in candidate_base_epsilon:
        for lam_cons in candidate_lambda_consistency:
            for lam_recon in candidate_lambda_recon:
                key = (lam_cons, lam_recon, epsilon_val)
                tqdm.write(f"\nTesting: λ_consistency={lam_cons}, λ_recon={lam_recon}, base_epsilon={epsilon_val}")
                set_seed(42)
                config = BertConfig.from_pretrained(model_name, num_labels=29)
                config.output_hidden_states = True
                model = BertForSequenceClassificationWithReconstruction.from_pretrained(model_name, config=config)
                model.to(device)
                optimizer = AdamW(model.parameters(), lr=2e-5)

                epoch_acc = []
                for epoch in tqdm(range(num_epochs), desc=f"Combo {key}", leave=False):
                    train_loss = train_improved(model, train_loader, optimizer, device,
                                                current_epoch=epoch, total_epochs=num_epochs,
                                                lambda_consistency=lam_cons, lambda_recon=lam_recon,
                                                base_epsilon=epsilon_val)
                    _, val_acc = evaluate(model, val_loader, device)
                    epoch_acc.append(val_acc)
                final_acc = epoch_acc[-1]
                grid_results[key] = final_acc
                grid_curves[key] = epoch_acc
                tqdm.write(f"Final Accuracy: {final_acc:.4f}", nolock=True)

    print("\n=== Grid Search Final Results (Accuracy) ===")
    best_key = None
    best_acc = -1
    for key, acc in grid_results.items():
        print(f"λ_consistency={key[0]}, λ_recon={key[1]}, base_epsilon={key[2]} --> Final Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_key = key
    print(
        f"\nBest combination: λ_consistency={best_key[0]}, λ_recon={best_key[1]}, base_epsilon={best_key[2]} with Accuracy: {best_acc:.4f}")

    ###############################
    # Charting Results
    ###############################
    # Plot baseline training curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), baseline_epoch_acc, marker='o', label="Baseline")
    plt.title("Baseline Model Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()

    # Build heatmaps for each candidate base_epsilon
    for epsilon_val in candidate_base_epsilon:
        heatmap = np.zeros((len(candidate_lambda_consistency), len(candidate_lambda_recon)))
        for i, lam_cons in enumerate(candidate_lambda_consistency):
            for j, lam_recon in enumerate(candidate_lambda_recon):
                key = (lam_cons, lam_recon, epsilon_val)
                heatmap[i, j] = grid_results.get(key, 0)
        plt.figure(figsize=(6, 5))
        plt.imshow(heatmap, cmap='viridis', aspect='auto')
        plt.colorbar(label="Final Accuracy")
        plt.title(f"Heatmap (base_epsilon={epsilon_val})")
        plt.xlabel("λ_recon Index")
        plt.ylabel("λ_consistency Index")
        plt.xticks(ticks=range(len(candidate_lambda_recon)), labels=candidate_lambda_recon)
        plt.yticks(ticks=range(len(candidate_lambda_consistency)), labels=candidate_lambda_consistency)
        plt.show()

    # Plot best improved model training curve
    best_curve = grid_curves[best_key]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), best_curve, marker='o', label="Improved Model (Best)")
    plt.title("Best Improved Model Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
