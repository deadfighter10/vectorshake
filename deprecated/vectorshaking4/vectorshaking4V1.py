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

# ------------------------------------------------------------------------
# 0. Fix Random Seeds for Reproducibility
# ------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------------------------------------------------------
# 1. Dataset Class for GoEmotions (Simplified)
# ------------------------------------------------------------------------
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

# ------------------------------------------------------------------------
# 2. Load GoEmotions Dataset (Simplified) for 28 Classes
# ------------------------------------------------------------------------
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

# ------------------------------------------------------------------------
# 3. Custom Model with Reconstruction Head
# ------------------------------------------------------------------------
class BertForSequenceClassificationWithReconstruction(BertForSequenceClassification):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # The reconstruction head maps a token's hidden representation back to its original space.
        self.reconstruct = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.xavier_uniform_(self.reconstruct.weight)

# ------------------------------------------------------------------------
# 4. Latent Manifold Mixup Training Function
# ------------------------------------------------------------------------
def train_latent_mixup(model, data_loader, optimizer, device, lambda_mix=1.0, beta_a=0.4, beta_b=0.4):
    """
    For each batch, extract the [CLS] token from the final hidden states, then perform mixup in the latent space.
    The mixup coefficient is sampled from a Beta(beta_a, beta_b) distribution.
    The mixed latent and the mixed label (soft) are used to compute a soft loss (KL divergence).
    We combine this with the standard classification loss.
    """
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        set_seed(42)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        # Forward pass to get hidden states; we require hidden states.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Extract [CLS] token from final hidden states for each sample
        latent = outputs.hidden_states[-1][:, 0, :]  # shape: [batch, hidden_size]
        logits_orig = outputs.logits
        loss_orig = F.cross_entropy(logits_orig, labels)

        # One-hot encode labels
        num_classes = model.config.num_labels
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

        # Sample mixup coefficient from Beta distribution
        alpha = torch.distributions.Beta(beta_a, beta_b).sample((latent.size(0),)).to(device)  # shape: [batch]
        alpha = alpha.unsqueeze(1)  # shape: [batch, 1]

        # Shuffle batch indices
        indices = torch.randperm(latent.size(0))
        latent_shuffled = latent[indices]
        labels_shuffled = labels_onehot[indices]

        # Mix latent representations and labels
        mixed_latent = alpha * latent + (1 - alpha) * latent_shuffled  # shape: [batch, hidden_size]
        mixed_labels = alpha * labels_onehot + (1 - alpha) * labels_shuffled  # shape: [batch, num_classes]

        # Pass mixed latent through classifier
        logits_mix = model.classifier(mixed_latent)
        log_probs = F.log_softmax(logits_mix, dim=-1)

        # Compute soft cross-entropy (KL divergence) loss between mixed predictions and mixed labels
        loss_mix = F.kl_div(log_probs, mixed_labels, reduction="batchmean")

        # Total loss is original loss plus weighted mixup loss
        loss = loss_orig + lambda_mix * loss_mix
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

# ------------------------------------------------------------------------
# 5. Evaluation Function
# ------------------------------------------------------------------------
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

# ------------------------------------------------------------------------
# 6. Grid Search Function for Latent Manifold Mixup Hyperparameters
# ------------------------------------------------------------------------
def grid_search_latent_mixup(num_epochs_grid=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_lambda_mix = [1, 2, 3]
    candidate_beta_a = [1, 2, 5]
    candidate_beta_b = [1, 2, 5]
    candidate_num_variations = [1]
    results = {}
    grid_count = 0
    total_configs = len(candidate_lambda_mix) * len(candidate_beta_a) * len(candidate_beta_b) * len(candidate_num_variations)

    for lambda_mix in candidate_lambda_mix:
        for beta_a in candidate_beta_a:
            for beta_b in candidate_beta_b:
                for n_var in candidate_num_variations:
                    grid_count += 1
                    console.log(f"[yellow]Testing config {grid_count}/{total_configs}: lambda_mix={lambda_mix}, beta_a={beta_a}, beta_b={beta_b}, num_variations={n_var}[/yellow]")
                    set_seed(42)
                    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=28)
                    config.output_hidden_states = True
                    model = BertForSequenceClassificationWithReconstruction.from_pretrained("bert-base-uncased", config=config)
                    model.to(device)
                    optimizer = AdamW(model.parameters(), lr=2e-5)
                    # Train for a few epochs for grid search
                    for epoch in range(1, num_epochs_grid+1):
                        train_latent_mixup(model, train_loader, optimizer, device, lambda_mix=lambda_mix, beta_a=beta_a, beta_b=beta_b)
                    _, val_acc = evaluate(model, val_loader, device)
                    config_key = (lambda_mix, beta_a, beta_b, n_var)
                    results[config_key] = val_acc
                    console.log(f"Config {config_key}: Final Val Accuracy = {val_acc:.4f}")
    return results

# ------------------------------------------------------------------------
# 7. Main Experiment: Train Improved Model with Best Hyperparameters and Compare with Baseline
# ------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50

    # Print baseline reference (trained baseline accuracy)
    console.rule("[bold blue]Baseline Model Reference")
    # For this experiment, assume baseline accuracy is 52.99%
    baseline_reference = 52.99
    console.print(f"[bold]Baseline Accuracy: {baseline_reference:.2f}%[/bold]\n")

    # Run grid search for latent manifold mixup hyperparameters
    console.rule("[bold blue]Grid Search for Latent Manifold Mixup Hyperparameters")
    grid_results = grid_search_latent_mixup(num_epochs_grid=5)
    best_config = None
    best_acc = 0.0
    for config_key, acc in grid_results.items():
        console.log(f"Config {config_key}: Val Accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_config = config_key
    console.print(f"\n[bold]Best Grid Search Config: lambda_mix={best_config[0]}, beta_a={best_config[1]}, beta_b={best_config[2]}, num_variations={best_config[3]} with Val Accuracy = {best_acc:.4f}[/bold]\n")

    # Train the improved model using the best hyperparameters from grid search
    console.rule("[bold blue]Improved Model Training with Latent Manifold Mixup")
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=28)
    config.output_hidden_states = True
    improved_model = BertForSequenceClassificationWithReconstruction.from_pretrained("bert-base-uncased", config=config)
    improved_model.to(device)
    optimizer = AdamW(improved_model.parameters(), lr=2e-5)

    improved_acc_history = []
    for epoch in track(range(1, num_epochs+1), description="Improved Training"):
        train_loss = train_latent_mixup(improved_model, train_loader, optimizer, device,
                                        lambda_mix=best_config[0],
                                        beta_a=best_config[1],
                                        beta_b=best_config[2])
        val_loss, val_acc = evaluate(improved_model, val_loader, device)
        improved_acc_history.append(val_acc)
        console.log(f"[Improved] Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}")

    improved_final = evaluate(improved_model, val_loader, device)
    console.print(f"\n[bold]Final Improved Model -- Loss: {improved_final[0]:.4f}, Accuracy: {improved_final[1]:.4f}[/bold]\n")

    # Plot the training curve
    plt.figure(figsize=(8,5))
    plt.plot(range(1, num_epochs+1), improved_acc_history, marker="o", label="Improved")
    plt.axhline(y=baseline_reference, color='r', linestyle='--', label=f"Baseline ({baseline_reference}%)")
    plt.title("Validation Accuracy Over Epochs (Latent Manifold Mixup)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()

    return grid_results

if __name__ == "__main__":
    grid_results = main()
