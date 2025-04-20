import os
import json
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset, DownloadConfig
import optuna
from rich.console import Console
from rich.progress import Progress
import warnings
import wandb
import datetime

warnings.filterwarnings("ignore")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


os.environ["HF_DATASETS_CACHE"] = "./huggingface_cache"
os.environ["WANDB_SILENT"] = "true"


class EnhancedBert(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.reconstruct = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.xavier_uniform_(self.reconstruct.weight)

class Config:
    SEED = 42
    NUM_EPOCHS = 30
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    OPTIMIZATION_TRIALS = 1500
    LOG_FILE = f"logs/training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json"
    RESULTS_FILE = "final_results.json"
    OPTIMIZATION_FILE = "best_params.json"
    STUIES_FILE = "studies.json"
    DATASET_NAME = "trec"


    DATASET_CONFIGS = {
        "ag_news": {
            "name": "AG News",
            "loader": lambda: robust_load_dataset("ag_news"),
            "train_split": "train",
            "test_split": "test",
            "text_key": "text",
            "label_key": "label",
            "train_percent": 0.01
        },
        "trec": {
            "name": "TREC-6",
            "loader": lambda: load_dataset("trec",  trust_remote_code=True),
            "train_split": "train",
            "test_split": "test",
            "text_key": "text",
            "label_key": "coarse_label",
            "train_percent": 0.2
        },
        "semeval": {
            "name": "SemEval-2018",
            "loader": lambda: load_dataset("sem_eval_2018_task_1", "subtask5.english", trust_remote_code=True),
            "train_split": "train",
            "test_split": "test",
            "text_key": "Tweet",
            "label_key": lambda ex: ["anger", "anticipation", "disgust", "fear", "joy",
                                     "love", "optimism", "pessimism", "sadness", "surprise", "trust"][
                np.argmax([ex[e] for e in ["anger", "anticipation", "disgust", "fear", "joy",
                                           "love", "optimism", "pessimism", "sadness", "surprise", "trust"]])
            ],
            "train_percent": 0.5
        },
        "sst5": {
            "name": "SST-5 (GLUE)",
            "loader": lambda: load_dataset("SetFit/sst5", trust_remote_code=True),
            "train_split": "train",
            "test_split": "validation",
            "text_key": "text",
            "label_key": "label",
            "train_percent": 0.5
        }
    }


# ========================
# Core Components
# ========================
console = Console()


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
        console.print(f"[red]Error loading {dataset_name}: {e}[/]")
        raise

def load_data():
    config = Config.DATASET_CONFIGS[Config.DATASET_NAME]
    dataset = config["loader"]()

    # Convert lambda processors to serializable format
    if callable(config["label_key"]):
        label_key_fn = config["label_key"]
        train_texts = [ex[config["text_key"]] for ex in dataset[config["train_split"]]]
        train_labels = [config["label_key"](ex) for ex in dataset[config["train_split"]]]
        test_texts = [ex[config["text_key"]] for ex in dataset[config["test_split"]]]
        test_labels = [config["label_key"](ex) for ex in dataset[config["test_split"]]]
    else:
        label_key_fn = config["label_key"]
        train_texts = dataset[config["train_split"]][config["text_key"]]
        train_labels = dataset[config["train_split"]][config["label_key"]]
        test_texts = dataset[config["test_split"]][config["text_key"]]
        test_labels = dataset[config["test_split"]][config["label_key"]]

    # Stratified sampling
    train_subset, _, labels_subset, _ = train_test_split(
        train_texts, train_labels,
        test_size=(100 - config["train_percent"]) / 100,
        random_state=Config.SEED,
        stratify=train_labels
    )

    if Config.DATASET_NAME == "ag_news":
        test_texts = test_texts[:len(test_texts) // 10]
        test_labels = test_labels[:len(test_labels) // 10]

    if Config.DATASET_NAME == "semeval":
        test_texts = test_texts[:len(test_texts) // 4]
        test_labels = test_labels[:len(test_labels) // 4]

    label_map = {label: idx for idx, label in enumerate(sorted(set(train_labels + test_labels)))}
    console.print(f"[bold]Loaded {config['name']} dataset[/]\nTrain: {len(train_subset)} | Test: {len(test_texts)}")
    return train_subset, labels_subset, test_texts, test_labels, label_map


class TextDataset(Dataset):
    def __init__(self, texts, labels, label_map):
        self.texts = texts
        self.labels = [label_map[l] for l in labels]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=Config.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ========================
# Training Components
# ========================
def fgsm_perturbation(latent, labels, model, epsilon):
    latent = latent.detach().requires_grad_(True)
    loss = F.cross_entropy(model.classifier(latent), labels)
    loss.backward()
    return (latent + epsilon * latent.grad.sign()).detach()


def train_baseline(model, loader, optimizer, device):
    set_seed(Config.SEED)

    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["label"].to(device)
        )
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss


def train_improved(model, loader, optimizer, device, params, epoch):
    set_seed(Config.SEED)

    model.train()
    total_loss = 0
    ce_loss_total = 0
    recon_loss_total = 0
    contrast_loss_total = 0
    mix_loss_total = 0
    consistency_loss_total = 0

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True)
        clean_latent = outputs.hidden_states[-1][:, 0, :]
        clean_logits = outputs.logits

        # ======== Loss Calculations ========
        # 1. Reconstruction Loss
        reconstructed = model.reconstruct(clean_latent)
        loss_reconstruct = F.mse_loss(reconstructed, clean_latent)

        # 2. Adversarial Components
        adv1 = fgsm_perturbation(clean_latent, labels, model, params["epsilon"])
        adv2 = fgsm_perturbation(clean_latent, labels, model, params["epsilon"])


        # 3. Contrastive Loss
        cos_loss = (2 - F.cosine_similarity(adv1, clean_latent) -
                     F.cosine_similarity(adv2, clean_latent)).mean()

        beta_a = torch.tensor(params["beta_a"], device=device)
        beta_b = torch.tensor(params["beta_b"], device=device)
        beta_dist = torch.distributions.Beta(beta_a, beta_b)

        # Generate mixup coefficients
        mix_alpha = beta_dist.sample((input_ids.size(0),))

        # Create shuffled indices on device
        shuffled_idx = torch.randperm(adv1.size(0), device=device)

        # Mixup calculation
        mixed = (mix_alpha.view(-1, 1) * adv1 +
                 (1 - mix_alpha.view(-1, 1)) * adv1[shuffled_idx])

        # KL divergence calculation
        logits_mix = model.classifier(mixed)
        loss_mix = F.kl_div(
            F.log_softmax(logits_mix, dim=-1),
            F.softmax(clean_logits.detach(), dim=-1),
            reduction="batchmean"
        )

        # 5. Consistency Loss
        logits_adv = model.classifier(adv1)
        loss_consistency = F.kl_div(F.log_softmax(logits_adv, dim=-1),
                                    F.softmax(clean_logits.detach(), dim=-1),
                                    reduction="batchmean")

        # ======== Total Loss ========
        loss = (F.cross_entropy(clean_logits, labels) +
                params["lambda_reconstruct"] * loss_reconstruct +
                params["lambda_contrast"] * cos_loss +
                params["lambda_mix"] * loss_mix +
                params["lambda_consistency"] * loss_consistency)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        ce_loss_total += F.cross_entropy(clean_logits, labels).item()
        recon_loss_total += loss_reconstruct.item()
        contrast_loss_total += cos_loss.item()
        mix_loss_total += loss_mix.item()
        consistency_loss_total += loss_consistency.item()

        # Print batch statistics
        if batch_idx % 200 == 0:
            console.print(f"Epoch {epoch} Batch {batch_idx} | "
                          f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                          f"ε: {params['epsilon']:.4f} | "
                          f"β: ({params['beta_a']:.2f},{params['beta_b']:.2f}) | "
                          f"Loss: {loss.item():.4f} = "
                          f"CE: {F.cross_entropy(clean_logits, labels).item():.4f} + "
                          f"Rec: {loss_reconstruct.item():.4f} + "
                          f"Con: {cos_loss.item():.4f} + "
                          f"Mix: {loss_mix.item():.4f} + "
                          f"Cons: {loss_consistency.item():.4f}")
    return {
        "total_loss": total_loss / len(loader),
        "ce_loss": ce_loss_total / len(loader),
        "recon_loss": recon_loss_total / len(loader),
        "contrast_loss": contrast_loss_total / len(loader),
        "mix_loss": mix_loss_total / len(loader),
        "consistency_loss": consistency_loss_total / len(loader)
    }


import sys

def dynamic_curriculum_ema(
    epoch_stats,         # Dictionary containing current epoch's metrics (e.g., {'acc': 0.85, 'mixup_loss': 0.5})
    params,              # Dictionary of current hyperparameters
    prev_ema_acc,        # EMA of accuracy from the *previous* epoch
    prev_ema_mixup_loss, # EMA of mixup loss from the *previous* epoch
    ema_alpha=0.1,       # Smoothing factor for EMA (lower value = more smoothing)
    strength=0.1,        # How aggressively to change parameters
    acc_threshold=0.004, # Minimum EMA accuracy change to trigger epsilon update
    mixup_threshold=0.004 # Minimum EMA mixup loss change to trigger mixup param update
):
    """
    Adjusts training hyperparameters based on the EMA-smoothed changes
    in accuracy and mixup loss over epochs. (Corrected Version)

    Args:
        epoch_stats (dict): Stats from the current epoch (must contain 'acc' and 'mixup_loss').
        params (dict): Current dictionary of hyperparameters to adjust
                       (must contain 'epsilon', 'lambda_mix', 'beta_a', 'beta_b').
        prev_ema_acc (float): The EMA of accuracy calculated in the previous epoch.
                              Use initial accuracy or a sentinel value like -1.0 on the first call.
        prev_ema_mixup_loss (float): The EMA of mixup loss calculated in the previous epoch.
                                   Use initial loss or a sentinel value like sys.float_info.max on the first call.
        ema_alpha (float): Smoothing factor for EMA (0 < ema_alpha <= 1). Default is 0.1.
        strength (float): Base factor for parameter update magnitude. Default is 0.1.
        acc_threshold (float): Threshold for EMA accuracy delta. Default is 0.004.
        mixup_threshold (float): Threshold for EMA mixup loss delta. Default is 0.004.

    Returns:
        tuple: A tuple containing:
            - dict: The updated hyperparameters (new_params).
            - float: The calculated EMA accuracy for the current epoch (current_ema_acc).
            - float: The calculated EMA mixup loss for the current epoch (current_ema_mixup_loss).
    """
    new_params = params.copy()
    current_acc = epoch_stats['acc']
    current_mixup_loss = epoch_stats['mixup_loss']

    # Define practical upper bounds for parameters
    MAX_EPSILON = 0.7  # Example: Sensible max for latent epsilon
    MAX_LAMBDA_MIX = 10.0 # Example: Upper limit for mixup weight
    MAX_BETA = 10.0     # Example: Upper limit for beta parameters

    # Define consistent lower bounds
    MIN_PARAM_VALUE = 0.0001 # Small positive value to avoid zero

    # --- Calculate current EMA values ---
    is_first_epoch = (prev_ema_acc < 0 or prev_ema_mixup_loss == sys.float_info.max)

    if is_first_epoch:
        current_ema_acc = current_acc
        current_ema_mixup_loss = current_mixup_loss
        ema_acc_delta = 0
        ema_mixup_delta = 0
    else:
        # Apply EMA formula (Ensure ema_alpha is valid, e.g., clamp or check input)
        valid_ema_alpha = max(0.0, min(1.0, ema_alpha)) # Ensure alpha is in [0, 1]
        current_ema_acc = valid_ema_alpha * current_acc + (1 - valid_ema_alpha) * prev_ema_acc
        current_ema_mixup_loss = valid_ema_alpha * current_mixup_loss + (1 - valid_ema_alpha) * prev_ema_mixup_loss

        ema_acc_delta = current_ema_acc - prev_ema_acc
        ema_mixup_delta = prev_ema_mixup_loss - current_ema_mixup_loss

    # --- Update parameters based on EMA deltas ---

    # Update epsilon
    if ema_acc_delta > acc_threshold:
        factor = 1 + strength * (ema_acc_delta / acc_threshold)
        # --- CORRECTED: Use MAX_EPSILON instead of math.inf ---
        new_params["epsilon"] = min(new_params["epsilon"] * factor, MAX_EPSILON)
    elif ema_acc_delta < -acc_threshold:
        factor = 1 - strength * (abs(ema_acc_delta) / acc_threshold)
        # --- CORRECTED: Use MIN_PARAM_VALUE ---
        new_params["epsilon"] = max(new_params["epsilon"] * factor, MIN_PARAM_VALUE)

    # Update mixup parameters
    if ema_mixup_delta > mixup_threshold:
        factor = 1 + strength * (ema_mixup_delta / mixup_threshold)
        # --- CORRECTED: Use MAX limits instead of math.inf ---
        new_params["lambda_mix"] = min(new_params["lambda_mix"] * factor, MAX_LAMBDA_MIX)
        new_params["beta_a"] = min(new_params["beta_a"] * factor, MAX_BETA)
        new_params["beta_b"] = min(new_params["beta_b"] * factor, MAX_BETA)
    elif ema_mixup_delta < -mixup_threshold:
        factor = 1 - strength * (abs(ema_mixup_delta) / mixup_threshold)
        # --- CORRECTED: Use MIN_PARAM_VALUE consistently ---
        new_params["lambda_mix"] = max(new_params["lambda_mix"] * factor, MIN_PARAM_VALUE)
        new_params["beta_a"] = max(new_params["beta_a"] * factor, MIN_PARAM_VALUE)
        new_params["beta_b"] = max(new_params["beta_b"] * factor, MIN_PARAM_VALUE)

    return new_params, current_ema_acc, current_ema_mixup_loss


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device))
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels.extend(batch["label"].numpy())
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro")
    }


# ========================
# Optimization & Execution
# ========================
def objective(trial: optuna.Trial):
    set_seed(Config.SEED)

    params = {
        "lambda_mix": trial.suggest_float("lambda_mix", 0.1, 5.0),
        "lambda_contrast": trial.suggest_float("lambda_contrast", 0.1, 5.0),
        "lambda_consistency": trial.suggest_float("lambda_consistency", 0.1, 5.0),
        "lambda_reconstruct": trial.suggest_float("lambda_reconstruct", 0.01, 2.0),

        "beta_a": trial.suggest_float("beta_a", 0.05, 5.0),
        "beta_b": trial.suggest_float("beta_b", 0.05, 6.0),

        "epsilon": trial.suggest_float("epsilon", 0.001, 0.3),

        "lr": trial.suggest_float("lr", 5e-7, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),

        "DAC_strength": trial.suggest_float("DAC_strength", 0.01, 0.3),
        "acc_threshold": trial.suggest_float("acc_threshold", 1e-5, 0.05, log=True),
        "mixup_threshold": trial.suggest_float("mixup_threshold", 1e-6, 0.1, log=True),

        "ema_alpha": trial.suggest_float("ema_alpha", 0.4, 0.9)
    }

    # Load data
    train_texts, train_labels, test_texts, test_labels, label_map = load_data()
    train_dataset = TextDataset(train_texts, train_labels, label_map)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_dataset = TextDataset(test_texts, test_labels, label_map)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedBert.from_pretrained(
        "bert-base-uncased",
        config=BertConfig.from_pretrained("bert-base-uncased", num_labels=len(label_map))
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    old_acc = 0
    old_mixup_loss = 0
    wandb.init(project="ALFA001",
               group=trial.study.study_name,
               name=f"Test {trial.number}",
               reinit=True,
               config=params)

    ema_acc = -1.0
    ema_mixup_loss = sys.float_info.max

    # Training loop
    for epoch in range(Config.NUM_EPOCHS):  # Short optimization run
        stats = train_improved(model, train_loader, optimizer, device, params, epoch)
        metrics = evaluate(model, test_loader, device)
        print(f"Trial {trial.number} Metrics {metrics}")

        stats["acc"] = metrics["accuracy"]
        stats["mixup_loss"] = stats["mix_loss"]
        wandb.log({
            "train_loss": stats['total_loss'],
            "epsilon": params['epsilon'],
            "beta_a": params['beta_a'],
            "beta_b": params['beta_b'],
            "recon_loss": stats['recon_loss'],
            "contrast_loss": stats['contrast_loss'],
            "mix_loss": stats['mix_loss'],
            "consistency_loss": stats['consistency_loss'],
            "val_accuracy": metrics["accuracy"],
            "val_f1": metrics['f1'],
        })

        # Dynamic curriculum update
        params, ema_acc, ema_mixup_loss = dynamic_curriculum_ema(
            epoch_stats=stats,
            params=params,
            prev_ema_acc=ema_acc,
            prev_ema_mixup_loss=ema_mixup_loss,
            ema_alpha=params["ema_alpha"],
            strength=params["DAC_strength"],
            acc_threshold=params["acc_threshold"],
            mixup_threshold=params["mixup_threshold"]
        )

    metrics = evaluate(model, test_loader, device)
    return metrics["accuracy"]


def main():
    # Initialization
    set_seed(Config.SEED)

    # Load data
    train_texts, train_labels, test_texts, test_labels, label_map = load_data()
    train_dataset = TextDataset(train_texts, train_labels, label_map)
    test_dataset = TextDataset(test_texts, test_labels, label_map)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Baseline training
    if console.input("Train baseline? (y/n): ").lower() == "y":
        console.print("[bold]Starting baseline training[/]")
        baseline_model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=len(label_map)).to(device)
        optimizer = AdamW(baseline_model.parameters(), lr=2e-5)

        with Progress() as progress:
            task = progress.add_task("[cyan]Baseline Training...", total=Config.NUM_EPOCHS)
            wandb.init(project="ALFA001",
                       group=f"Baseline Trains - {Config.DATASET_NAME}",
                       name=f"Baseline Train - {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')} - {Config.DATASET_CONFIGS[Config.DATASET_NAME]['train_percent']}%",
                       reinit=True)
            for epoch in range(Config.NUM_EPOCHS):
                train_loss = train_baseline(baseline_model, train_loader, optimizer, device)
                metrics = evaluate(baseline_model, test_loader, device)
                wandb.log({
                    "train_loss": train_loss,
                    "val_accuracy": metrics["accuracy"],
                    "val_f1": metrics["f1"]
                })
                progress.update(task, advance=1)
            metrics = evaluate(baseline_model, test_loader, device)
            console.print(f"[bold]Baseline metrics:[/] {metrics}")
            wandb.finish()

    # Hyperparameter optimization
    if console.input("Run optimization? (y/n): ").lower() == "y":
        db = optuna.storages.RDBStorage("sqlite:///optuna_5V4.db")
        with open(Config.STUIES_FILE) as f:
            studies = json.load(f)
        studies = dict(studies)

        console.print("[bold]-1. Create new study")
        for i, study in studies.items():
            console.print(f"{int(i)}. {study}")

        study_number = console.input("Enter study number: ")
        try:
            int(study_number)
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/]")
            while True:
                study_number = console.input("Enter study number: ")
                try:
                    int(study_number)
                    break
                except ValueError:
                    console.print("[red]Invalid input. Please enter a number.[/]")
        console.print(f"[green]Accepted study number: {study_number}[/]")
        if str(study_number) in studies.keys() and study_number != "-1":
            study_name = studies[str(study_number)]
        else:
            study_name = console.input("Enter study name: ")
            study_name = f"Optimization Test: {study_name} - {Config.DATASET_NAME} - {datetime.datetime.now().strftime('%Y.%m.%d')} - {Config.DATASET_CONFIGS[Config.DATASET_NAME]['train_percent']}%"
            if study_name not in studies.values():
                studies[int(list(studies.keys())[-1])+1] = study_name

                with open(Config.STUIES_FILE, "w") as f:
                    json.dump(studies, f, indent=4)
            else:
                console.print("[bold]Study name already exists. Using existing study.[/]")

        console.rule("[bold green]Starting hyperparameter optimization")
        study = optuna.create_study(direction="maximize", study_name=study_name, storage=db, load_if_exists=True)
        study.optimize(objective, n_trials=Config.OPTIMIZATION_TRIALS)
        best_params = study.best_params
        with open(Config.OPTIMIZATION_FILE, "w") as f:
            json.dump(best_params, f)
    else:
        db = optuna.storages.RDBStorage("sqlite:///optuna_5V4.db")

        with open(Config.STUIES_FILE) as f:
            studies = json.load(f)
        studies = dict(studies)

        for i, study in studies.items():
            study1 = optuna.create_study(direction="maximize", study_name=study, storage=db, load_if_exists=True)
            console.print(f"{int(i)}. {study} - {study1.best_value}")
        study_number = console.input("Enter study number to use parameters of: ")
        try:
            int(study_number)
            if study_number not in studies.keys():
                raise ValueError
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/]")
            while True:
                study_number = console.input("Enter study number: ")
                try:
                    int(study_number)
                    if study_number not in studies.keys():
                        raise ValueError
                    break
                except ValueError:
                    console.print("[red]Invalid input. Please enter a number.[/]")

        study_name = studies[str(study_number)]
        study = optuna.create_study(direction="maximize", study_name=study_name, storage=db, load_if_exists=True)
        best_params = study.best_params

    console.print(f"[bold]Loading improved training with the following Config:[/] \nBest Params: {best_params}\nDataset: {Config.DATASET_NAME}\nDataset Config: {Config.DATASET_CONFIGS[Config.DATASET_NAME]}")
    ans = console.input("Do you want to proceed? (y/n): ")
    if ans.lower() != "y":
        console.print("[red]Exiting...[/]")
        return

    # Improved training with DAC
    console.print("[bold]Starting improved training[/]")
    set_seed(Config.SEED)

    improved_model = EnhancedBert.from_pretrained(
        "bert-base-uncased",
        config=BertConfig.from_pretrained("bert-base-uncased", num_labels=len(label_map))
    ).to(device)
    optimizer = AdamW(improved_model.parameters(), lr=best_params["lr"],
                      weight_decay=best_params["weight_decay"])

    # Dynamic parameters
    current_params = best_params

    with Progress() as progress:
        task = progress.add_task("[cyan]Improved Training...", total=Config.NUM_EPOCHS)
        wandb.init(project="ALFA001",
                   group=f"Independent Tests - {Config.DATASET_NAME}",
                   name=f"Independent Test - {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')} - {Config.DATASET_CONFIGS[Config.DATASET_NAME]['train_percent']}%",
                   reinit=True,
                   config=current_params)

        ema_acc = -1.0
        ema_mixup_loss = sys.float_info.max

        for epoch in range(Config.NUM_EPOCHS):
            # Train improved model with current parameters
            train_stats = train_improved(
                model=improved_model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                params=current_params,
                epoch=epoch
            )

            # Evaluate on validation set
            val_metrics = evaluate(
                model=improved_model,
                loader=test_loader,
                device=device
            )
            val_acc = val_metrics['accuracy']
            train_stats["acc"] = val_acc
            train_stats["mixup_loss"] = train_stats["mix_loss"]

            wandb.log({
                "train_loss": train_stats['total_loss'],
                "epsilon": current_params['epsilon'],
                "beta_a": current_params['beta_a'],
                "beta_b": current_params['beta_b'],
                "recon_loss": train_stats['recon_loss'],
                "contrast_loss": train_stats['contrast_loss'],
                "mix_loss": train_stats['mix_loss'],
                "consistency_loss": train_stats['consistency_loss'],
                "val_accuracy": val_acc,
                "val_f1": val_metrics['f1'],
            })

            # Dynamic curriculum update
            current_params, ema_acc, ema_mixup_loss = dynamic_curriculum_ema(
                epoch_stats=train_stats,
                params=current_params,
                prev_ema_acc=ema_acc,
                prev_ema_mixup_loss=ema_mixup_loss,
                ema_alpha=current_params["ema_alpha"],
                strength=current_params["DAC_strength"],
                acc_threshold=current_params["acc_threshold"],
                mixup_threshold=current_params["mixup_threshold"]
            )

            # Print detailed epoch summary
            console.print(f"\n[bold cyan]Epoch {epoch + 1}/{Config.NUM_EPOCHS} Summary[/]")
            console.print(f"| Training Loss: {train_stats['total_loss']:.4f}")
            console.print(f"| Validation Accuracy: {val_acc:.4%}")
            console.print(f"| Validation F1: {val_metrics['f1']:.4%}")
            console.print(f"| Current Parameters: ε={current_params['epsilon']:.4f} "
                          f"β=({current_params['beta_a']:.2f},{current_params['beta_b']:.2f}) "
                          f"λ=[R:{current_params['lambda_reconstruct']} C:{current_params['lambda_contrast']} "
                          f"M:{current_params['lambda_mix']} K:{current_params['lambda_consistency']}]")
            console.print(f"| Loss Components - CE: {train_stats['ce_loss']:.4f} "
                          f"Recon: {train_stats['recon_loss']:.4f} "
                          f"Contrast: {train_stats['contrast_loss']:.4f} "
                          f"Mix: {train_stats['mix_loss']:.4f} "
                          f"Consistency: {train_stats['consistency_loss']:.4f}\n")
            progress.update(task, advance=1)

    # Save final results
    final_metrics = evaluate(improved_model, test_loader, device)
    console.print(f"[bold]Final metrics:[/] {final_metrics}")
    with open(Config.RESULTS_FILE, "w") as f:
        json.dump({
            "dataset": Config.DATASET_NAME,
            "label_map": label_map,
            "final_metrics": final_metrics,
            "best_params": best_params,
            "log_file": Config.LOG_FILE
        }, f, indent=2)
    wandb.finish()


if __name__ == "__main__":
    main()