import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.strategies.core import switch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset, DownloadConfig
import optuna
from rich.console import Console
import warnings
import sys
import copy

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_DATASETS_CACHE"] = "./huggingface_cache"
os.environ["WANDB_SILENT"] = "true"

torch.use_deterministic_algorithms(True)
warnings.filterwarnings("ignore")


class EnhancedBert(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.reconstruct = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.xavier_uniform_(self.reconstruct.weight)


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
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class Parameters:
    def __init__(self,
                 lambda_mix = 0.2,
                 lambda_contrast = 0.1,
                 lambda_consistency = 0.5,
                 lambda_reconstruct = 0.5,
                 beta_a = 0.4,
                 beta_b = 0.4,
                 epsilon = 0.1,
                 lr = 5e-5,
                 weight_decay = 0.01,
                 DAC_strength = 0.1,
                 acc_threshold = 0.01,
                 mixup_threshold = 0.01,
                 ema_alpha = 0.8,
                 batch_size=16, max_len=128):
        self.lambda_mix = lambda_mix
        self.lambda_contrast = lambda_contrast
        self.lambda_consistency = lambda_consistency
        self.lambda_reconstruct = lambda_reconstruct
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.epsilon = epsilon
        self.lr = lr
        self.weight_decay = weight_decay
        self.DAC_strength = DAC_strength
        self.acc_threshold = acc_threshold
        self.mixup_threshold = mixup_threshold
        self.ema_alpha = ema_alpha
        self.batch_size = batch_size
        self.max_len = max_len

    def getDict(self):
        return {
            "lambda_mix": self.lambda_mix,
            "lambda_contrast": self.lambda_contrast,
            "lambda_consistency": self.lambda_consistency,
            "lambda_reconstruct": self.lambda_reconstruct,
            "beta_a": self.beta_a,
            "beta_b": self.beta_b,
            "epsilon": self.epsilon,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "DAC_strength": self.DAC_strength,
            "acc_threshold": self.acc_threshold,
            "mixup_threshold": self.mixup_threshold,
            "ema_alpha": self.ema_alpha
        }

    def setDict(self, params):
        self.lambda_mix = params["lambda_mix"]
        self.lambda_contrast = params["lambda_contrast"]
        self.lambda_consistency = params["lambda_consistency"]
        self.lambda_reconstruct = params["lambda_reconstruct"]
        self.beta_a = params["beta_a"]
        self.beta_b = params["beta_b"]
        self.epsilon = params["epsilon"]
        self.lr = params["lr"]
        self.weight_decay = params["weight_decay"]
        self.DAC_strength = params["DAC_strength"]
        self.acc_threshold = params["acc_threshold"]
        self.mixup_threshold = params["mixup_threshold"]
        self.ema_alpha = params["ema_alpha"]

    def set_default(self):
        self.lambda_mix = 0.2
        self.lambda_contrast = 0.1
        self.lambda_consistency = 0.5
        self.lambda_reconstruct = 0.5
        self.beta_a = 0.4
        self.beta_b = 0.4
        self.epsilon = 0.1
        self.lr = 5e-5
        self.weight_decay = 0.01
        self.DAC_strength = 0.1
        self.acc_threshold = 0.01
        self.mixup_threshold = 0.01
        self.ema_alpha = 0.8

class Trainer:
    def __init__(self,
                 dataset_name: str,
                 epochs: int,
                 seed: int,
                 args: Parameters,
                 subtask: str = None,
                 K_class: int = None,
                 dataset_percentage: float = None,
                 column_mapping: dict = None,
                 model = None,
                 ):

        self.dataset_name = dataset_name
        self.subtask = subtask
        self.dataset_percentage = dataset_percentage
        self.K_class = K_class
        if self.dataset_percentage is not None and self.K_class is not None:
            raise ValueError("Please provide either dataset_percentage or K_class, not both.")
        if self.dataset_percentage is None and self.K_class is None:
            raise ValueError("Please provide either dataset_percentage or K_class.")
        if self.dataset_percentage:
            self.dataset_percentage = self.dataset_percentage/100
        self.train_loader = None
        self.eval_loader = None
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.seed = seed
        self.console = Console()
        self.args = args
        self.column_mapping = column_mapping
        self.num_labels = None
        self.load_data()
        if model is None:
            self.set_seed()
            self.base_model = EnhancedBert.from_pretrained(
                "bert-base-uncased",
                config=BertConfig.from_pretrained("bert-base-uncased", num_labels=self.num_labels)
            )
        else:
            self.set_seed()
            self.base_model = model

        self.model = copy.deepcopy(self.base_model).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.is_searching = False
        self.inside_eval = False
        self.manual_params = False
        self.switch_DAC_off = False

    def load_data(self):
        self.set_seed()
        download_config = DownloadConfig(
            num_proc=2,
            max_retries=5,
            resume_download=True
        )
        if self.subtask is not None:
            dataset = load_dataset(self.dataset_name, self.subtask, download_config=download_config, trust_remote_code=True)
        else:
            dataset = load_dataset(self.dataset_name, download_config=download_config, trust_remote_code=True)

        #print(list(dataset["train"].features.keys()))
        if self.column_mapping is not None:
            if "text" not in list(dataset["train"].features.keys()):
                dataset = dataset.rename_column(self.column_mapping["text"], "text")
            if "label" not in list(dataset["train"].features.keys()):
                dataset = dataset.rename_column(self.column_mapping["label"], "label")


        train_texts, train_labels = dataset["train"]["text"], dataset["train"]["label"]
        test_texts, test_labels = dataset["test"]["text"], dataset["test"]["label"]
        if self.dataset_name == "go_emotions":
            train_labels = [i[0] for i in train_labels]
            test_labels = [i[0] for i in test_labels]

        label_map = {label: i for i, label in enumerate(set(train_labels))}
        self.num_labels = len(label_map)

        train_charted = {label: [] for label in label_map.keys()}

        for i in range(len(train_labels)):
            train_charted[label_map[train_labels[i]]].append(train_texts[i])

        self.set_seed()
        if self.K_class is None:
            datapoints = int(len(train_labels)*self.dataset_percentage)
        else:
            datapoints = self.K_class * self.num_labels
        final_texts = []
        final_labels = []

        for i in range(datapoints):
            label = i % self.num_labels
            if len(train_charted[label]) > 0:
                text = random.choice(train_charted[label])
                final_texts.append(text)
                final_labels.append(label)
                train_charted[label].remove(text)

        # Split the training set to create a smaller subset
        '''train_texts, _, train_labels, _ = train_test_split(
            train_texts, train_labels, test_size=1-self.dataset_percentage, stratify=train_labels, random_state=self.seed
        )'''
        train_texts = final_texts
        train_labels = final_labels
        print(f"Training data: {train_texts}")
        self.set_seed()

        # Limit the number of testing samples to 700, use train_test_split to ensure stratification
        '''if len(test_texts) > 750:
            size = 700/len(test_texts)
            _, test_texts, _, test_labels = train_test_split(
                test_texts, test_labels, test_size=size, stratify=test_labels, random_state=self.seed
            )'''

        self.set_seed()

        self.train_loader = DataLoader(
            TextDataset(train_texts, train_labels, label_map),
            batch_size=self.args.batch_size,
            shuffle=True,
        )

        self.eval_loader = DataLoader(
            TextDataset(test_texts, test_labels, label_map),
            batch_size=self.args.batch_size,
        )

        self.console.print(f"Loaded dataset: {self.dataset_name} with {len(train_texts)} training samples and {len(test_texts)} test samples.")

    def train(self):
        self.set_seed()
        self.model = copy.deepcopy(self.base_model).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        if not self.is_searching and not self.manual_params:
            self.get_best_params()

        ema_acc = -1.0
        ema_mixup_loss = sys.float_info.max

        for epoch in range(self.epochs):
            # Train improved model with current parameters
            train_stats = self.train_improved(epoch=epoch)

            # Evaluate on validation set
            self.inside_eval = True
            #TODO Val metrics -> Train metrics
            val_metrics = self.evaluate()
            self.inside_eval = False
            train_stats["acc"] = val_metrics['accuracy']
            train_stats["mixup_loss"] = train_stats["mix_loss"]

            # Dynamic curriculum update
            if not self.switch_DAC_off:
                self.console.print("[bold]Dynamic Curriculum Update...[/bold]")
                current_params, ema_acc, ema_mixup_loss = self.dynamic_curriculum_ema(
                    epoch_stats=train_stats,
                    params=self.args.getDict(),
                    prev_ema_acc=ema_acc,
                    prev_ema_mixup_loss=ema_mixup_loss,
                    ema_alpha=self.args.ema_alpha,
                    strength=self.args.DAC_strength,
                    acc_threshold=self.args.acc_threshold,
                    mixup_threshold=self.args.mixup_threshold
                )
                self.args.setDict(current_params)

    def evaluate(self):
        self.model.eval()
        self.set_seed()
        preds, labels = [], []
        with torch.no_grad():
            for batch in self.eval_loader:
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device))
                preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                labels.extend(batch["label"].numpy())
        if not self.inside_eval:
            self.console.print(f"[bold green]Evaluation:\n\nAccuracy: {accuracy_score(labels, preds):.4f}\n"
                                f"F1: {f1_score(labels, preds, average='macro'):.4f}\n"
                                f"Precision: {precision_score(labels, preds, average='macro'):.4f}\n"
                                f"Recall: {recall_score(labels, preds, average='macro'):.4f}"
        )
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro"),
            "precision": precision_score(labels, preds, average="macro"),
            "recall": recall_score(labels, preds, average="macro")
        }

    def set_seed(self):
        seed = self.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fgsm_perturbation(self, latent, labels):
        latent = latent.detach().requires_grad_(True)
        loss = F.cross_entropy(self.model.classifier(latent), labels)
        loss.backward()
        return (latent + self.args.epsilon * latent.grad.sign()).detach()

    def train_improved(self, epoch):
        #self.set_seed()
        self.model.train()
        total_loss = 0
        ce_loss_total = 0
        recon_loss_total = 0
        contrast_loss_total = 0
        mix_loss_total = 0
        consistency_loss_total = 0

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            clean_latent = outputs.hidden_states[-1][:, 0, :]
            clean_logits = outputs.logits

            # ======== Loss Calculations ========
            # 1. Reconstruction Loss
            reconstructed = self.model.reconstruct(clean_latent)
            loss_reconstruct = F.mse_loss(reconstructed, clean_latent)

            # 2. Adversarial Components
            adv1 = self.fgsm_perturbation(clean_latent, labels)
            adv2 = self.fgsm_perturbation(clean_latent, labels)


            # 3. Contrastive Loss
            cos_loss = (2 - F.cosine_similarity(adv1, clean_latent) - F.cosine_similarity(adv2, clean_latent)).mean()

            beta_a = torch.tensor(self.args.beta_a, device=self.device)
            beta_b = torch.tensor(self.args.beta_b, device=self.device)
            beta_dist = torch.distributions.Beta(beta_a, beta_b)

            mix_alpha = beta_dist.sample((input_ids.size(0),))

            shuffled_idx = torch.randperm(adv1.size(0), device=self.device)

            # Mixup calculation
            mixed = (mix_alpha.view(-1, 1) * adv1 +
                     (1 - mix_alpha.view(-1, 1)) * adv1[shuffled_idx])

            # KL divergence calculation
            logits_mix = self.model.classifier(mixed)
            loss_mix = F.kl_div(
                F.log_softmax(logits_mix, dim=-1),
                F.softmax(clean_logits.detach(), dim=-1),
                reduction="batchmean"
            )

            # 5. Consistency Loss
            logits_adv = self.model.classifier(adv1)
            loss_consistency = F.kl_div(F.log_softmax(logits_adv, dim=-1),
                                        F.softmax(clean_logits.detach(), dim=-1),
                                        reduction="batchmean")

            # ======== Total Loss ========
            loss = (F.cross_entropy(clean_logits, labels) +
                    self.args.lambda_reconstruct * loss_reconstruct +
                    self.args.lambda_contrast * cos_loss +
                    self.args.lambda_mix * loss_mix +
                    self.args.lambda_consistency * loss_consistency)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            ce_loss_total += F.cross_entropy(clean_logits, labels).item()
            recon_loss_total += loss_reconstruct.item()
            contrast_loss_total += cos_loss.item()
            mix_loss_total += loss_mix.item()
            consistency_loss_total += loss_consistency.item()

            #TODO Train accuracy calc

            # Print batch statistics
            if batch_idx % 200 == 0:
                self.console.print(f"[bold]IMPROVED TRAINING:[/bold] Epoch {epoch} Batch {batch_idx} | Loss: {loss.item():.4f}")
                '''console.print(f"Epoch {epoch} Batch {batch_idx} | "
                              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                              f"ε: {params['epsilon']:.4f} | "
                              f"β: ({params['beta_a']:.2f},{params['beta_b']:.2f}) | "
                              f"Loss: {loss.item():.4f} = "
                              f"CE: {F.cross_entropy(clean_logits, labels).item():.4f} + "
                              f"Rec: {loss_reconstruct.item():.4f} + "
                              f"Con: {cos_loss.item():.4f} + "
                              f"Mix: {loss_mix.item():.4f} + "
                              f"Cons: {loss_consistency.item():.4f}")'''
        return {
            "total_loss": total_loss / len(self.train_loader),
            "ce_loss": ce_loss_total / len(self.train_loader),
            "recon_loss": recon_loss_total / len(self.train_loader),
            "contrast_loss": contrast_loss_total / len(self.train_loader),
            "mix_loss": mix_loss_total / len(self.train_loader),
            "consistency_loss": consistency_loss_total / len(self.train_loader)
        }

    def dynamic_curriculum_ema(
            self,
            epoch_stats,
            params,
            prev_ema_acc,
            prev_ema_mixup_loss,
            ema_alpha=0.1,
            strength=0.1,
            acc_threshold=0.004,
            mixup_threshold=0.004 ):
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

    def search_parameters(self, trials=100):
        self.is_searching = True
        self.set_seed()
        db = optuna.storages.RDBStorage("sqlite:///vectorshake_optim_data.db")
        if self.dataset_percentage is not None:
            study_name = f"ALFA001_{self.dataset_name}_{self.dataset_percentage}_{self.epochs}_{self.seed}"
        else:
            study_name = f"ALFA001_{self.dataset_name}_{self.K_class}_{self.epochs}_{self.seed}"
        study = optuna.create_study(study_name=study_name, direction="maximize", storage=db, load_if_exists=True)
        study.optimize(self.objective, n_trials=trials)
        self.console.print(f"[bold green]Best trial: {study.best_trial.number} with value: {study.best_trial.value}[/bold green]")
        self.is_searching = False

    def objective(self, trial: optuna.Trial):
        self.set_seed()

        params = {
            # --- Loss Weights (Log scale often good if unsure of magnitude) ---
            # How much each regularization component contributes vs base CE loss.
            # Values >> 1.0 might dominate CE loss; start < 1.0 or around 1.0.
            "lambda_mix": trial.suggest_float("lambda_mix", 0.01, 1.0, log=True),
            "lambda_contrast": trial.suggest_float("lambda_contrast", 0.01, 1.0, log=True),
            "lambda_consistency": trial.suggest_float("lambda_consistency", 0.05, 1.5, log=True),
            # Maybe allow slightly > 1
            "lambda_reconstruct": trial.suggest_float("lambda_reconstruct", 0.005, 0.5, log=True),
            # Reconstruction often weighted lower

            # --- Beta Distribution (Mixup Shape) ---
            # Controls how inputs are mixed. Values near 0.1-0.5 are common.
            # Symmetric values (a=b) often used, but not required.
            "beta_a": trial.suggest_float("beta_a", 0.1, 4.0),
            "beta_b": trial.suggest_float("beta_b", 0.1, 4.0),

            # --- FGSM Epsilon (Latent Space Perturbation Size) ---
            # Smaller values usually needed for latent space vs input space.
            "epsilon": trial.suggest_float("epsilon", 0.001, 0.1),  # Start narrow, 0.3 was likely too high

            # --- Optimizer Hyperparameters ---
            # Standard ranges for BERT fine-tuning.
            "lr": trial.suggest_float("lr", 5e-6, 9e-4, log=True),  # Centered around typical 1e-5 to 5e-5
            "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.03),  # Focused around 0.01

            # --- Dynamic Curriculum (DAC) Parameters ---
            # Controls how the *other* parameters adapt during training.
            # Tuning these simultaneously adds complexity. Consider fixing them initially.
            "DAC_strength": trial.suggest_float("DAC_strength", 0.02, 0.6),  # How strongly DAC adjusts params
            "acc_threshold": trial.suggest_float("acc_threshold", 5e-4, 0.03, log=True),
            # Sensitivity to accuracy change
            "mixup_threshold": trial.suggest_float("mixup_threshold", 5e-4, 0.05, log=True),
            # Sensitivity to mixup loss change
            "ema_alpha": trial.suggest_float("ema_alpha", 0.1, 0.7)
            # Smoothing factor (lower = more smoothing/slower adaptation)
        }

        # Load data
        self.args.setDict(params)
        self.set_seed()
        self.model = copy.deepcopy(self.base_model).to(self.device)
        self.train()
        self.set_seed()
        metrics = self.evaluate()
        return metrics["accuracy"]

    def get_best_params(self):
        db = optuna.storages.RDBStorage("sqlite:///vectorshake_optim_data.db")
        self.set_seed()
        if self.dataset_percentage is not None:
            study_name = f"ALFA001_{self.dataset_name}_{self.dataset_percentage}_{self.epochs}_{self.seed}"
        else:
            study_name = f"ALFA001_{self.dataset_name}_{self.K_class}_{self.epochs}_{self.seed}"
        try:
            study = optuna.load_study(study_name=study_name, storage=db)
            best_trial = study.best_trial
            best_params = best_trial.params
            self.args.setDict(best_params)
            self.console.print(f"[bold green]Study found and loaded with potential accuracy: {best_trial.value}.[/bold green]")
        except KeyError:
            print(f"Study {study_name} not found.")
            self.args.set_default()
        except ValueError:
            print(f"Study {study_name} not found.")
            self.args.set_default()

    def train_baseline(self):
        self.set_seed()
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=self.num_labels).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        self.console.print("[bold]Training Baseline Model...[/bold]")
        self.model.train()
        for epoch in range(self.epochs):
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["label"].to(self.device)
                )
                outputs.loss.backward()
                self.optimizer.step()
        self.inside_eval = True
        metrics = self.evaluate()
        self.inside_eval = False
        self.console.print(f"[bold green]Baseline Evaluation:\n\nAccuracy: {metrics['accuracy']:.4f}\n"
                            f"F1: {metrics['f1']:.4f}\n"
                            f"Precision: {metrics['precision']:.4f}\n"
                            f"Recall: {metrics['recall']:.4f}"
        )
        return metrics