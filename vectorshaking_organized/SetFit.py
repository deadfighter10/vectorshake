import random

import torch
import math
import numpy as np
from datasets import load_dataset, Dataset
# *** CHANGE 1: Import Trainer and TrainingArguments instead of SetFitTrainer ***
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.model_selection import train_test_split

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_setfit(dataset, epochs, seed, K_class=None, dataset_percentage=None, column_mapping=None):
    set_seed(seed)
    # Load dataset
    print(f"Loading dataset for setfit: {dataset}")
    dataset_names = dataset
    dataset = load_dataset(dataset, trust_remote_code=True)

    if column_mapping is not None:
        if "text" not in column_mapping or "label" not in column_mapping:
            raise ValueError("Column mapping must include 'text' and 'label' keys")
        if "text" not in dataset["train"].features:
            dataset = dataset.rename_column(column_mapping["text"], "text")
        if "label" not in dataset["train"].features:
            print("in2")
            dataset = dataset.rename_column(column_mapping["label"], "label")

    train_texts, train_labels = dataset["train"]["text"], dataset["train"]["label"]
    test_texts, test_labels = dataset["test"]["text"], dataset["test"]["label"]

    if dataset_names == "go_emotions":
        train_labels = [i[0] for i in train_labels]
        test_labels = [i[0] for i in test_labels]

    if K_class is not None and dataset_percentage is not None:
        raise ValueError("Only one of K_class and dataset_percentage can be specified")
    if dataset_percentage is None and K_class is None:
        raise ValueError("Either K_class or dataset_percentage must be specified")

    label_map = {label: i for i, label in enumerate(set(train_labels))}
    num_labels = len(label_map)

    train_charted = {label: [] for label in label_map.keys()}

    for i in range(len(train_labels)):
        train_charted[label_map[train_labels[i]]].append(train_texts[i])

    if dataset_percentage is not None:
        datapoints = int(len(train_labels) * dataset_percentage)
    else:
        datapoints = int(K_class * num_labels)

    final_texts = []
    final_labels = []

    for i in range(datapoints):
        label = i % num_labels
        if len(train_charted[label]) > 0:
            text = random.choice(train_charted[label])
            final_texts.append(text)
            final_labels.append(label)
            train_charted[label].remove(text)

    train_ds_subset = {"text": final_texts, "label": final_labels}
    train_ds_subset = Dataset.from_dict(train_ds_subset)

    '''if len(test_texts) > 750:
        size = 700 / len(test_texts)
        _, test_texts, _, test_labels = train_test_split(
            test_texts, test_labels, test_size=size, stratify=test_labels, random_state=seed
        )'''

    test_ds = {"text": test_texts, "label": test_labels}
    test_ds = Dataset.from_dict(test_ds)

    # Initialize Trainer
    print("Initializing TrainingArguments...")
    args = TrainingArguments(
        output_dir="../SOTA trainings/setfit-agnews-001-percent-new-trainer",
        num_epochs=epochs,
        num_iterations=20,
        batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="none"
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
        args=args,
        train_dataset=train_ds_subset,
        eval_dataset=test_ds,
        metric="accuracy",
    ) #TODO SetFit trainer.py 688 line batching the prediction, use little chunks
    #TODO Rewrite paper: prediction batching, original library overwritten, eval is not changed!

    # Train the model
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # Evaluate the model
    print("Evaluating model on the test set...")
    metrics = trainer.evaluate()
    print(f"\n--- Evaluation Results ---")
    print(f"Metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    # Example usage
    dataset_name = "ag_news"
    epochs = 3
    seed = 42
    K_class = 3
    dataset_percentage = None
    column_mapping = {"text": "text", "label": "label"}

    metrics = train_setfit(dataset_name, epochs, seed, K_class=K_class, dataset_percentage=dataset_percentage, column_mapping=column_mapping)