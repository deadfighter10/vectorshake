import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification
import json
import numpy as np
import wandb
from rich.progress import track

# Initialize wandb
try:
    wandb.init(
        project="ALFA001",
        entity="sajtoskecske",
        config={
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 10,
            "noise_level": 0.01
        }
    )
except wandb.Error as e:
    print(f"WandB initialization failed: {e}")
    exit()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=29)
bert_model.to(device)


# Functions from the original script

def extract_embeddings(texts, model, tokenizer, device):
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.pooler_output
    return embeddings


def augment_with_noise(embeddings, noise_level=0.01, num_augmentations=5):
    augmented_data = []
    for _ in range(num_augmentations):
        noise = torch.randn_like(embeddings) * noise_level
        augmented_embeddings = embeddings + noise
        augmented_data.append(augmented_embeddings)
    return torch.cat(augmented_data, dim=0)


class EmbeddingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        embedding = torch.tensor(item['embedding'], dtype=torch.float32)
        label = int(item['label'])
        return embedding, label


# Hyperparameter grid search setup
sweep_config = {
    'method': 'grid',
    'parameters': {
        'learning_rate': {
            'values': [1e-5, 1e-4, 5e-4]
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'epochs': {
            'values': [5, 10]
        },
        'noise_level': {
            'values': [0.01, 0.1]
        }
    }
}
sweep_id = wandb.sweep(sweep_config, project="ALFA001")


def train():
    config = wandb.config 

    # Load data
    with open('../data/data.json') as f:
        data = json.load(f)

    original_dataset = []
    for item in data:
        text = item['query']
        embedding = extract_embeddings([text], bert_model, tokenizer, device)
        original_dataset.append({
            'embedding': embedding.squeeze(0).cpu().numpy(),
            'label': item['label']
        })

    augmented_dataset = []
    for item in original_dataset:
        embedding = torch.tensor(item['embedding'], dtype=torch.float32).to(device)
        augmented_embeddings = augment_with_noise(embedding.unsqueeze(0), noise_level=config.noise_level,
                                                  num_augmentations=5)
        for aug_embedding in augmented_embeddings:
            augmented_dataset.append({
                'embedding': aug_embedding.cpu().numpy(),
                'label': item['label']
            })

    final_dataset = original_dataset + augmented_dataset
    dataset = EmbeddingDataset(final_dataset)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=29).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        classifier.train()
        total_loss = 0
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        wandb.log({'epoch': epoch, 'loss': total_loss / len(dataloader)})

    # Evaluate
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = classifier(embeddings)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    wandb.log({'accuracy': accuracy})


wandb.agent(sweep_id, train, count=10)
