import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import json
import pickle
from warnings import filterwarnings
from sklearn.decomposition import PCA
import numpy as np
from rich.progress import track

filterwarnings("ignore")
torch.cuda.empty_cache()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=29)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

def compute_semantical_subspace(embeddings, num_semantical_dims=None):
    """
    Compute the semantical subspace using PCA.
    Args:
        embeddings: Torch tensor of shape [num_samples, hidden_size].
        num_semantical_dims: Number of semantical dimensions to retain.
    Returns:
        pca: Fitted PCA model.
    """
    pca = PCA(n_components=num_semantical_dims)
    pca.fit(embeddings.cpu().numpy())
    return pca

# Function to extract embeddings from BERT
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
        embeddings = outputs.pooler_output  # [batch_size, hidden_size]
    return embeddings

# Function to augment embeddings with Gaussian noise
def augment_with_noise(embeddings, noise_level=0.01, num_augmentations=5):
    augmented_data = []
    for _ in range(num_augmentations):
        noise = torch.randn_like(embeddings) * noise_level
        augmented_embeddings = embeddings + noise
        augmented_data.append(augmented_embeddings)
    return torch.cat(augmented_data, dim=0)

# Function to augment embeddings in the semantical subspace
def augment_semantical_dimensions(embeddings, pca, noise_level=0.01):
    """
    Add Gaussian noise only to the semantical dimensions.
    Args:
        embeddings: Torch tensor of shape [batch_size, hidden_size].
        pca: Fitted PCA model representing the semantical subspace.
        noise_level: Standard deviation of Gaussian noise.
    Returns:
        Augmented embeddings: Torch tensor of the same shape as input.
    """
    # Project embeddings into the semantical subspace
    embeddings_np = embeddings.cpu().numpy()
    semantical_components = pca.transform(embeddings_np)

    # Add Gaussian noise to the semantical dimensions
    noise = np.random.randn(*semantical_components.shape) * noise_level
    semantical_components_noisy = semantical_components + noise

    # Reconstruct embeddings with augmented semantical dimensions
    augmented_embeddings_np = pca.inverse_transform(semantical_components_noisy)

    # Combine original embeddings with the augmented semantical dimensions
    augmented_embeddings = torch.tensor(augmented_embeddings_np, dtype=torch.float32).to(embeddings.device)
    return augmented_embeddings

# Augment embeddings during augmentation step
def augment_with_semantical_noise(embeddings, pca, noise_level=0.01, num_augmentations=5):
    augmented_data = []
    for _ in range(num_augmentations):
        augmented_embeddings = augment_semantical_dimensions(embeddings, pca, noise_level)
        augmented_data.append(augmented_embeddings)
    return torch.cat(augmented_data, dim=0)

# Define a Dataset class for training
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

# Define a complex, high-end classifier head
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define a Dataset class for testing
class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        embedding = torch.tensor(item['embedding'], dtype=torch.float32)
        label = int(item['label'])
        return embedding, label

NOISE_LEVEL = 0.1
NUM_AUGMENTATIONS = 100
noises = []
accuracies = []
augmentations = []
for _ in track(range(20), description="Running experiments: "):
    # Load original dataset
    with open('../data/data.json') as f:
        data = json.load(f)

    # Prepare original data with embeddings
    original_dataset = []
    classes_present = []
    for item in data:
        if int(item['label']) not in classes_present:
            classes_present.append(int(item['label']))
        else:
            continue
        text = item['query']  # Assuming 'query' contains the text
        embedding = extract_embeddings([text], bert_model, tokenizer, device)
        original_dataset.append({
            'embedding': embedding.squeeze(0).cpu().numpy(),  # Convert embedding to numpy array
            'label': item['label']
        })
    print(len(original_dataset))

    # Load the original dataset and compute the semantical subspace
    original_embeddings = torch.stack([torch.tensor(item['embedding'], dtype=torch.float32) for item in original_dataset]).to(device)
    pca = compute_semantical_subspace(original_embeddings)

    # Update the augmentation loop
    augmented_dataset = []
    for item in original_dataset:
        embedding = torch.tensor(item['embedding'], dtype=torch.float32).to(device)
        augmented_embeddings = augment_with_semantical_noise(embedding.unsqueeze(0), pca, noise_level=NOISE_LEVEL, num_augmentations=NUM_AUGMENTATIONS)
        for aug_embedding in augmented_embeddings:
            augmented_dataset.append({
                'embedding': aug_embedding.cpu().numpy(),
                'label': item['label']
            })

    # Combine original and augmented datasets
    final_dataset = original_dataset + augmented_dataset

    print(len(final_dataset))

    # Split dataset into training and testing sets
    full_dataset = EmbeddingDataset(final_dataset)
    '''train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])'''
    train_dataloader = DataLoader(full_dataset, batch_size=1, shuffle=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    #classifier = Classifier(input_dim=bert_model.config.hidden_size, num_classes=29)
    classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=29)
    classifier.to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # TRAINING
    epochs = 15
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        for embeddings, labels in train_dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()



    # EVALUATION
    with open('../data/data.json') as f:
        data = json.load(f)

    original_dataset = []
    for item in data:
        text = item['query']  # Assuming 'query' contains the text
        embedding = extract_embeddings([text], bert_model, tokenizer, device)
        original_dataset.append({
            'embedding': embedding.squeeze(0).cpu().numpy(),  # Convert embedding to numpy array
            'label': item['label']
        })
    test_dataset = TestDataset(original_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # Evaluation loop
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for embeddings, labels in test_dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = classifier(embeddings)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    print(f"Original Dataset Test Accuracy: {correct / total:.4f}")

    noises.append(NOISE_LEVEL)
    accuracies.append(correct / total)
    augmentations.append(NUM_AUGMENTATIONS)
    NOISE_LEVEL += 0.2
    #NUM_AUGMENTATIONS += 50

import matplotlib.pyplot as plt
plt.ylim(0, 1)
plt.xlim(0.1, 3)
plt.plot(noises, accuracies)
plt.xlabel('Split Ratios')
plt.ylabel('Accuracies')
plt.title('Accuracies for different split ratios')
plt.show()
