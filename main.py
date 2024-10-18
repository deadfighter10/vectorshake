import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from pprint import pprint
import json

# import data from json file
with open('data/data.json') as f:
    data = json.load(f)

# sort data by label
data = sorted(data, key=lambda x: int(x['label']))
pprint(data)

nltk.download('punkt')
from nltk.tokenize import word_tokenize
exit()

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / (self.embed_dim ** 0.5), dim=2)
        return torch.bmm(attn_weights, V)


class CommandClassifier(nn.Module):
    def __init__(self, embed_dim, action_embeddings):
        super(CommandClassifier, self).__init__()
        self.attention = SelfAttention(embed_dim)
        self.action_embeddings = action_embeddings  # Predefined action embeddings

    def forward(self, x):
        x = self.attention(x)
        x = x.mean(dim=1)
        similarities = F.cosine_similarity(x.unsqueeze(1), self.action_embeddings, dim=2)
        return similarities


def get_embedding(sentence):
    tokens = word_tokenize(sentence.lower())
    embeds = [glove[token] for token in tokens if token in glove.stoi]
    if embeds:
        return torch.stack(embeds)
    else:
        return torch.zeros((1, glove.dim))


# Define action embeddings
actions = ["Switch off the lamp", "Turn on the computer"]
action_embeddings = []
for action in actions:
    embeds = get_embedding(action)
    action_embedding = embeds.mean(dim=0)
    action_embeddings.append(action_embedding)
action_embeddings = torch.stack(action_embeddings)  # Shape: [num_actions, embed_dim]
action_embeddings = action_embeddings.unsqueeze(0)  # Shape: [1, num_actions, embed_dim]

# Instantiate the model
model = CommandClassifier(embed_dim=300, action_embeddings=action_embeddings)

# Example commands
commands = [
    "I'm tired, I'm going to sleep",
    "I'm home, I have to work",
    "Feeling exhausted, time to rest",
    "Need to finish the project tonight"
]

for command in commands:
    input_embeds = get_embedding(command).unsqueeze(0)  # Shape: [1, sequence_length, embed_dim]
    with torch.no_grad():
        similarities = model(input_embeds)
    predicted_action = torch.argmax(similarities, dim=1).item()
    print(f"Command: '{command}'")
    print(f"Predicted Action: {actions[predicted_action]}\n")
