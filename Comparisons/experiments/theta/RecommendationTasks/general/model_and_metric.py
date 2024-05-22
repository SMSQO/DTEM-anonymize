#!/usr/bin/env python3

import json

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import *

import numpy as np


class Net(nn.Module):
    def __init__(self, total_emb_dim: int):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(total_emb_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.sigmoid(x).squeeze()

def collate_fn(device='cpu'):
    def ret(batch): 
        samples = []
        labels = []
        for sample in batch:
            samples.append(sample[0])
            labels.append(sample[1])
        return torch.stack(samples).to(device), torch.LongTensor(labels).to(device)
        # return torch.FloatTensor(samples), torch.LongTensor(labels)

def metric(pos_right, neg_right, pos_total, neg_total) -> Tuple[float, float, float]:
    precision = (pos_right) / (pos_right + neg_total - neg_right)
    recall = (pos_right) / (pos_right + pos_total - pos_right)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

class EmbeddingLoader(): 

    FILEPATH_MAP = {                                            #   count x    dim
        'contributor':  'contributor_embedding/embedding.pt',   # 394,474 x 14,336
        'repository':   'repo_embedding/embedding.pt',          #  50,000 x  5,376
        'pr':           'pr_embedding/embedding.pt',            # 379,496 x  4,864
        'issue':        'issue_embedding/embedding.pt',         # 692,554 x  4,096
    }

    def __init__(self, base_path: str='embeddings', device='cpu'): 
        self.base_path = base_path
        self.device = device

    def load_embedding(self, name: str) -> torch.FloatTensor:
        print(f'Loading embedding for {name}...')
        return torch.load(f'{self.base_path}/{self.FILEPATH_MAP[name]}')

def train(total_dim: int, dl_train, dl_test, dl_eval, *, model_path: str, device='cpu', result_file='train_nn_result.txt') -> Tuple[int, int, int]: 
    fout = open(result_file, 'w')
    print('start training...')
    model = Net(total_emb_dim=total_dim).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    best_f1 = 0
    epochs = 60

    # Training loop
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        model.train()
        for _, (x, labels) in tqdm(enumerate(dl_train), total=len(dl_train)):
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}: loss={running_loss / len(dl_train)}", file=fout, flush=True)

        # Validation
        if dl_eval is not None: 
            model.eval()
            pos_rights = 0
            neg_rights = 0
            pos_totals = 0
            neg_totals = 0

            with torch.no_grad():
                for _, (x, labels) in enumerate(dl_eval):
                    x, labels = x.to(device), labels.to(device)
                    logits = model(x)
                    pred = (logits > 0.5).long()
                    pos_right = ((pred == 1) & (labels == 1)).sum().item()
                    neg_right = ((pred == 0) & (labels == 0)).sum().item()
                    pos_total = (labels == 1).sum().item()
                    neg_total = (labels == 0).sum().item()

                    pos_rights += pos_right
                    neg_rights += neg_right
                    pos_totals += pos_total
                    neg_totals += neg_total

            precision, recall, f1 = metric(pos_rights, neg_rights, pos_totals, neg_totals)
            print(f"Epoch {epoch+1}: precision={precision}, recall={recall}, f1={f1}", file=fout, flush=True)

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), model_path)
                print("Model saved.")

    if dl_eval is None: 
        torch.save(model.state_dict(), model_path)

    # Test  
    model.load_state_dict(torch.load(model_path))
    model.eval()
    pos_rights = 0
    neg_rights = 0
    pos_totals = 0
    neg_totals = 0
    with torch.no_grad():
        for _, (x, labels) in enumerate(dl_test):
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            pred = (logits > 0.5).long()
            pos_right = ((pred == 1) & (labels == 1)).sum().item()
            neg_right = ((pred == 0) & (labels == 0)).sum().item()
            pos_total = (labels == 1).sum().item()
            neg_total = (labels == 0).sum().item()

            pos_rights += pos_right
            neg_rights += neg_right
            pos_totals += pos_total
            neg_totals += neg_total
    precision, recall, f1 = metric(pos_rights, neg_rights, pos_totals, neg_totals)
    print(f"Test: precision={precision}, recall={recall}, f1={f1}", file=fout, flush=True)
    fout.close()
    return precision, recall, f1

