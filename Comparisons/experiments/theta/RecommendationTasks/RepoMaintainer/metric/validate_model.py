import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import numpy as np

from general.model_and_metric import \
        EmbeddingLoader, collate_fn, train, Net

class MyDataset(Dataset):
    def __init__(self, repo_idx, contributor_idxs, repo_embedding, contributor_embedding, is_tensor=True) -> None:
        super().__init__()
        self.data = []
        # print(repo_idx, contributor_idxs)
        src_embedding = repo_embedding[repo_idx]
        if is_tensor:
            src_embedding = src_embedding.cpu().numpy().tolist()
        for contributor_idx in contributor_idxs:
            dst_embedding = contributor_embedding[contributor_idx]
            if is_tensor:
                dst_embedding = dst_embedding.cpu().numpy().tolist()
            self.data.append([contributor_idx, src_embedding + dst_embedding])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    samples = []
    contributor_idxs = []
    for sample in batch:
        contributor_idxs.append(sample[0])
        samples.append(sample[1])
    return torch.FloatTensor(samples), torch.LongTensor(contributor_idxs)


if __name__ == "__main__":
    trained_model_path = "../bin/model_llm_textonly.bin"
    feat_size = 14_336 + 5376
    validation_dataset = "./data/dataset_valid_test.json"
    dst_result_path = "./result/model_result_valid_test.json"
    trained_embedding_path = "../../../GNN/HetSAGE/node_embedding/HetSAGE_node_embedding.bin"


    device = 'cuda:1'
    embedding_loader = EmbeddingLoader(device=device, base_path='../embeddings')

    contributor_embedding, repo_embedding = [
        embedding_loader.load_embedding(it)
        for it in ['contributor', 'repository']
    ]

    model = Net(total_emb_dim=feat_size)
    model.load_state_dict(torch.load(trained_model_path, map_location="cpu"))

    with open(validation_dataset, "r", encoding="utf-8") as inf:
        dataset = json.load(inf)
    
    topks = {}
    for repo_idx, search_scope, labels in dataset:
        if len(search_scope) < 10:
            continue
        d = MyDataset(repo_idx, search_scope, repo_embedding, contributor_embedding)
        dataloader = DataLoader(d, batch_size=128, shuffle=False, collate_fn=collate_fn)
        model.eval()
        output = {}
        with torch.no_grad():
            for batch in dataloader:
                samples, contributor_idxs = batch
                results = model(samples).squeeze().numpy()
                if len(results.shape) == 0:
                    results = [results]
                for idx, result in zip(contributor_idxs, results):
                    output[idx] = result
        
        output = sorted(output.items(), key=lambda x: x[1], reverse=True)
        output = [x[0] for x in output[:20]]
        # print(output)
        rets = np.zeros(21).tolist()
        for i in range(1, 21):
            rets[i] = rets[i - 1]
            if i-1 < len(output) and output[i - 1] in labels:
                rets[i] = rets[i - 1] + 1

        topks[repo_idx] = (rets, search_scope)
    
    with open(dst_result_path, "w", encoding="utf-8") as ouf:
        json.dump(topks, ouf, indent=4)
