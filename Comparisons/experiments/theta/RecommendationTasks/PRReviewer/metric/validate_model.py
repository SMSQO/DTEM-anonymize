import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import numpy as np

from general.model_and_metric import \
        EmbeddingLoader, collate_fn, train, Net

class MyDataset(Dataset):
    def __init__(self, repo_idx, pr_idx, contributor_idxs, repo_embedding, pr_embedding, contributor_embedding, is_tensor=True) -> None:
        super().__init__()
        self.data = []
        src_embedding = repo_embedding[repo_idx]
        if is_tensor:
            src_embedding = src_embedding.cpu().numpy().tolist()
        mid_embedding = pr_embedding[pr_idx]
        if is_tensor:
            mid_embedding = mid_embedding.cpu().numpy().tolist()
        for contributor_idx in contributor_idxs:
            dst_embedding = contributor_embedding[contributor_idx]
            if is_tensor:
                dst_embedding = dst_embedding.cpu().numpy().tolist()
            self.data.append([contributor_idx, src_embedding + mid_embedding + dst_embedding])

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


def validate_model(
        total_dim: int, 
        model_filepath: str, 
        *, 
        repo_embedding,
        pr_embedding,
        contributor_embedding,
        device='cpu'
): 
    with open('data/dataset_valid_test_modified.json') as inf:
        dataset = json.load(inf)

    model = Net(total_emb_dim=total_dim)
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    
    topks = {}
    for repo_idx, pr_idx, search_scope, labels \
            in tqdm(dataset, total=len(dataset)):
        if len(search_scope) < 10: continue

        d = MyDataset(repo_idx, pr_idx, search_scope, repo_embedding, pr_embedding, contributor_embedding)
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

    return topks
    

if __name__ == "__main__":
    device = 'cuda:1'
    embedding_loader = EmbeddingLoader(device=device, base_path='../embeddings')

    repo_emb, contributor_emb, pr_emb = [
        embedding_loader.load_embedding(it)
        for it in ['repository', 'contributor', 'pr']
    ]

    print('Validating model...')
    topks = validate_model(
        14_336 + 5376 + 4864, 
        '../bin/model_llm_textonly.bin',
        repo_embedding=repo_emb, 
        pr_embedding=pr_emb, 
        contributor_embedding=contributor_emb,
        device=device
    )

    with open('./result/model_result_valid_test_modified.json', 'w') as ouf:
        json.dump(topks, ouf, indent=4)

