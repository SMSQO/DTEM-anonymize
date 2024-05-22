from torch.utils.data import DataLoader
from typing import *
from general.model_and_metric import \
        EmbeddingLoader, collate_fn, train

from torch.utils.data import Dataset
import json
import torch

class MyDataset(Dataset):
    def __init__(self, samples, repo_node_embedding, contributor_embedding) -> None:
        super().__init__()
        # self.data = []
        if isinstance(samples, str):
            with open(samples, "r", encoding="utf-8") as inf:
                samples = json.load(inf)

        self.samples = samples
        self.embeddings = [repo_node_embedding, contributor_embedding]

    
    def __len__(self):
        return 2 * len(self.samples)
    
    def __getitem__(self, idx):
        idx, pn = divmod(idx, 2)
        sample = self.samples[idx]
        assert len(sample) == 3

        contributor_idx, pos_repo_idx, neg_repo_idx = sample
        repo_emb, contributor_emb = self.embeddings

        return [
            torch.cat([
                contributor_emb[contributor_idx], 
                repo_emb[pos_repo_idx]], dim=-1), 
            1
        ] if pn == 0 else [
            torch.cat([
                contributor_emb[contributor_idx], 
                repo_emb[neg_repo_idx]], dim=-1),
            0
        ]

# deprecated
def main_classical(): 
    embedding_loader = EmbeddingLoader()

    repo_emb, contributor_emb = [
        embedding_loader.load_embedding(it)
        for it in ['repository', 'contributor']
    ]

    print('Preparing dataloader...')
    lisa = ['train', 'test', 'valid']
    lisa = map(lambda it: f'./data/{it}.json', lisa)
    lisa = map(lambda it: MyDataset(
                samples=it, 
                repo_node_embedding=repo_emb, 
                contributor_embedding=contributor_emb
            ), lisa)
    lisa = map(lambda it: DataLoader(
                it, 
                batch_size=32, 
                shuffle=True, 
                collate_fn=collate_fn
            ), lisa)
    dl_train, dl_test, dl_eval = lisa  # dl: dataloader

    train(
        14_336 + 5376, 
        dl_train, 
        dl_test, 
        dl_eval, 
        model_path='./bin/model_llm_textonly.bin',
        device='cuda:0'
    )


if __name__ == "__main__":
    embedding_loader = EmbeddingLoader(device='cuda:1')

    repo_emb, contributor_emb = [
        embedding_loader.load_embedding(it)
        for it in ['repository', 'contributor']
    ]

    ret = []

    for i in range(10): 
        print(f'epoch [{i+1}/10] ...')
        print('Preparing dataloader...')
        lisa = ['train', 'test']
        lisa = map(lambda it: f'./data/10fold/{it}{i}.json', lisa)
        lisa = map(lambda it: MyDataset(
                    samples=it, 
                    repo_node_embedding=repo_emb, 
                    contributor_embedding=contributor_emb
                ), lisa)
        lisa = map(lambda it: DataLoader(
                    it, 
                    batch_size=32, 
                    shuffle=True, 
                    collate_fn=collate_fn('cuda:0')
                ), lisa)
        dl_train, dl_test = lisa  # dl: dataloader

        precision, recall, f1 = train(
            14_336 + 5376, 
            dl_train, 
            dl_test, 
            None, 
            model_path='./bin/model_llm_textonly.bin',
            device='cuda:1',
            result_file='result_fold10.txt'
        )
        ret.append((precision, recall, f1))

    avg_prec, avg_recall, avg_f1 = map(lambda it: sum(it) / len(it), zip(*ret))
    print(f'Average precision: {avg_prec}, recall: {avg_recall}, f1: {avg_f1}')
    
