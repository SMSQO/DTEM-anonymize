#!/usr/bin/env python3

from llama3_embedding import Llama3Embedder
from typing import *
import json
from tqdm import tqdm
import os
import pickle
import numpy as np 

class CodeSnippetEmbedding: 
    
    def __init__(self, device='cuda:0'): 
        self.model = Llama3Embedder(device)

    @staticmethod
    def code_snippet_seq(filename: str) -> Iterable[dict]: 
        with open(f'{SRC_DIR}/{filename}') as fp: 
            for line in fp: 
                # line: {project, path, code, lang}
                yield json.loads(line)


    @staticmethod
    def push_embedding(ret: Dict, project: str, path: str, embedding: np.ndarray, weight: int=1) -> Dict:
        '''
        push an embedding to a Dict, 
        where Dict = JSON <| { $repo_name: { $path: [ Vector<4096>, int ] } }
        '''
        size = embedding.shape

        proj = ret[project] = (ret.get(project) or {})
        pth = proj[path] = (proj.get(path) or [np.zeros(size, dtype=np.float32), 0])
        pth[0] += embedding; pth[1] += weight

        return ret

    def code_snippet_embedding(self, filename: str, ret: Dict) -> Dict: 

        # region get file `size`
        # not necessary, but can be useful to see the progress
        if not hasattr(self, '_filelength'): 
            with open('./data/code_snippet/filelength.json') as fp: 
                self._filelength = json.load(fp)
        size: int = int(self._filelength[filename])
        # endregion


        for it in tqdm(self.code_snippet_seq(filename), total=size): 
            project, path, code = it['project'], it['path'], it['code']
            embedding = self.model(code)
            # embedding = np.ones((4096,), dtype=np.float16)  # MOCK
            self.push_embedding(ret, project, path, embedding)
        
        return ret 



SRC_DIR = './data/code_snippet/codes'
DST_DIR = './data/code_snippet/result'

class ProecessManager(): 
    def __init__(self, device): 
        all_files = os.listdir(SRC_DIR)
        tried_files = [it for it in os.listdir(DST_DIR)]

        self.files = [
            it for it in all_files if 
            it not in [f'{it.split(".")[0]}.jsonl' for it in tried_files]
        ]
        self.klee = CodeSnippetEmbedding(device)

    def next(self): 
        if self.files == []: 
            print('all tasks finished/tried.')
            return 
        
        file = self.files[0]

        assert not os.path.exists(f'{DST_DIR}/{file.split(".")[0]}.pkl')

        with open(f'{DST_DIR}/{file.split(".")[0]}.pkl', 'wb') as fp: pass

        # for it in self.files: 
        #     option = input(f'select this? "{it}"? [Y/n] ')
        #     if option.upper() == 'Y': 
        #         file = it
        #         break
        
        print(file)
        result = {}
        try: 
            self.klee.code_snippet_embedding(file, result)
        finally: 
            result = dict(result)
            with open(f'{DST_DIR}/{file.split(".")[0]}.pkl', 'wb') as fp:
                pickle.dump(result, fp)  # JSON <| *[$project][$path] = $embedding


    # never tested
    def merge_embeddings(self) -> Dict: 

        def item_seq(filepathes: List[str]): 
            for i, filepath in enumerate(filepathes): 
                print(f'[{i}/{len(filepathes)}] {filepath}')
                with open(filepath, 'rb') as fp: 
                    data = pickle.load(fp)

                for proj_k, proj_v in tqdm(data): 
                    for path_k, path_v in proj_v: 
                        yield proj_k, path_k, path_v
        
        all_files = [f'{SRC_DIR}/{it.split(".")[0]}.pkl' for it in os.listdir(SRC_DIR)]
        ret = {}
        for project, path, (embed, weight) in item_seq(all_files): 
            self.klee.push_embedding(ret, project, path, embed, weight)

        # OK, merging vectors. ([vector, int] => vector)
        ret = {
            proj_k: {
                path_k: path_v[0] / path_v[1]
                for path_k, path_v in proj_v.items()
            }
            for proj_k, proj_v in tqdm(ret.items())
        }

        return ret  # JSON <| { $project: { $path: Vector<4096> } }


if __name__ == '__main__': 
    opt = input('Using which gpu? [0/1] ')
    device = {
        '0': 'cuda:0',
        '1': 'cuda:1'
    }[opt]
    klee = ProecessManager(device=device)
    klee.next()

    exit(0)

    # TODO WARNING: NEVER TESTED! (code below)
    # this code is used to merge all embeddings (of different languages) into one file
    # Since the file is huge, OOM may occur during the gneration.
    # Remember to recheck(read-through) 'klee.merge_embeddings()' before executing it.

    ret = {'hello': 'world'}  # TODO test output file path first
    # ret = klee.merge_embeddings()  
    with open('data/code_snippet/code_snippet.pkl', 'wb') as fp: 
        pickle.dump(fp)

