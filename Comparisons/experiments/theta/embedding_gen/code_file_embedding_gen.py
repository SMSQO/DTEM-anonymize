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
    def file_seq(lang_type: str) -> Iterable[Tuple[str, str, str]]:  # (project, path, code)
        files = os.listdir(f'{SRC_DIR}')
        files = filter(lambda it: it.startswith(lang_type), files)

        ret: Dict[str, Dict[str, str]] = {}  # {project: {path: code}}
        def push_snippet(ret: Dict, project: str, path: str, code: str): 
            if project not in ret: 
                ret[project] = {}
            if path not in ret[project]:
                ret[project][path] = ''
            ret[project][path] += ('\n\n' + code)
            
        for filename in tqdm(list(files), desc=f'Loading {lang_type} files...'): 
            with open(f'{SRC_DIR}/{filename}') as fp: 
                for line in fp: 
                    # it: {project, path, code, lang}
                    it = json.loads(line)
                    push_snippet(ret, it['project'], it['path'], it['code'])
        
        ret = [ 
            (proj_k, path_k, code)
            for proj_k, proj_v in ret.items()
            for path_k, code in proj_v.items()
        ]
        for proj, path, code in tqdm(ret): 
            yield proj, path, code


    # NOT CHECKED YET 
    @staticmethod
    def push_embedding(ret: Dict, project: str, path: str, embedding: np.ndarray) -> Dict:
        '''
        push an embedding to a Dict, 
        where Dict = JSON <| { $repo_name: { $path: Vector<4096> } }
        '''
        if project not in ret: 
            ret[project] = {}   
        if path not in ret[project]:
            ret[project][path] = np.zeros((4096,), dtype=np.float32)
        else: 
            print(f"WARN: {project} {path} already exists in the embedding dict.")
        return ret

    def code_file_embedding(self, ret: Dict, lang_type: str) -> Dict: 
        for proj, path, code in self.file_seq(lang_type):
            embedding = self.model(code)
            # embedding = np.ones((4096,), dtype=np.float16)  # TODO MOCK
            self.push_embedding(ret, proj, path, embedding)
        
        return ret 



SRC_DIR = './data/code_snippet/codes'
DST_DIR = './data/code_snippet/result'

# deprecated
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
    # deprecated? 
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


def embed_lang(lang: str, result_filepath: str, device: str): 
    print(f'Embedding {lang}...')
    if os.path.exists(result_filepath):
        return
    else: 
        with open(result_filepath, 'wb') as fp: 
            pass 
    embedder = CodeSnippetEmbedding(device)
    ret = embedder.code_file_embedding({}, lang)
    with open(result_filepath, 'wb') as fp: 
        pickle.dump(ret, fp)

import sys

if __name__ == '__main__':
    device = f'cuda:{sys.argv[1]}'
    lang_types = ['python', 'java', 'ruby', 'go', 'php', 'javascript']

    for lang in lang_types:
        embed_lang(lang, f'{DST_DIR}/{lang}.pkl', device)
