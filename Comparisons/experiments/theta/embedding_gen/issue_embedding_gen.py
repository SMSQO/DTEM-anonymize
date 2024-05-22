#!/usr/bin/env python3

'''
From theta. 
Remember to compare to experiment alpha.
'''

from config import cfg
from typing import *
import json
from llama3_embedding import Llama3Embedder
from rich import print as rprint
import pickle
from tqdm import tqdm
import os
from utils import kv_emb_to_tensor
import numpy as np 
import torch

class IssueDict(TypedDict): 
    name:    str  # e.g. datalux/osintgram#670
    title:   str
    content: str 

class RepoDict(TypedDict): 
    name:   str
    tags:   List[str]
    topic:  str
    readme: str       

ElemType = Literal['index', 'issue', 'devloper', 'repository']

class IssueEmbeddings(): 

    def __init__(self): 
        self.model = Llama3Embedder()
        pass

    # region # basic works

    _idx_2_name: Dict[int, str] = None
    _name_2_issue: Dict[str, IssueDict] = None


    def _ensure_2_tables(self): 
        def load_tbl(): 
            gcfg = cfg['general']

            # step 1: load _name_2_issue
            path = gcfg['basepath'] + gcfg['filepath']['issue_idx_file']
            with open(path) as fp: 
                self._idx_2_name = {v: k for k, v in json.load(fp).items()}

            # step 2.1: load name_2_title
            def name_2_title(): 
                path = gcfg['basepath'] + cfg['alpha']['raw']['issue_title_file_path']
                with open(path) as fp: 
                    for line in fp: 
                        it = json.loads(line)
                        yield it['name'], it['title']
            name_2_title = dict(name_2_title())

            # step 2.2: load _name_2_issue
            def tbl(): 
                path = gcfg['basepath'] + gcfg['filepath']['issue_content_file']
                with open(path) as fp: 
                    for line in fp: 
                        it = json.loads(line)
                        issue_name = f"{it['project']}#{it['number']}"
                        if issue_name == 'friendsofphp/security-advisories#666':
                            print(f'loading table: found name: friendsofphp/security-advisories#666')
                        issue = {
                            'name':     issue_name,
                            'title':    name_2_title.get(issue_name, ''),
                            'content':  it['text'],
                        }

                        if issue['title'] is None: issue['title'] = ''
                        if issue['content'] is None: issue['content'] = ''
                        yield issue_name, issue

            self._name_2_issue = {k: v for k, v in tbl()}  # 687358 < 692554, some issues don't have a dict.

        if self._idx_2_name is None or \
                self._name_2_issue is None: 
            load_tbl()

    def index_to_issue(self, index: int) -> IssueDict: 
        self._ensure_2_tables()
        name = self._idx_2_name.get(index)
        if name is None: 
            print(f'warning: can\'t find name for index "{index}"')
            return None
        ret = self._name_2_issue.get(name)
        if ret is None: 
            print(f'warning: can\'t find issue_dict for name "{name}"')
        return ret
    
    # endregion

    def get_issue_embedding(self, issue_idx): 
        issue: IssueDict = self.index_to_issue(issue_idx)
        if issue is None: 
            return None
        issue_corpus = '\n\n'.join([issue['name'], issue['title'], issue['content']])
        return self.model(issue_corpus)
    
    @property
    def all_issue_embedding_filepath(self): 
        gcfg = cfg['general']
        return gcfg['basepath'] + cfg['theta']['filepath']['issue_embeddings']
    
    def generate_all_issue_embeddings(self, force=False): 
        '''
        Side effect: generate issue embedding file: JSON <| { $issue_name: Tensor<4096> }
        @see also: /Comparison/experiments/config.yaml, theta.filepath.issue_embeddings
        '''
        # phase 1: check if the file exists
        filepath = self.all_issue_embedding_filepath
        if os.path.exists(filepath) and not force: 
            print('Issue embedding file exists.')
            return 

        # phase 2: get embeddings of all issues
        self._ensure_2_tables()

        issue_embeddings = {
            issue_name: self.get_issue_embedding(issue_idx)
            # for (issue_idx, issue_name), _ in tqdm(zip(self._idx_2_name.items(), range(100)))
            for issue_idx, issue_name in tqdm(self._idx_2_name.items())
        }
        print(len(issue_embeddings))
        print(len([it for it in issue_embeddings.values() if it is None]))
        issue_embeddings = {key: value for key, value in issue_embeddings.items() if value is not None}

        # phase 3. Now, output to files.
        with open(filepath, 'wb') as fp: 
            pickle.dump(issue_embeddings, fp)

        print('OK, new issue embedding file generated.')

    def post_precess(self): 
 
        FILE_PREFIX = 'data/issue_embedding'
        with open(f'{FILE_PREFIX}/issue_embeddings.pkl', 'rb') as fp: 
            data = pickle.load(fp)
        data = {key: value for key, value in data.items() if value is not None}

        def load_issue_idx() -> Dict[str, int]: 
            '''issue_name to issue_idx'''
            if not hasattr(self, '_issue_idx'): 
                with open(f'{FILE_PREFIX}/issue_idx.json') as fp: 
                    self._issue_idx = json.load(fp)
            return self._issue_idx

        issue_idx = load_issue_idx()
        issue_embedding = kv_emb_to_tensor(data, issue_idx)
        issue_embedding = torch.from_numpy(issue_embedding)
        print('Handled. Now save to file.')
        torch.save(issue_embedding, f'{FILE_PREFIX}/embedding.pt')
    

if __name__ == '__main__': 
    klee = IssueEmbeddings()
    # klee.generate_all_issue_embeddings(force=True)
    klee.post_precess()