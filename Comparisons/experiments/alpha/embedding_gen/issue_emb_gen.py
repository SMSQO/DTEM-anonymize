#!/usr/bin/env python3 

import json
from typing import Dict, List, Any, Optional, TypedDict, Generator, Tuple
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import tokenize
from gensim.parsing.preprocessing import *

from Comparisons.experiments.general import \
    IssueDict, load_issue_index, load_yaml_cfg, load_contributor_commit_repo, load_contributor_index, load_repository_index, \
    load_jsonl, \
    dict_invert

import numpy as np 
import re
import nltk
import os
import pickle
from tqdm import tqdm

cfg_general = load_yaml_cfg()['general']
cfg = load_yaml_cfg()['alpha']


'''
    1. contributor ->> issues 
    2. read two files, to get IssueDict
    3. handle unexist Issues. 
    4. other details to do. 
'''

class ContributorIssueEmbedding: 

    model:          Doc2Vec 
    proposals:      Dict[int, List[int]]
    issue_idx_rvt:  Dict[int ,str]
    issues:         Dict[str, IssueDict]

    def __init__(self): 

        # step 1. get self.model 
        self.load_model()


        # step 2. get self.issues (all issues here)
        issue_titles: Dict[str, str]  = { 
            it['name']: it['title']
            for it in load_jsonl(cfg['raw']['issue_title_file_path']) 
        }
        issue_content: List[Dict] = list(load_jsonl(cfg_general['filepath']['issue_content_file']))

        issues: Dict[str, IssueDict] = {}
        for it in issue_content: 
            issue_name = f"{it['project']}#{it['number']}"
            issue = {
                'name':     issue_name,
                'title':    issue_titles.get(issue_name, ''),
                'content':  it['text'],
            }
            if issue['title'] is None: issue['title'] = ''
            if issue['content'] is None: issue['content'] = ''
            issues[issue_name] = issue
        self.issues = issues

        # step 3. get the relationship between contributors and issues 
        # i.e. propose (a issue). Stored as self.proposals
        def issue_proposals():
            with open(cfg_general['filepath']['contributor_propose_issue_file']) as fp: 
                for line in fp:
                    lst = line.split('\t')
                    yield int(lst[0]), int(lst[1])
        
        proposals: Dict[int, List[int]] = {}
        for user_id, issue_id in issue_proposals():
            proposals.setdefault(user_id, [])
            proposals[user_id].append(issue_id)

        self.proposals = proposals


        self.contributor_idx: Dict[str, int] = load_contributor_index()
        self.issue_idx_rvt = dict_invert(load_issue_index())

    def load_model(self): 
        model = Doc2Vec.load(cfg['model']['dev2vec_issue_file_path'])   # output_vec_size: 150
        self.model = model


    def convert(self, user_name: str) -> np.ndarray: 

        # step 2: prepare issues

        proposals = self.proposals
        idx2issue = self.issue_idx_rvt
        user2idx  = self.contributor_idx

        user_idx = user2idx[user_name]
        issue_indices = proposals.get(user_idx) 
        if issue_indices is None: 
            return np.array(self.model.infer_vector([]))
        issues: List[str]       = (idx2issue[it] for it in issue_indices)
        issues: List[str]       = (it for it in issues if it is not None)
        issues: List[IssueDict] = (self.issues.get(it) for it in issues)
        issues: List[IssueDict] = [it for it in issues if it is not None]

        # step 4. collect the token generated by the user (|>tokens)
        tokens: List[List[str]] = []

        for issue in issues: 
            tokens.append(self._tokenize_issue(
                issue['title'],
                issue['content']
            ))

        # step 5. invoke the model, get the vector 
        return np.array(self.model\
                .infer_vector([it for l in tokens for it in l]))
    

    def _tokenize_issue(self,
            issue_title:    str, 
            issue_content:  str
    ) -> List[str]: 
        '''
            @see also: 
                Comparisons/experiments/alpha/embedding_gen/repo_emb_gen.py 
                        -> ContributorRepoEmbedding._tokenize_repo
        '''
        try: 
            words = ContributorIssueEmbedding.words 
        except: 
            words = ContributorIssueEmbedding.words \
                = set(nltk.corpus.words.words())  

        def newline(text): 
            return '\n'.join([p for p in re.split(r'\\n|\\r|\\n\\n|\\r\\n|\r\n', text) if len(p) > 0])

        text = '\n'.join([issue_title, newline(issue_content)])
        gen = text.split()
        gen = (it.lower() for doc in gen for it in tokenize(doc, lower=True))  # token stream

        tokens: List[str] = preprocess_string(
            ' '.join(list(gen)), 
            filters=[remove_stopwords, split_alphanum, strip_numeric, strip_tags, strip_non_alphanum])
        tokens = (it for it in tokens if it.lower() in words or not it.isalpha())
        tokens = (it for it in tokens if len(it) > 2)
        return list(tokens)


def main(): 
    klee = ContributorIssueEmbedding()

    devlopers = load_contributor_index().keys()

    result = {
        dev_name: klee.convert(dev_name)
        for dev_name in tqdm(devlopers)
    }

    with open(cfg['embedding']['contributor_issue_embedding'], 'wb') as fp: 
        pickle.dump(result, fp)