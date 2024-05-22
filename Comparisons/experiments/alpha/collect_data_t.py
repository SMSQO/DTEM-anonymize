#!/usr/bin/env python

from rich import print as rprint
from Comparisons.experiments.general import \
    ignore_exception, \
    load_yaml_cfg, \
    load_contributor_index, load_repository_index, load_issue_index, \
    dict_invert, dict_inspect, dict_invert_mul, \
    github_token, \
    load_jsonl, save_jsonl, \
    data_divide, \
    RepoDict
from typing import Dict, List, Any, Optional, TypedDict, Generator, Tuple
from github import Github
from github.Repository import Repository
from tqdm import tqdm
import os, sys

# Repository:   Name, Tags, Topics, README
# Issue:        Body & Title
# API:          (appear at least 5 times)

CONFIG_FILE_PATH = 'Comparisons/experiments/config.yaml'
cfg: Dict[str, Any] = load_yaml_cfg()['alpha']



class RepoDataCrawler(): 

    repos: Dict[str, RepoDict]
    repo_names: Dict[int, str]

    def __init__(self):
        self.github = Github(github_token(2))
        self.repo_names = dict_invert(load_repository_index())
        self.repos = {}

    def fetch_all(self): 
        stdout = cfg['raw']['repo_file_path']
        self.repos = { it['name']: it for it in load_jsonl(stdout) }

        scope: List[Any] = list(self.repo_names.items())

        try: 
            for _, repo_name in tqdm(scope): 
                if repo_name in self.repos: 
                    continue
                @ignore_exception
                def _():
                    self.get_repo_info(repo_name)
        finally: 
            save_jsonl(stdout, self.repos.values())


    def get_repo_info(self, repo_name: str) -> RepoDict: 
 
        if repo_name in self.repos: 
            return self.repos[repo_name]
        
        repo: Repository = self.github.get_repo(repo_name)

        repo = {
            'name':     repo_name,
            'tags':     self._get_repo_tag(repo),
            'topic':    self._get_repo_topic(repo),
        }

        self.repos[repo_name] = repo
        return repo


    def _get_repo_tag(self, repo: Repository) -> List[str]: 
        return repo.get_topics()


    def _get_repo_topic(self, repo: Repository) -> str: 
        return repo.description

class IssueDataCrawler: 

    repo_issue_indices: Dict[str, List[int]]

    def __init__(self): 
        ret = {}
        for issue_name in load_issue_index():
            tmp = issue_name.split('#')
            repo_name, issue_idx = tmp[0], int(tmp[1])

            if repo_name not in ret: 
                ret[repo_name] = []
            ret[repo_name].append(issue_idx)

        self.repo_issue_indices = ret

    def fetch_all(self, idx: int, total=4, gh_token_idx=None): 
        # jsonl file
        # elem looks like: {'name': 'datalux/osintgram#670', 'title': '...'}
        stdout = cfg['raw']['issue_title_file_path'] + f'.{idx}'
        klee = Github(github_token(gh_token_idx))
        crawled_issues: List[str] = \
            list(it['name'] for it in load_jsonl(stdout))
        
        data = list(data_divide(self.repo_issue_indices.items(), idx, total))
        for repo_name, issue_indices in tqdm(data): 
            ret = []
            try: 
                @ignore_exception
                def _():
                    repo: Repository = klee.get_repo(repo_name)
                    for idx in tqdm(issue_indices): 
                        issue_name = f'{repo_name}#{idx}'
                        if issue_name in crawled_issues: 
                            continue
                        # print(f'self.get_issue_title({repo}, {idx})')
                        title = self.get_issue_title(repo, idx)
                        ret.append({
                            'name':     issue_name,
                            'title':    title,
                        })
            finally:
                save_jsonl(stdout, ret, 'a')

    def get_issue_title(self, repo: Repository, issue_id: int) -> str:
        return repo.get_issue(issue_id).title

idx = int(sys.argv[1])

klee = IssueDataCrawler()
klee.fetch_all(idx, total=4, gh_token_idx=idx)
