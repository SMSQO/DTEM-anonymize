#!/usr/bin/env python3

exit(0)

from Comparisons.experiments.general import \
    load_yaml_cfg, github_token, \
    data_divide
import os
from tqdm import tqdm
import sys
from github import Github


cfg = load_yaml_cfg()['alpha']['collect_data']

possible_readme_names = [
    "Readme.md", 
    "README.md",        "readme.md",
    "README.textile",   "readme.textile", 
    "README.adoc",      "readme.adoc",
    "README.rst",       "readme.rst",
    "README.txt",       "readme.txt",
    "README",           "readme",
    "README.markdown",  "readme.markdown",
    "README.html",      "readme.html", 
    "README.htm",       "readme.htm",
]  # copied from `GHCrawler/crawl_repo_readme.py`


def crawl(idx: int, gh_token_idx=-1, total=4):

    repos = list(data_divide([
        it.strip().replace('#', '/') 
        for it in open(cfg['repo_without_readme_list_file'])],
    idx, total))

    klee = Github(github_token(gh_token_idx))

    all_readme_files = os.listdir(cfg['readme_directory'])

    for repo_name in tqdm(repos): 
        filename = f'{repo_name.replace("/", "#")}.md'
        if filename in all_readme_files: continue
        
        try: 
            repo = klee.get_repo(repo_name)
        except: continue


        for readme_name in possible_readme_names: 
            try: 
                print(f'trying {repo_name} => {readme_name}')
                content = repo.get_contents(readme_name)\
                        .decoded_content.decode('utf-8')

                with open(os.path.join(cfg['readme_directory'], filename), 'w') as fp: 
                    fp.write(content)
                break
            except KeyboardInterrupt: exit(0)
            except: continue


crawl(*map(int, sys.argv[1:]))
