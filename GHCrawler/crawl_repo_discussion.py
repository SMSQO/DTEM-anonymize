import requests 
import json
from shutil import copyfile
import time
import datetime
import os
import sys
import copy

from path import SELECTED_REPO_PATH 


dst_filename = 'repo_discussion.txt'
token = ''

headers = {
    "Authorization": f"Bearer {token}",
}
url = "https://api.github.com/graphql"


def dump_repo_discussion(repo_name, cnt, discussion):
    with open('./rawdata/' + dst_filename, 'a') as opfile:
        opfile.write("{}\t{}\t{}\n".format(repo_name, cnt, json.dumps(discussion, ensure_ascii=False)))


repo_names = []
with open(SELECTED_REPO_PATH, 'r', encoding="utf-8") as inf:
    repo_names = json.load(inf)
     
crawled_repo_names = []
with open('./rawdata/repo_discussion.txt', 'r') as f:
    for line in f:
        repo_name, cnt, _ = line.split("\t")
        crawled_repo_names.append(repo_name)
print(len(crawled_repo_names))

for i, repo_name in enumerate(repo_names):
    if repo_name in crawled_repo_names:
        continue
    
    print("Crawling repo %d %s" % (i, repo_name), end=' ')
    owner, repo = repo_name.split('/')
    p = 0
    diss = []
    flt = 0
    discussion_cnt = 0
    query = '''
    {
        repository(owner: "%s", name: "%s") {
            discussions {
                totalCount
            }
        }
    }
    ''' % (owner, repo)
    response = requests.post(url, headers=headers, json={'query': query})
    if not 'json' in response.headers['Content-Type']:
        crt = 0
        flt = 1
        while "We couldn't respond to your request in time" in response.text or "We had issues producing the response to your request":
            time.sleep(1)
            response = requests.post(url, headers=headers, json={'query': query})
            crt += 1
            flt = 0
            if crt > 10:
                print('Error:', i, owner, repo,"couldn't respond or had issues")
                flt = 1
                break
        if flt:
            dump_repo_discussion(repo_name, discussion_cnt, diss)
            continue
        
    # while 'API rate limit exceeded' in response.text:
    while 'errors' in response.json() and response.json()['errors'][0]['type'] == 'RATE_LIMITED':
        now_time = datetime.datetime.now()
        print(i, now_time.strftime('%H:%M:%S'))
        sys.stdout.flush()
        retime = response.headers['x-ratelimit-reset']
        notime = time.time()
        time.sleep(int(retime) - int(notime) + 1)
        response = requests.post(url, headers=headers, json={'query': query})
    if response.status_code != 200:
        print('Error:', i, owner, repo, response.text)
        dump_repo_discussion(repo_name, discussion_cnt, diss)
        continue

       
    try:
        discussion_cnt = response.json()['data']['repository']['discussions']['totalCount']
    except Exception as e:
        print(i, owner, repo, e, response.text)
        dump_repo_discussion(repo_name, discussion_cnt, diss)
        continue
    
    print("get %d discussions" % discussion_cnt)
    sys.stdout.flush()
    if discussion_cnt == 0:
        dump_repo_discussion(repo_name, discussion_cnt, diss)
        continue

    query = '''
    query {
        repository(owner: "%s", name: "%s") {
            discussions(first: 100) {
                nodes {
                    title
                    body
                    author {
                        login
                    }
                    category {
                        name
                    }
                    closed
                    comments(first: 100) {
                        nodes {
                            body
                            author {
                                login
                            }
                        }
                    }
                    answer {
                        body
                        author {
                            login
                        }
                    }
                    createdAt
                    number
                    labels(first: 100) {
                        nodes {
                            name
                        }
                    }
                    stateReason
                }
                pageInfo {
                    endCursor
                    hasNextPage
                }
            }
        }
    }
    ''' % (owner, repo)
    
    while True:
        if p != 0:
            query = '''
            query {
                repository(owner: "%s", name: "%s") {
                    discussions(first: 100, after: "%s") {
                        nodes {
                            title
                            body
                            author {
                                login
                            }
                            category {
                                name
                            }
                            closed
                            comments(first: 100) {
                                nodes {
                                    body
                                    author {
                                        login
                                    }
                                }
                            }
                            answer {
                                body
                                author {
                                    login
                                }
                            }
                            createdAt
                            number
                            labels(first: 100) {
                                nodes {
                                    name
                                }
                            }
                            stateReason
                        }
                        pageInfo {
                            endCursor
                            hasNextPage
                        }
                    }
                }
            }
          ''' % (owner, repo, response.json()['data']['repository']['discussions']['pageInfo']['endCursor'])
        response = requests.post(url, headers=headers, json={'query': query})
        p += 1
        if not 'json' in response.headers['Content-Type']:
            crt = 0
            flt = 1
            while "We couldn't respond to your request in time" in response.text or "We had issues producing the response to your request":
                time.sleep(1)
                response = requests.post(
                    url, headers=headers, json={'query': query})
                crt += 1
                flt = 0
                if crt > 10:
                    print('Error:', i, owner, repo,
                          "couldn't respond or had issues")
                    flt = 1
                    break
            if flt:
                break
            
        # while 'API rate limit exceeded' in response.text:
        while 'errors' in response.json() and response.json()['errors'][0]['type'] == 'RATE_LIMITED':
            now_time = datetime.datetime.now()
            print(i, now_time.strftime('%H:%M:%S'))
            sys.stdout.flush()
            retime = response.headers['x-ratelimit-reset']
            notime = time.time()
            time.sleep(int(retime) - int(notime) + 1)
            response = requests.post(
                url, headers=headers, json={'query': query})
        if response.status_code != 200:
            print('Error:', i, owner, repo, response.text)
            flt = 1
            break
                
        try:
            data = response.json()['data']['repository']['discussions']['nodes']
        except Exception as e:
            print(i, owner, repo, e, response.text)
            flt = 1
            break
        
        for node in data:
            if node['author']:
                node['author'] = node['author']['login']
            node['category'] = node['category']['name']
            comts = node['comments']['nodes']
            for comt in comts:
                if comt['author']:
                    comt['author'] = comt['author']['login']
            node['comments'] = comts
            node['labels'] = [label['name']
                              for label in node['labels']['nodes']]
            if node['answer']:
                if node['answer']['author']:
                    node['answer']['author'] = node['answer']['author']['login']
            diss.append(node)
        if not response.json()['data']['repository']['discussions']['pageInfo']['hasNextPage']:
            break
    if flt:
        dump_repo_discussion(repo_name, discussion_cnt, diss)
        continue
    
    dump_repo_discussion(repo_name, discussion_cnt, diss)
        
