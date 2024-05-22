#!/usr/bin/env python3

# Author: XHZ
# DELETE READY

import json

with open('user_watch_repos.json') as fp: 
    data = json.load(fp)

print(len(data))

