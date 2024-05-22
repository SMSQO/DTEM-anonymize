#!/usr/bin/env python

import pickle
import sys

def show_degrees(file: str):
    with open(file, 'rb') as f:
        print("loading file...")
        data = pickle.load(f)
        keys = list(data.keys())
        item = data[keys[0]]
        print(f'len(item) = {len(item)}')
        print(f'Size of data: {len(keys)}')


if __name__ == '__main__':
    show_degrees(sys.argv[1])