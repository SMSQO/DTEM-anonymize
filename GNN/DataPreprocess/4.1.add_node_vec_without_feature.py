#!/usr/bin/env python3

import traceback


import time
from datetime import datetime

import os
import sys
import json

import torch
import dgl

from dgl import load_graphs
from dgl.data.utils import save_graphs

from utils import RepositoryFeatureLoader, IssueFeatureLoader, PRFeatureLoader


structure_graph_file            = "./full_graph/structure_graph.bin"
dst_graph_file                  = "./full_graph/structure_graph_without_feature.bin"

if __name__ == '__main__':

    device = torch.device("cpu")

    # expertise features: [name], [size] and [dim]
    expertise_features = {
        'pr':           (379496, 1536),
        'issue':        (692554, 768),
        'repository':   (50000,  2048),
        'contributor':  (394474, 0),
    }

    hg = load_graphs(structure_graph_file)[0][0]
    for name, size in expertise_features.items(): 
        hg.nodes[name].data['feat'] = torch.randn((size[0], size[1] + 256), device=device)

    save_graphs(dst_graph_file, [hg])
