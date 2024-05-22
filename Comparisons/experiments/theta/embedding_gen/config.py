#!/usr/bin/env python3

from typing import *
import yaml

CONFIG_FILE_PATH = '../../config.yaml'

with open(CONFIG_FILE_PATH) as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)
