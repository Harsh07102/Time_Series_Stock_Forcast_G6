import os
import json
import logging
import numpy as np
import random

def setup_logging(log_path='logs/pipeline.log'):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path='config/config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
