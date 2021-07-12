import pickle 
import json
import os
import random
from collections import namedtuple

def save(obj, obj_path):
    with open(obj_path, 'wb') as fout:
        pickle.dump(obj, fout)

def load(obj_path):
    with open(obj_path, 'rb') as fin:
        fin.seek(0)
        obj = pickle.load(fin)
    return obj

def listdir_nohidden(input):
    """ Lists files in a dir, ignoring hidden files that can cause issues """
    dirlist = os.listdir(input)
    dirlist = [dir for dir in dirlist if not dir.startswith('.')]
    return dirlist

def test_train_split(items, test_split, seed=123):
    assert test_split >=  0
    assert test_split <=  1

    random.Random(seed).shuffle(items)
    test_split_idx = int(len(items) * test_split)
    test = items[:test_split_idx]
    train = items[test_split_idx:]
    return train, test

def test_train_val_split(items, test_split, val_split, seed=123):
    assert test_split >=  0
    assert test_split <=  1
    assert val_split >=  0
    assert val_split <=  1

    random.Random(seed).shuffle(items)
    test_split_idx = int(len(items) * test_split)
    val_split_idx = int(len(items) * (test_split + val_split))
    test = items[:test_split_idx]
    validation = items[test_split_idx:val_split_idx]
    train = items[val_split_idx:]

    return train, test, validation

def config_init(argv):
    ### Initialize config file ###
    if len(argv) != 2:
        raise ValueError("ERROR, need config file. Ie: python generate_data.py config.json")    
    configuration_file = str(argv[1])
    with open(configuration_file, 'r') as f:
        f = f.read()
        PARAMS = json.loads(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    return PARAMS

