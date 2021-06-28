import pickle 
import json
import os
from collections import namedtuple

def save(obj, obj_path):
    with open(obj_path, 'wb') as fout:
        pickle.dump(obj, fout)

def load(obj_path):
    with open(obj_path, 'rb') as fin:
        obj = pickle.load(fin)
    return obj

def listdir_nohidden(input):
    """ Lists files in a dir, ignoring hidden files that can cause issues """
    dirlist = os.listdir(input)
    dirlist = [dir for dir in dirlist if not dir.startswith('.')]
    return dirlist

def config_init(argv):
    ### Initialize config file ###
    if len(argv) != 2:
        raise ValueError("ERROR, need config file. Ie: python generate_data.py config.json")    
    configuration_file = str(argv[1])
    with open(configuration_file, 'r') as f:
        f = f.read()
        PARAMS = json.loads(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    return PARAMS
