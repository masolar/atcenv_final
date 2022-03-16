import os
import pickle

ModelsPath = "results"


def getPath():
    return ModelsPath

def save_DDQL(Path, Name, agent):
    ''' Saves actions, rewards and states (images) in DataPath'''
    if not os.path.exists(Path):
        os.makedirs(Path)
    agent.save_weights(Path + "/" + Name)
    print(Name, "saved")

def dump_pickle(obj, name):
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name):
    with open(name, 'rb') as handle:
        obj = pickle.load(handle)
    return obj