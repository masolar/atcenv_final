import os
import pickle

ModelsPath = "results"

Temporal_Buffer = 4


def save_DDQL(Path, Name, agent):
    ''' Saves actions, rewards and states (images) in DataPath'''
    if not os.path.exists(Path):
        os.makedirs(Path)
    agent.model.save(Path + "/" + Name)
    print(Name, "saved")

def dump_pickle(obj, name):
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name):
    with open(name, 'rb') as handle:
        obj = pickle.load(handle)
    return obj