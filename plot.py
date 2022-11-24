"""
Plot the result of an evaluation run
"""
import matplotlib.pyplot as plt
import torch
import argparse

def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument('filepath', type=str, help='The filepath to the file with data to graph')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args(argparse.ArgumentParser())

    data_dict = torch.load(args.filepath)

    plt.figure()
    plt.plot(list(range(len(data_dict['num_conflicts']))), data_dict['num_conflicts'])

    plt.figure()
    plt.plot(list(range(len(data_dict['episode_lengths']))), data_dict['episode_lengths'])

    plt.show()
