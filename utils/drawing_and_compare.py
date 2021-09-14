import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_arg():
    parser = argparse.ArgumentParser(description="draw the figure")
    parser.add_argument("--path", default='.')
    args = parser.parse_args()
    return args

def main():
    args = parse_arg()
    for file in os.listdir(args.path):
        with open(args.path+'/'+file, 'r') as myfile:
            line = myfile.readline()
            line = myfile.readline()
            iter = list()
            loss = list()
            while(line):
                if file == "val_0.log":
                    line_set = line.split("\t")
                else:
                    line_set = line.split("\t\t")
                iter.append(int(line_set[0]))
                loss.append(float(line_set[1]))
                line = myfile.readline()
        if file == "val_0.log":
            iter = iter[:812]
            loss = loss[:812]
        iter = np.array(iter)
        loss = np.array(loss)
        plt.plot(iter, loss, linewidth=0.1)
    plt.savefig("figure_of_loss/temp.jpg")
    print("ok")

if __name__=="__main__":
    main()