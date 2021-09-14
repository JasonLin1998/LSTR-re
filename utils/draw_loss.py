import torch
from tensorboardX import SummaryWriter
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="draw the figure")
    parser.add_argument("--exp", default='example')
    parser.add_argument("--mode", default='train')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    file_path_train = '../experiment/{}/savetrain_loss.log'.format(args.exp)
    file_path_val = '../experiment/{}/saveval_loss.log'.format(args.exp)
    tb_writer = SummaryWriter( '../experiment/{}'.format(args.exp) + '/Tensorboard')
    with open(file_path_train, 'r') as my_file:
        a = my_file.readline()
        a = my_file.readline()
        while(a):
            cell = a.split('\t')
            tb_writer.add_scalar("loss/train_loss", float(cell[1]), cell[0])
            a = my_file.readline()

    with open(file_path_val, 'r') as my_file:
        a = my_file.readline()
        a = my_file.readline()
        while(a):
            cell = a.split('\t')
            tb_writer.add_scalar("loss/val_loss", float(cell[1]), cell[0])
            a = my_file.readline()
    print("ok")

if __name__ == "__main__":
    main()
