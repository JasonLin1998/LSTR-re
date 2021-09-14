import matplotlib.pyplot as plt
import os

def Log(exp, mode, rr=False, root='./experiment'):
    log_path = os.path.join(root, '{}/save{}_loss.log'.format(exp, mode))
    figure_dir = os.path.join(root, '{}/figure'.format(exp))
    isExists = os.path.exists(figure_dir)
    if not isExists:
        os.mkdir(figure_dir)
    with open(log_path, 'r') as f:
        for ind, line in enumerate(f.readlines()):
            line.strip()
            lines = line.split()
            if not ind == 0:
                iter.append(int(lines[0]))
                loss.append(float(lines[1]))
                loss_rr.append(float(lines[2]))
                loss_curve.append(float(lines[3]))
                lr.append(float(lines[4]))

    if len(iter) == len(loss) == len(loss_rr) == len(loss_curve):
        if mode == 'train':
            sample_iter = iter[0:len(iter):10]
            sample_loss = loss[0:len(loss):10]
            sample_loss_rr = loss_rr[0:len(loss_rr):10]
            sample_loss_curve = loss_curve[0:len(loss_curve):10]
        else:
            sample_iter = iter[0:len(iter):50]
            sample_loss = loss[0:len(loss):50]
            sample_loss_rr = loss_rr[0:len(loss_rr):50]
            sample_loss_curve = loss_curve[0:len(loss_curve):50]

        if len(sample_iter) == len(sample_loss) == len(sample_loss_rr) == len(sample_loss_curve):
            if rr:
                # plt.plot(sample_iter, sample_loss, 's-', color='r', label='loss')
                plt.plot(sample_iter, sample_loss_rr, 'o-', color='g', label='loss_rr')
                # plt.plot(sample_iter, sample_loss_curve, 'v-', color='b', label='loss_curve')
                plt.title('Loss')
                plt.xlabel('iter')
                plt.ylabel('{} loss rr {}'.format(mode, exp))
                plt.legend(loc='best')
                plt.savefig('./experiment/{}/figure/{}_loss_rr_{}.png'.format(exp, mode, exp))
                plt.show()
            else:
                plt.plot(sample_iter, sample_loss, 's-', color='r', label='loss')
                # plt.plot(sample_iter, sample_loss_rr, 'o-', color='g', label='loss_rr')
                plt.plot(sample_iter, sample_loss_curve, 'v-', color='b', label='loss_curve')
                plt.title('Loss')
                plt.xlabel('iter')
                plt.ylabel('{} loss {}'.format(mode, exp))
                plt.legend(loc='best')
                plt.savefig('./experiment/{}/figure/{}_loss_{}.png'.format(exp, mode, exp))
                plt.show()

if __name__ == '__main__':
    iter = []
    loss = []
    loss_rr = []
    loss_curve = []
    lr = []

    exp = 'try10'
    mode = 'val'
    rr = True

    data_root = './experiment'
    Log(exp, mode, rr, data_root)