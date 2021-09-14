#!/usr/bin/env python
# I change the dataloader
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import json
import torch
import math
import numpy as np
import queue
import pprint
import random
import argparse
import importlib
import threading
import traceback

from tqdm import tqdm
from utils import stdout_to_tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from torch.multiprocessing import Process, Queue, Pool
from db.datasets import datasets
import models.py_utils.misc as utils
from datasets.TuSimple import TUSIMPLE
from sample import tusimple
from utils import crop_image, normalize_, color_jittering_, lighting_
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from datasets import TuSimple
from tensorboardX import SummaryWriter

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train CornerNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--GPU", default='0')
    parser.add_argument("--threads", dest="threads", default=4, type=int)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--exp", dest='exp', default='example')
    parser.add_argument("--log_loss", type=int, default=500)
    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def train(training_dbs, validation_db, start_iter=0, freeze=False, args=None):
    learning_rate    = system_configs.learning_rate
    max_iteration    = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    snapshot         = system_configs.snapshot
    val_iter         = system_configs.val_iter
    display          = system_configs.display
    decay_rate       = system_configs.decay_rate
    stepsize         = system_configs.stepsize
    batch_size       = system_configs.batch_size

    # save the args and mkdir
    if not os.path.exists('experiment'):
        os.mkdir('experiment')
    exp_path = 'experiment/' + args.exp
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    with open(exp_path+'/opt.txt', 'w') as file:
        file.write("args:{}".format(args))
        file.write('\n system_configs:')
        for k in system_configs._configs.keys():
            file.write("{}:{}\n".format(k, system_configs._configs[k]))
    save_path = exp_path + '/save'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_checkpoints_path = save_path + "/checkpoints"
    if not os.path.exists(save_checkpoints_path):
        os.mkdir(save_checkpoints_path)
    if start_iter == 0:
        with open(save_path + 'train_loss.log', 'w') as file:
            file.write('iter\t\tloss\t\tloss_rr\t\tloss_curve\t\tlr\n')
        with open(save_path + 'val_loss.log', 'w') as file:
            file.write('iter\t\tloss\t\tloss_rr\t\tloss_curve\t\tlr\n')
    # getting the size of each database
    training_size   = len(training_dbs.db_inds)
    validation_size = len(validation_db.db_inds)
    num_epochs = math.ceil(max_iteration/math.ceil(training_size/batch_size))
    print("building model...")
    nnet = NetworkFactory(flag=True, args=args)

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)
    start_epoch = 1
    cnt = 0

    if start_iter:
        learning_rate /= (decay_rate ** (start_iter // stepsize))
        start_epoch = start_iter//math.ceil(training_size/batch_size) + 1
        cnt = start_iter
        # load_the_checkpoint
        cache_file = save_checkpoints_path + '/LSTR_{}.plk'.format(start_iter)
        with open(cache_file, "rb") as f:
            params = torch.load(f)
            model_dict = nnet.model.state_dict()
            if len(params) != len(model_dict):
                pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
            else:
                pretrained_dict = params
            nnet.model.state_dict().update(pretrained_dict)
            nnet.model.load_state_dict(model_dict)




        nnet.set_lr(learning_rate)
        print("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)

    print("training start...")
    nnet.cuda()
    nnet.train_mode()
    header = None
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    train_loader = torch.utils.data.DataLoader(dataset=training_dbs,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=16,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset=validation_db,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=16,
                                               pin_memory=True)

    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        for step, (images, labels, masks, img_idxs, rr) in tqdm(enumerate(train_loader)):
            labels_all = []
            cnt = cnt + 1

            for label in labels:
                tgt_ids = label[:, 0]
                label = label[tgt_ids > 0]
                label = torch.stack([label] * batch_size, axis=0)
                labels_all.append(label)

            training = {
                "xs": [images, masks],
                "ys": [images, *labels_all],
                "rr": rr
            }
            iteration = cnt
            viz_split = 'train'
            save = True if (display and iteration % display == 0) else False
            (set_loss, loss_dict) \
                = nnet.train(iteration, save, viz_split, **training)
            (loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled, loss_value) = loss_dict
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
            metric_logger.update(lr=learning_rate)

            if iteration % args.log_loss == 0:
                with open(save_path + 'train_loss.log', 'a') as file:
                    file.write('{}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}\n'.format(iteration, loss_value,
                                                                loss_dict_reduced_scaled['loss_rr'], loss_dict_reduced_scaled['loss_curves'], learning_rate))

            if iteration % 1000 == 0:
                writer.add_scalar('train/loss_ce', loss_dict_reduced_scaled['loss_ce'], iteration)
                writer.add_scalar('train/loss_lowers', loss_dict_reduced_scaled['loss_lowers'], iteration)
                writer.add_scalar('train/loss_uppers', loss_dict_reduced_scaled['loss_uppers'], iteration)
                writer.add_scalar('train/loss_curves', loss_dict_reduced_scaled['loss_curves'], iteration)
                writer.add_scalar('train/loss', loss_value, iteration)

                if not system_configs.rr_weight == 0:
                    writer.add_scalar('val/loss_rr', loss_dict_reduced_scaled['loss_rr'], iteration)
                if system_configs.aux_loss:
                    writer.add_scalar('train/loss_ce_0', loss_dict_reduced_scaled['loss_ce_0'], iteration)
                    writer.add_scalar('train/loss_lowers_0', loss_dict_reduced_scaled['loss_lowers_0'], iteration)
                    writer.add_scalar('train/loss_uppers_0', loss_dict_reduced_scaled['loss_uppers_0'], iteration)
                    writer.add_scalar('train/loss_curves_o', loss_dict_reduced_scaled['loss_curves_0'], iteration)
                    if system_configs.aux_rr:
                        writer.add_scalar('train/loss_rr_0', loss_dict_reduced_scaled['loss_rr_0'], iteration)
            del set_loss

            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                nnet.eval_mode()
                viz_split = 'val'
                save = True
                try:
                    (images, labels, masks, img_idxs, rr) = next(val_iteration)
                except:
                    val_iteration = iter(val_loader)
                    (images, labels, masks, img_idxs, rr) = next(val_iteration)
                labels_all = []
                for label in labels:
                    tgt_ids = label[:, 0]
                    label = label[tgt_ids > 0]
                    label = torch.stack([label] * batch_size, axis=0)
                    labels_all.append(label)
                validation = {
                "xs": [images, masks],
                "ys": [images, *labels_all],
                "rr": rr
                 }

                (val_set_loss, val_loss_dict) \
                    = nnet.validate(iteration, save, viz_split, **validation)
                (loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled, loss_value) = val_loss_dict
                print('[VAL LOG]\t[Saving training and evaluating images...]')
                metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
                metric_logger.update(class_error=loss_dict_reduced['class_error'])
                metric_logger.update(lr=learning_rate)
                with open(save_path + 'val_loss.log', 'a') as file:
                    file.write('{}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}\n'.format(iteration, loss_value,
                                                                 loss_dict_reduced_scaled['loss_rr'],
                                                                 loss_dict_reduced_scaled['loss_curves'], learning_rate))

                writer.add_scalar('val/loss_ce', loss_dict_reduced_scaled['loss_ce'], iteration)
                writer.add_scalar('val/loss_lowers', loss_dict_reduced_scaled['loss_lowers'], iteration)
                writer.add_scalar('val/loss_uppers', loss_dict_reduced_scaled['loss_uppers'], iteration)
                writer.add_scalar('val/loss_curves', loss_dict_reduced_scaled['loss_curves'], iteration)
                writer.add_scalar('val/loss', loss_value, iteration)

                if not system_configs.rr_weight == 0:
                    writer.add_scalar('val/loss_rr', loss_dict_reduced_scaled['loss_rr'], iteration)
                if system_configs.aux_loss:
                    writer.add_scalar('val/loss_ce_0', loss_dict_reduced_scaled['loss_ce_0'], iteration)
                    writer.add_scalar('val/loss_lowers_0', loss_dict_reduced_scaled['loss_lowers_0'], iteration)
                    writer.add_scalar('val/loss_uppers_0', loss_dict_reduced_scaled['loss_uppers_0'], iteration)
                    writer.add_scalar('val/loss_curves_o', loss_dict_reduced_scaled['loss_curves_0'], iteration)
                    if system_configs.aux_rr:
                        writer.add_scalar('val/loss_rr_0', loss_dict_reduced_scaled['loss_rr_0'], iteration)
                nnet.train_mode()

            # save the models
            if iteration % snapshot == 0:
                cache_file = save_checkpoints_path + '/LSTR_{}.plk'.format(iteration)
                print("saving model to {}".format(cache_file))
                with open(cache_file, "wb") as f:
                    params = nnet.model.state_dict()
                    torch.save(params, f)

            if iteration % stepsize == 0:
                learning_rate /= decay_rate
                nnet.set_lr(learning_rate)

            if iteration % (training_size // batch_size) == 0:
                metric_logger.synchronize_between_processes()
                print("Averaged stats:", metric_logger)

if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)

    configs["system"]["snapshot_name"] = args.cfg_file  # CornerNet
    system_configs.update_config(configs["system"])

    writer = SummaryWriter(comment='{}'.format(args.exp))

    train_split = system_configs.train_split
    val_split   = system_configs.val_split

    dataset = system_configs.dataset  # MSCOCO | FVV
    print("loading all datasets {}...".format(dataset))

    threads = args.threads  # 4 every 4 epoch shuffle the indices
    training_dbs  = TUSIMPLE(configs["db"], train_split)
    validation_db = TUSIMPLE(configs["db"], val_split)

    # print("system config...")
    # pprint.pprint(system_configs.full)
    #
    # print("db config...")
    # pprint.pprint(training_dbs[0].configs)

    print("len of training db: {}".format(len(training_dbs.db_inds)))
    print("len of testing db: {}".format(len(validation_db.db_inds)))

    print("freeze the pretrained network: {}".format(args.freeze))
    train(training_dbs, validation_db, args.start_iter, args.freeze, args=args) # 0
