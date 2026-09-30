import os
import argparse
import time
import datetime
import sys
import torch
import numpy as np
import gc
from loader import Loader
from dataList import dataList
import logging
from utils import TrainingInfo, SetWeights, Tools, Metrics
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--approach', choices = ['ehat', 'hat', 'ehat_c', 'singleTask', 'Unet50', 'Unet25'], type = str, help = "Training strategies.")
parser.add_argument('--listPath', default = './dataset/lists/', type = str, help = "Path to data lists.")
parser.add_argument('--slr', default = 5e-4, type = float, help = "Initial learning rate.")
parser.add_argument('--elr', default = 5e-6, type = float, help = "End learning rate.")
parser.add_argument('--patience', default = 4, type = int, help = "Patience of early stopping.")
parser.add_argument('--warmUp', default = 5, type = int, help = "Number of epoch for warming up optimizer.")
parser.add_argument('--epochs', default = 30, type = int, help = "Number of training epochs.")
parser.add_argument('--batch', default = 16, type = int, help = "Batch size")
parser.add_argument('--smax', default = 1000, type = float, help = "Hyperparameter.")
parser.add_argument('--lamb', default = 6, type = float, help = "Regularization strength for HAT.")
parser.add_argument('--seed', default = 0, type = int, help = "Training order.")
parser.add_argument('--size', default = [1, 192, 192], type = int, nargs = '+', help = "Input image size, [C, W, H]")
parser.add_argument('--reset', action = 'store_true', help = "Generate new training, validation and testing sets.")
parser.add_argument('--fourTasks', action = 'store_true', help = "Use four tasks.")
parser.add_argument('--init', action = 'store_true', help = "Initialize the unused weightings.")
parser.add_argument('--split', default = [0.6, 0.2, 0.2], type = float, nargs = '+', help = "Percentage for splitting data to [train, val, test].\
        It only works if --reset = true")
parser.add_argument('--lazy_loading', action = 'store_true', help = "Lazy loading to keep RAM usage low.")
parser.add_argument('--manually_split', action = 'store_true', help = "Generate lists with current split lists.")
args = parser.parse_args()

if args.approach in ["ehat", "Unet50", "Unet25"]:
    from approach.ehat import Approach
elif args.approach in ["hat", "ehat_c"]:
    from approach.hat import Approach
elif args.approach == "singleTask":
    from approach.singleTask import Approach
else:
    sys.exit()
if args.approach in ["ehat", "ehat_c"]:
    from model.unet_ehat import Net
elif args.approach == "Unet50":
    from model.unet50_ehat import Net
elif args.approach == "Unet25":
    from model.unet25_ehat import Net
elif args.approach == "hat":
    from model.unet_hat import Net
elif args.approach == "singleTask":
    from model.unet import Net
####################### logs #######################################
exeTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
num = Tools.recordExp(exeTime)
print("{:=^100}".format("Exp{:0>3d}".format(num)))
expPath = "./experiments/exp{:0>3d}".format(num)
logPath = os.path.join(expPath, "logs")
weightPath = os.path.join(expPath, "weights")
boardPath = os.path.join(expPath, "tensorboard")
Tools.makedirs(logPath)
Tools.makedirs(weightPath)
Tools.makedirs(boardPath)

LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
DATE_FORMAT = '%Y%m%d %H:%M:%S'
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join(logPath, 'training.log'))
file_handler.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG,
                    handlers=[file_handler, console_handler], 
                    format = LOGGING_FORMAT, datefmt = DATE_FORMAT)
logging.captureWarnings(True)

writer = SummaryWriter(log_dir = boardPath)
LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
DATE_FORMAT = '%Y%m%d %H:%M:%S'
logging.basicConfig(level = logging.INFO, handlers=[logging.FileHandler('app.log'), logging.StreamHandler()],format = LOGGING_FORMAT, datefmt = DATE_FORMAT)

try:
######################## Set path ##################################
    path = {}
    path['lumbar_train'] = os.path.join(args.listPath, "train_list/lumbar_train.txt")
    path['lumbar_val'] = os.path.join(args.listPath,"val_list/lumbar_val.txt")
    path['lumbar_test'] = os.path.join(args.listPath,"test_list/lumbar_test.txt")
    path['RV_train'] = os.path.join(args.listPath,"train_list/RV_train.txt")
    path['RV_val'] = os.path.join(args.listPath,"val_list/RV_val.txt")
    path['RV_test'] = os.path.join(args.listPath,"test_list/RV_test.txt")
    path['tumor_train'] = os.path.join(args.listPath,"train_list/tumor_train.txt")
    path['tumor_val'] = os.path.join(args.listPath,"val_list/tumor_val.txt")
    path['tumor_test'] = os.path.join(args.listPath,"test_list/tumor_test.txt")
    path['lumbar_MR_train'] = os.path.join(args.listPath, "train_list/lumbar_MR_train.txt")
    path['lumbar_MR_val'] = os.path.join(args.listPath,"val_list/lumbar_MR_val.txt")
    path['lumbar_MR_test'] = os.path.join(args.listPath,"test_list/lumbar_MR_test.txt")

####################### Load data ###################################
    logging.info("#"*100)
    logging.info("Selected Approach: {}, weight_init: {}".format(args.approach, args.init))
    logging.info("#"*100)

    # Edit split_list directly and call this function can help you generate the split lists
    exists = False
    L = Loader(size = args.size)
    if args.manually_split:
        dataList.make_from_split_list()
    elif args.reset:
        DL = dataList(train = args.split[0], val = args.split[1], test = args.split[2])
        DL.reallocate()
    else:
        exists = L.check_all_data_exists(path, args.fourTasks)

    order = ""
    if not exists:
        logging.debug("Creating All Dataset...")
        L.make(path = path, seed = args.seed, fourTasks = args.fourTasks)
        logging.debug("Done!")
    if args.lazy_loading:
        metadata, info = L.load_metadata(seed = args.seed, fourTasks = args.fourTasks)
        order = metadata['order']
    else:
        logging.debug("Loading All Data...")
        load_s = time.time()
        data, info = L.load_all(seed = args.seed, fourTasks = args.fourTasks)
        order = data['order']
        logging.debug("Loading Time: {}sec".format(time.time() - load_s))

    gc.collect()

###################### Set up training procedure ###############################
    logging.info("Initialize...")
    if args.approach == "singleTask":
        args.smax = None
        net = Net(args.size[0], info[0][1])
    else:
        net = Net(args.size[0], info)
    net.apply(SetWeights.he_init)
    net.cuda()
    if args.init:
        torch.save(net, os.path.join(weightPath, "w_init"))

    appr = Approach(appr = args.approach, info = info, model = net, epoch = args.epochs, batch = args.batch, writer = writer, patience = args.patience,\
            s_lr = args.slr, e_lr = args.elr, warm_up = args.warmUp, smax = args.smax, weightPath = weightPath, init = args.init, lamb = args.lamb)
    TrainingInfo.printTask(info, order, args.seed)
    TrainingInfo.printHyper(args.approach, args.slr, args.elr, args.epochs, args.batch, args.patience, args.warmUp, args.smax, args.lamb, appr.opt)
    TrainingInfo.printModel(net)
    del net
    torch.cuda.empty_cache()

###################### Training loop ##############################
    logging.info("#"*100)
    logging.info("Start training...")
    res = np.zeros((len(info), len(info)))
    Total_s = time.time()
    for t in range(len(info)):
        if t == 1 and args.approach != "singleTask":
            appr.model.apply(SetWeights.zero_drop)

        if args.lazy_loading:
            train_data = L.load_task(order[t], 'train')
        else:
            train_data = data[t]

        t_s = time.time()
        logging.info("*"*100)
        logging.info("Task {:2d} ({:s})".format(t, train_data['name']))
        appr.train(t, train_data)
        logging.info("-"*100)
        del train_data
        gc.collect()

        eval_tasks = [t] if args.approach == "singleTask" else range(t + 1)
        for pt in eval_tasks:
                if args.lazy_loading:
                    test_data = L.load_task(order[pt], 'test')
                else:
                    test_data = data[pt]
                x_test = test_data['test']['x']
                y_test = test_data['test']['y']

                test_loss, test_DC = appr.eval(pt, x_test, y_test)
                test_mDC = np.mean(test_DC[1:])
                logging.info('>>> Test on task {:2d} - {:15s}: loss = {:.3f}, DC = {:^5.2%} <<<'.format(pt, order[pt], test_loss, test_mDC))
                res[pt][t] = test_mDC
        
        if args.approach == "singleTask" and (t + 1 < len(info)):
            net = Net(args.size[0], info[t + 1][1])
            net.apply(SetWeights.he_init)
            net.cuda()
            appr.model = net             #reset to initial weights for training a new task
            del net

        del test_data, x_test, y_test
        torch.cuda.empty_cache()
        gc.collect()

        Metrics.test_res(res, order)
        logging.info("> Task {} Elapsed Time: {:.1f} h".format(t, ((time.time() - t_s) / 3600)))
        logging.info("-"*100)
        logging.info("*"*100)

    Tools.res2csv(res, order, expPath)
    logging.info("> Total Elapsed Time: {:.1f} h".format(((time.time() - Total_s) / 3600)))
    logging.info("#"*100)

except:
    logging.exception("Got Exception.")
    exit(-1)
