import logging
import os
import shutil
import torch
import numpy as np
import csv
import scipy.io
import random
import gc
from tqdm import trange

class TrainingInfo():

    @classmethod
    def printTask(cls, info, order, seed):
        '''print information of training tasks\n
        Args:
            info: information of task
            order: order of training
            seed: seed number
        '''
        classes = name = task = "|{:^13}" * (len(info) + 1) +"|"
        line = "-" * 14 * (len(info) + 1) + "-"
        task = task.format(" ", *(["Task {}".format(i) for i in range(len(info))]))
        name = name.format("Name", *(order))
        classes = classes.format("Classes", *(c[1] for c in info))
        logging.info("Training Order (seed {}):".format(seed))
        logging.info(line)
        logging.info(task)
        logging.info(line)
        logging.info(name)
        logging.info(line)
        logging.info(classes)
        logging.info(line)

    @classmethod
    def printHyper(cls, appr, s_lr, e_lr, epoch, batch, patience, warm_up, smax, lamb, optm):
        '''print selected approach and hyperparameters\n
        Args:
            appr: Approach
            s_lr: Initial learning rate
            e_lr: Minimum learning rate for learning rate decay
            epoch: Total epochs for training
            batch: Batch size
            patience: Early stopping criteria
            warm_up: Mumber of warming up epochs
            smax: Hyperparameter. Control the sigmoid gate
            lamb: Hyperparameter. Control regularization strength
            optm: Optimizer
        '''
        lr = "s_lr = {:.2e} | e_lr = {:.2e}".format(s_lr, e_lr)
        E = "Epochs = {}".format(epoch)
        B = "Batch size = {}".format(batch)
        early_stop = "Patience = {}".format(patience)
        warm = "Warm up epoch(s) = {}".format(warm_up)
        s = "smax = {}".format(smax)
        lamb = "lamb = {}".format(lamb)
        logging.info("Hyperparameters:")
        logging.info(E)
        logging.info(B)
        logging.info(lr)
        logging.info(early_stop)
        logging.info(warm)
        logging.info(s)
        if appr in ["hat", "ehat_c"]:
            logging.info(lamb)
        logging.info("{}".format(optm))

    @classmethod
    def printModel(cls, model):
        '''
        print number of parameters in a model
        '''
        count = 0
        for p in model.parameters():
            count+=np.prod(p.size())
        logging.info("Num parameters = {}".format(cls._human_format(count)))

    @classmethod
    def _human_format(cls, num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

class Metrics():
    
    @classmethod
    def confusion_matrix(cls, y_true, y_pred, classes):
        '''Confusion matrix\n
        Args:
            y_true: ground truth
            y_pred: prediction
            classes: number of classes
        
        Return:
            confusion matrix
        '''
        N = classes
        y = N * y_true.long() + y_pred.long()
        y = torch.bincount(y).cuda()
        if len(y) < N * N:
            y = torch.cat([y, torch.zeros(N * N - len(y), dtype=torch.long).cuda()])
        y = y.reshape(N, N)
        return y
    
    @classmethod
    def confusion(cls, y_true, y_pred, classes):
        '''get metric for calculating DC\n
        Args:
            y_true: ground truth
            y_pred: prediction
            classes: number of classes

        Return:
            True positive, False positive, False negative
        '''
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        conf = cls.confusion_matrix(y_true, y_pred, classes).cpu().numpy()
        TP = np.zeros(classes)
        FP = np.zeros(classes)
        FN = np.zeros(classes)
        for i in range(classes):
            TP[i] = conf[i][i]
            FP[i] = np.sum(conf[:, i]) - TP[i]
            FN[i] = np.sum(conf[i]) - TP[i]

        return TP, FP, FN

    @classmethod
    def val_res(cls, epoch, val_loss, val_DC, val_mDC, lr, mark):
        '''print validation result\n
        Args:
            epoch: current epoch index
            val_loss: validation loss
            val_DC: validation DC
            val_mDC: validation mean DC
            lr: current learning rate
            mark: mark * for the patient
        '''
        info = "Epoch {:0>2d} | val_loss: {:.3f} | val_mDC: {:.2%} | lr: {:.2e} |".format(epoch + 1, val_loss, val_mDC, lr)
        if mark:
            info += " *"
        line = "-" * 10 * (len(val_DC) + 1) + "-"
        classes = "|{:^9}" * (len(val_DC) + 1) +"|"
        DC = "|{:^9}" + "|{:^9.2%}" * (len(val_DC)) +"|"
        classes = classes.format("Class", *([i for i in range(len(val_DC))]))
        DC = DC.format("DC(%)", *(val_DC))
        table = [line, classes, line, DC, line]
        #print and save to log
        logging.debug(info)
        for txt in table:
            logging.debug(txt)

    @classmethod
    def test_res(cls, res, order):
        '''
        print testing result
        '''
        line = "-" * 10 * (len(order) + 1) + "-"
        row = "|{:^9}" * (len(order) + 1) +"|"
        DC = "|{:^9}" + "|{:^9.2%}" * (len(order)) +"|"
        first_row = row.format("DC(%)", *(order))
        logging.info(line)
        logging.info(first_row )
        for i in range(len(order)):
            info = DC.format(order[i], *(res[i]))
            logging.info(line)
            logging.info(info)
        logging.info(line)

class SetWeights():

    @staticmethod
    def he_init(m):
        '''
        Initialize model with He initialization
        '''
        layerName = m.__class__.__name__
        if layerName == 'Conv2d':
            torch.nn.init.kaiming_normal_(m.weight, a = 0, mode='fan_in', nonlinearity='relu')
            m.bias.data.fill_(0)

    @staticmethod
    def zero_drop(m):
        '''
        Set drop out ration to 0
        '''
        layerName = m.__class__.__name__
        if layerName == 'Dropout2d':
            m.p = 0

    @staticmethod
    def unused_init(net, init_weight, mask_pre):
        '''Initialize unused parameters with its original values (i.e., values before the first task was trained.)\n
        Args:
            net: network to be initialized
            init_weight: the original network (i.e., untrained)
            mask_pre: cumulative masks
        '''
        weights = {}
        for (name, para), init in zip(net.named_parameters(), init_weight.parameters()):
            values = net.get_view_for(name, mask_pre)
            para = para.clone().detach()
            init = init.clone().detach()
            if values is not None:
                #laryer to be concated
                if "conv2" in name and name[0] == "c":
                    pass
                else:
                    print(name)
                    para *= values
                    init *= (1 - values)
                    para += init
            weights[name] = para
        return weights

class Summary():
    '''Summary writer\n
    Args:
        writer: SummaryWriter (torch.utils.tensorboard)
    '''
    def __init__(self, writer):
        self._writer = writer

    def write(self, t, e, lr, train_total_loss, train_reg, val_loss, val_mDC):
        '''write to tensorboard\n
        Args:
            t: task ID
            e: current epoch index
            lr: current learning rate
            train_total_loss: total loss (training loss and regularization loss)
            train_reg: regularization loss
            val_loss: validation loss
            val_DC: validation mean DC
        '''
        self._writer.add_scalar('Task{}_lr'.format(t), lr, e)
        self._writer.add_scalar('Task{}_train_total_loss'.format(t), train_total_loss, e)
        self._writer.add_scalar('Task{}_train_loss'.format(t), train_total_loss - train_reg, e)
        self._writer.add_scalar('Task{}_train_reg'.format(t), train_reg, e)
        self._writer.add_scalar('Task{}_val_loss'.format(t), val_loss, e)
        self._writer.add_scalar('Task{}_val_mDC'.format(t), val_mDC, e)

class Tools():

    @classmethod
    def mkdir(cls, path):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
    @classmethod
    def makedirs(cls, path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
    @classmethod
    def clean(cls, path):
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass

    @classmethod
    def recordExp(cls, exeTime):
        cls.mkdir("./experiments")
        with open("./experiments/exp.log", mode = "a+") as f:
            f.seek(0)
            lines = f.readlines()
            num = len(lines) + 1
            f.write("{}: Running exp{:0>3d}\n".format(exeTime, num))
        cls.mkdir("./experiments/exp{:0>3d}".format(num))
        return num

    @classmethod
    def res2csv(cls, res, order, savePath):
        table = []
        first_row = [""] + order
        table.append(first_row)
        for i in range(len(order)):
            row = [order[i]]
            row += list(res[i])
            table.append(row)
        with open(os.path.join(savePath, 'res.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(table)
            f.close()

    @classmethod
    def color_map(cls, image):
        cmap = {0 : [0, 0, 0], 1 : [255, 255, 255], 2 : [255, 0, 0], 3 : [0, 255, 0], 4 : [0, 0, 255], 5 : [255, 255, 0],\
                6 : [0, 255, 255], 7 : [255, 0, 255]}
        u, inv = np.unique(image, return_inverse = True)
        return np.array([cmap[key] for key in u], dtype = np.uint8)[inv].reshape(image.shape[0], image.shape[1], 3)
    
    @classmethod
    def normalize(cls, image):
        M = np.max(image)
        m = np.min(image)
        image = (image - m) / (M - m)
        return np.float32(image)

    @classmethod
    def loadMat(cls, path):
        ''' Load .mat file to numpy array
        Args:
            path: path to .mat file
        
        Return:
            numpy array
        '''
        image = scipy.io.loadmat(path, mat_dtype = True)
        image = image['varName']

        return image

    @classmethod
    def order(cls, seed, fourTasks):
        ''' Training order
        Args:
            seed: seed for random function
            fourTasks: Boolean. True: four-task training, False: three-task training
        
        Return:
            A list, training order
        '''
        dic = {'L':0, 'RV':1, 'T':2, 'L_MR': 3}
        if fourTasks:
            random.seed(seed * seed)
            order = list(dic.values())
            random.shuffle(order)
            return order

        order = [[dic['L'], dic['RV'], dic['T']],
                    [dic['RV'], dic['L'], dic['T']],
                    [dic['T'], dic['L'], dic['RV']],
                    [dic['L'], dic['T'], dic['RV']],
                    [dic['T'], dic['RV'], dic['L']],
                    [dic['RV'], dic['T'], dic['L']]]

        return order[seed]

    @classmethod
    def readFromList(cls, dataList):
        ''' Read from image list
        Args:
            dataList: path to data lists
        
        Return:
            numpy array, image and mask
        '''
        f = open(dataList, 'r')
        images = []
        masks = []
        for line in f:
            try:
                image, mask = line.strip("\n").split(" ")
            
            except ValueError:
                print("Path Error!")
                return -1
            image, mask = cls.loadMat(image), cls.loadMat(mask)
            images.append(image)
            masks.append(mask)
        return np.asarray(images, dtype=np.float32), np.asarray(masks, dtype=np.uint8)

    @classmethod
    def readlines(cls, dataList):
        ''' Read from list and count the number of data
        Args:
            dataList: path to data lists
        
        Return:
            number of data
        '''

        f = open(dataList, 'r')
        cnt = 0
        for line in f:
            cnt+=1
        return cnt
