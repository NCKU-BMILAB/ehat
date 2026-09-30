import numpy as np
import os
import gc
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import Metrics, Summary, SetWeights
import logging
from tqdm import trange
import warnings
import psutil
from abc import ABC, abstractmethod

class Base(ABC):
    '''
    Args:
        appr: Approach (ehat, hat, single task)
        info: Informations for tasks. List of tuple, [(Task1 ID, Number of class in task 1), ...].
        model: Model for training.
        epoch: Total epochs for training.
        batch: Batch size.
        writer: Summary writer
        patience: Early stopping criteria.
        s_lr: Initial learning rate.
        e_lr: Minimum learning rate for learning rate decay.
        warm_up: Mumber of warming up epochs.
        smax: Hyperparameter. Control the sigmoid gate.
        weightPath: Path to save trained model.
        init: Initialize unused parameters.
        lamb: None.
    '''
    def __init__(self, appr, info, model, epoch, batch, writer, patience, s_lr, e_lr, warm_up, smax, weightPath, init, lamb = None):
        self.appr = appr
        self.info = info
        self.model = model
        self.epoch = epoch
        self.batch = batch
        self.s_lr = s_lr
        self.e_lr = e_lr
        self.warm_up = warm_up
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.opt = self._optimizer()
        self.summary = Summary(writer)
        self.smax = smax
        self.patience = patience
        self.weightPath = weightPath
        self.init = init
        self.lamb = lamb
        self.clipgrad = 10000

        self.mask_pre = None            #Cumulative masks
        self.mask_back = None           #Reversed attention masks

    def _optimizer(self, lr = None):
        '''
        Args:
            lr: Learning rate of optimizer.
        
        Returns:
            AdamW with learning rate = lr.
        '''
        if lr is None:
            lr = self.s_lr
        return torch.optim.AdamW(self.model.parameters(), lr = lr)

    def _scheduler(self, epoch, lr):
        '''
        The learning rate scheduler which use a linearly decay strategy.
        Args:
            epoch: Epoch index.
        
        Returns:
            Learning rate.
        '''
        #set up warming up epochs and learning rate
        if epoch < self.warm_up:
            lr = 1e-6 + epoch * (self.s_lr - 1e-6) / self.warm_up
            return lr

        # learning rate decay (linear decay)
        lr = self.s_lr + 1.5 * (epoch - self.warm_up) * (self.e_lr - self.s_lr) / (self.epoch - self.warm_up - 1)
        if lr < self.e_lr:
            return self.e_lr
        return lr



    def train(self, t, data):
        ''' Training function
        Args:
            task: Task ID.
            x_train: An array of shape [N, C, W, H]. N = Number of training images, C = Channels, W = Width, H = Height
            y_train: An array of shape [N, 1, W, H]. N = Number of training labels, C = Channels, W = Width, H = Height
            x_val: An array of shape [N, C, W, H]. N = Number of validation images, C = Channels, W = Width, H = Height
            y_val: An array of shape [N, 1, W, H]. N = Number of validation labels, C = Channels, W = Width, H = Height
            
        Note:
            All labels should be sparse (i.e., for each number in array [2, 0, 2], every number represents a class).
            One-hot encoding labels cannot be used in this case (i.e., [[0, 0, 1], [1, 0, 0], [0, 0, 1]]).
        '''
        train_loss, val_loss = 0, 0
        best_DC = -1
        p = 0
        lr = 1e-6
        mark = False
        try:
            for e in range(self.epoch):
                #get lr from scheduler
                lr = self._scheduler(e, lr)
                #set up optimizer
                self.opt = self._optimizer(lr)
                #get training loss values
                train_loss, train_reg, _ = self.train_epoch(
                t, data['train']['x'], data['train']['y'], e)
                #get validation loss values and DC
                val_loss, val_DC = self.eval(t, data['val']['x'], data['val']['y'])
                #calculate validation mean DC (background is excluded)
                val_mDC = np.mean(val_DC[1:])

                #write to tensorboard
                self.summary.write(t, e, lr, train_loss, train_reg, val_loss, val_mDC)
                if val_mDC > best_DC:
                    #record best validation DC
                    best_DC = val_mDC
                    #reset patience
                    p = 0
                    #remove previously saved model
                    try:
                        save_list = os.listdir(self.weightPath)
                        for files in save_list:
                            if "best_task{}".format(t) in files:
                                os.remove(os.path.join(self.weightPath, files))
                    except:
                        pass
                    #save best model
                    torch.save(self.model.state_dict(), os.path.join(self.weightPath, "best_task{}_e{}".format(t, e + 1)))
                #if validation DC doesn't improve
                elif e >= self.warm_up - 1:
                    p+=1
                    #mark with * in log
                    mark = True
                #informations for printing
                Metrics.val_res(e, val_loss, val_DC, val_mDC, lr, mark)
                mark = False

                #Early stopping
                if p >= self.patience:
                    logging.info("Epoch {}: Early Stop...".format(e+1))
                    break           
        #Keyboard interrupt, save model
        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), os.path.join(self.weightPath, "Interrupt_task{}_e{}".format(t, e+1)))
            logging.info("Stop Training on Task {}.".format(t), exc_info=True)
        #save model (last epoch)
        torch.save(self.model.state_dict(), os.path.join(self.weightPath, "task{}_e{}".format(t, e + 1)))
        #Restore best model
        logging.debug("Restore best validation model...")
        try:
            best_model = torch.load(os.path.join(self.weightPath, "best_task{}_e{}".format(t, e - self.patience + 1)))
        except FileNotFoundError:
            best = os.listdir(self.weightPath)
            for files in best:
                if "best_task{}".format(t) in files:
                    best = files
                elif "task{}".format(t) in files:
                    best = files
            best_model = torch.load(os.path.join(self.weightPath, "{}".format(best)))
        self.model.load_state_dict(best_model)
        del best_model
        torch.cuda.empty_cache()

        if self.appr != "singleTask":
            #number of task
            task = torch.LongTensor([t]).cuda()
            #get attention masks
            mask = self.model.mask(task, s = self.smax)
            for i in range(len(mask)):
                mask[i] = mask[i].detach().clone()
            #cummulative masks
            if t == 0:
                self.mask_pre = mask
            else:
                for i in range(len(self.mask_pre)):
                    self.mask_pre[i] = torch.max(self.mask_pre[i], mask[i])
            del mask
            torch.cuda.empty_cache()
            #reversed masks
            self.mask_back = {}
            for name, _ in self.model.named_parameters():
                values = self.model.get_view_for(name, self.mask_pre)
                if values is not None:
                    self.mask_back[name] = 1 - values

            #initialize unused weightings
            if self.init:
                init_weight = torch.load(os.path.join(self.weightPath, "w_init"))
                init_weight = SetWeights.unused_init(self.model, init_weight, self.mask_pre)
                self.model.load_state_dict(init_weight)
                del init_weight
                torch.cuda.empty_cache()

        logging.info("Task {} finished training!".format(t))

    def train_epoch(self, t, x, y, e):
        #initialize
        masks = None
        total_loss, total_reg = 0, 0
        _, classes = self.info[t]
        total_TP, total_FP, total_FN = np.zeros(classes), np.zeros(classes), np.zeros(classes)
        epsilon = 1e-6

        #train mode
        self.model.train()

        # access data randomly
        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()
        iteration = int(np.ceil(len(r) / self.batch))

        # Loop over batches
        with trange(iteration) as tr:
            tr.bar_format = "{desc}{percentage:3.0f}%|{bar:75}| {n_fmt}/{total_fmt} {elapsed}<{remaining} {postfix}"
            tr.ncols = 200
            tr.set_description(f'Epoch{e+1:3d}/{self.epoch}')

            for i in range(0, len(r), self.batch):
                b = r[i:i + self.batch].cpu()
                images = x[b].detach().clone().cuda()
                targets = y[b].detach().clone().cuda()
                task = torch.LongTensor([t]).cuda()

                if self.appr != "singleTask":
                    # Annealing Strategy
                    s = (self.smax - 1 / self.smax) * i / len(r) + 1 / self.smax

                    # Forward & compute loss
                    output, masks = self.model.forward(task, images, s)
                else:
                    output = self.model.forward(images)

                loss, reg = self.criterion(output, targets, masks)

                # Backward
                self.opt.zero_grad()
                loss.backward()
                
                # Gradient Restriction
                if t > 0 and self.appr != "singleTask":
                    for name, para in self.model.named_parameters():
                        if name in self.mask_back:
                            para.grad.data *= self.mask_back[name]
                
                if self.appr == "hat" or self.appr == "ehat_c":
                    # Compensate embedding gradients
                    for n,p in self.model.named_parameters():
                        if n.startswith('e'):
                            num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                            den=torch.cosh(p.data)+1
                            p.grad.data*=self.smax/s*num/den
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)

                self.opt.step()

                pred = torch.argmax(output, dim=1)
                TP, FP, FN = Metrics.confusion(targets, pred, classes)
                total_TP += TP; total_FP += FP; total_FN += FN

                total_loss += loss.item()
                total_reg += reg.item() if isinstance(reg, torch.Tensor) else reg
                
                tr.update(1)
                if self.appr != "singleTask":
                    tr.set_postfix(total_loss=f'{loss: 6.3f}', loss=f'{loss-reg: 6.3f}', reg=f'{reg:6.3f}')
                else:
                    tr.set_postfix(loss = f'{loss: 6.3f}')
                

            torch.cuda.empty_cache()

            mem = psutil.virtual_memory()
            if mem.percent >= 90:
                warnings.warn("Mem usage >= 90%, good to use --lazy_loading to reduce the usage!")

        avg_loss = total_loss / iteration
        avg_reg = total_reg / iteration
        DC = np.zeros(classes)
        for i in range(classes):
            DC[i] = (2 * total_TP[i]) / (2 * total_TP[i] + total_FP[i] + total_FN[i] + epsilon)

        return avg_loss, avg_reg, DC

    def eval(self, t, x, y):
        '''Validation function
        Args:
            task: Task ID
            x: An array of shape [N, C, W, H]. N = Number of training images, C = Channels, W = Width, H = Height
            y: An array of shape [N, 1, W, H]. N = Number of training labels, C = Channels, W = Width, H = Height
        
        Return:
            valiation loss and DC
        '''
        #initialize
        masks = None
        _, classes = self.info[t]
        total_loss = 0
        total_num = 0  # number of validation data
        epsilon = 1e-6 # avoid dividing zero (DC)
        # validation dice coef
        DC, total_TP, total_FP, total_FN = np.zeros(classes), np.zeros(classes), np.zeros(classes), np.zeros(classes)

        self.model.eval()
        with torch.no_grad():
            r = np.arange(x.size(0))
            r = torch.LongTensor(r).cuda()

            for i in range(0, len(r), self.batch):
                if i + self.batch <= len(r):
                    b = r[i:i+self.batch]
                else:
                    b = r[i:]
                b = b.cpu()
                images = x[b].detach().clone().cuda()
                targets = y[b].detach().clone().cuda()
                task = torch.LongTensor([t]).cuda()

                if self.appr != "singleTask":
                    # Forward
                    output,masks = self.model.forward(task ,images , s = self.smax)
                else:
                    output = self.model.forward(images)
                loss, reg = self.criterion(output, targets, masks)
                
                pred = torch.argmax(output, dim = 1)
                TP, FP, FN = Metrics.confusion(targets, pred, classes)
                del pred, images, targets, task
                torch.cuda.empty_cache()

                #validation loss (without regularization loss)
                total_loss += ((loss.detach().cpu().numpy().item() - reg.detach().cpu().numpy().item())*len(b))
                total_num += len(b)
                total_TP += TP
                total_FP += FP
                total_FN += FN
            
            for i in range(classes):
                DC[i] = (2 * total_TP[i]) / (2 * total_TP[i] + total_FP[i] + total_FN[i] + epsilon)

        return total_loss / total_num, DC

    @abstractmethod
    def criterion(self, output, targets, masks = None):
        """ Subclasses MUST implement this, or the code won't run.
        Returns:
                loss, reg
        if no reg, just return 0
        """
        pass
