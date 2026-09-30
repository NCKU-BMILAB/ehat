import numpy as np
import torch
import argparse
#import matplotlib.pyplot as plt
import scipy.io
import random
import os
from loader import Loader
from utils import Metrics, Tools

class Prediction():
    
    def __init__(self, taskcla, model, smax, res, save, batch = 16):
        self.taskcla = taskcla
        self.model = model
        self.batch = batch
        self.smax = smax
        self.res = res
        self.save = save
    
    #continual learning evaluation
    def eval(self, t, x, y):
        _, classes = self.taskcla[t]
        DC, total_TP, total_FP, total_FN = np.zeros(classes), np.zeros(classes), np.zeros(classes), np.zeros(classes)
        total_num=0
        epsilon = 1e-6
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.batch):
            if i+self.batch <=len(r): b=r[i:i+self.batch]
            else: b=r[i:]
            b = b.cpu()
            images = x[b].clone().detach().cuda()
            targets = y[b].clone().detach().cuda()
            task = torch.LongTensor([t]).cuda()

            # Forward
            output, masks=self.model.forward(task,images,s=self.smax)

            predions = torch.argmax(output, dim = 1)

            #save the predicted figures as npy
            if self.save:
                for i, image in enumerate(images):
                    image = image.cpu().numpy()
                    image = np.squeeze(image, 0)
                    np.save(os.path.join(self.res, "image/task{}/{:0>4}.npy".format(t, total_num + i)), image)

                for i, (pred, label) in enumerate(zip(predions, targets)):
                    pred = pred.cpu().numpy()
                    label = label.cpu().numpy()
                    np.save(os.path.join(self.res, "prediction/task{}/{:0>4}.npy".format(t, total_num + i)), pred)
                    np.save(os.path.join(self.res, "label/task{}/{:0>4}.npy".format(t, total_num + i)), label)

            TP, FP, FN = Metrics.confusion(targets, predions, classes)

            # Log
            total_num+=len(b)
            total_TP += TP
            total_FP += FP
            total_FN += FN
        
        for i in range(classes):
            DC[i] = (2 * total_TP[i]) / (2 * total_TP[i] + total_FP[i] + total_FN[i] + epsilon)

        return DC

    #signle-task evaluation
    def single_eval(self, t, x, y):
        #initialize
        _, classes = self.taskcla[t]
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

                # Forward
                output = self.model.forward(images)
                #output = output[t]
                
                predions = torch.argmax(output, dim = 1)

                if self.save:
                    for i, image in enumerate(images):
                        image = image.cpu().numpy()
                        image = np.squeeze(image, 0)
                        np.save(os.path.join(self.res, "image/task{}/{:0>4}.npy".format(t, total_num + i)), image)

                    for i, (pred, label) in enumerate(zip(predions, targets)):
                        pred = pred.cpu().numpy()
                        label = label.cpu().numpy()
                        np.save(os.path.join(self.res, "prediction/task{}/{:0>4}.npy".format(t, total_num + i)), pred)
                        np.save(os.path.join(self.res, "label/task{}/{:0>4}.npy".format(t, total_num + i)), label)

                TP, FP, FN = Metrics.confusion(targets, predions, classes)
                # Log
                total_num += len(b)
                total_TP += TP
                total_FP += FP
                total_FN += FN
            
            for i in range(classes):
                DC[i] = (2 * total_TP[i]) / (2 * total_TP[i] + total_FP[i] + total_FN[i] + epsilon)

        return  DC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', choices = ['ehat', 'hat', 'ehat_c', 'singleTask', 'Unet50', 'Unet25'], type = str, help = "Training strategies.")
    parser.add_argument('--path', type = str, help = "Path to model for prediction. (single-tasks models split by comma)")
    parser.add_argument('--seed', default = 0, type = int, help = "Training order.") #In thesis, single-task training use seed = 5.
    parser.add_argument('--batch', default = 16, type = int, help = "Batch size.")
    parser.add_argument('--size', default = [1, 192, 192], type = int, nargs = '+', help = "Input image size, [C, W, H]")
    parser.add_argument('--smax', default = 1000, type = float, help = "Hyperparameter.")
    parser.add_argument('--save', action = 'store_true', help = "save the predicted results as npy files.")
    parser.add_argument('--res', default = './result', type = str, help = "Path to save the predicted results.")
    parser.add_argument('--fourTasks', action = 'store_true', help = "Use four tasks.")
    parser.add_argument('--clean', action = 'store_true', help = "Clean the directary.")
    parser.add_argument('--listPath', default = './dataset/lists/', type = str, help = "Path to testing data lists.")
    args = parser.parse_args()

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
        model_list = args.path.split(',')
        args.fourTasks = True

    L = Loader(args.size)
    print("Loading data...")
    data, info = L.load_all(seed = args.seed, fourTasks = args.fourTasks)
    print("Done!")
    if args.save:
        if args.clean:
            Tools.clean(args.res)
        Tools.mkdir(args.res)
        Tools.mkdir(os.path.join(args.res, "image"))
        Tools.mkdir(os.path.join(args.res, "prediction"))
        Tools.mkdir(os.path.join(args.res, "label"))
        for i in range(len(info)):
            Tools.mkdir(os.path.join(args.res, "image/task{}".format(i)))
            Tools.mkdir(os.path.join(args.res, "prediction/task{}".format(i)))
            Tools.mkdir(os.path.join(args.res, "label/task{}".format(i)))
    print("Loading model...")
    if args.approach != "singleTask":
        saved_model = torch.load(args.path)
        net = Net(args.size[0], info).cuda()
        net.load_state_dict(saved_model)
        pred = Prediction(info, net, args.smax, args.res, args.save, batch = args.batch)
    print("Done!")
    print("Testing...")
    for t in range(len(info)):
        if args.approach == "singleTask":
            saved_model = torch.load(model_list[t])
            net = Net(args.size[0], info[t][1]).cuda()
            net.load_state_dict(saved_model)
            pred = Prediction(info, net, args.smax, args.res, args.save, batch = args.batch)
        x = data[t]['test']['x']
        y = data[t]['test']['y']
        if args.approach == "singleTask":
            test_DC = pred.single_eval(t,x,y)
        else:
            test_DC = pred.eval(t,x,y)
        test_mDC = np.mean(test_DC[1:])
        print('>>> Test on task {:2d} - {:8s}:  DC = {:5.2f}% <<<'.format(t,data[t]['name'],100*test_mDC))
