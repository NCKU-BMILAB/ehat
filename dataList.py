import os
import random
import yaml
from utils import Tools

class dataList():
    def __init__(self, train, val, test):
        self.test = test 
        self.val = train * val

    def _RV(self, from_split = False):
        train_list = open("./dataset/lists/train_list/RV_train.txt", "w")
        val_list = open("./dataset/lists/val_list/RV_val.txt", "w")
        test_list = open("./dataset/lists/test_list/RV_test.txt", "w")
        if from_split:
            with open('./dataset/lists/split_list/RV_split.txt', 'r') as file:
                split = yaml.load(file, Loader=yaml.FullLoader)
            train_ID = split['train']
            val_ID = split['val']
            test_ID = split['test']
        else:
            #patient ID
            split_list = open("./dataset/lists/split_list/RV_split.txt", "w")
            ID = ["P{:0>2d}".format(i) for i in range(1, 17)]
            #randomly split to three set
            random.shuffle(ID)
            num_test = round(len(ID) * self.test)
            num_val = round(len(ID) * self.val)
            num_train = len(ID) - num_test - num_val
            train_ID = ID[:num_train]
            val_ID = ID[num_train: num_train + num_val]
            test_ID = ID[-num_test:]
            split_list.write("train: {}\nval: {}\ntest: {}".format(train_ID, val_ID, test_ID))
            split_list.close()
        #data path
        data_path = "./dataset/exp_data/RV/2DSlice/data"
        aug_path = "./dataset/exp_data/RV/2DSlice/augmentation"
        augFolder = os.listdir(aug_path)
        augFolder.sort()
        print("RV >> Augmentation:", augFolder)
        #write to list
        imagePath = os.path.join(data_path, "images")
        maskPath = os.path.join(data_path, "masks")
        imageName = os.listdir(imagePath)
        imageName.sort()
        maskName = os.listdir(maskPath)
        maskName.sort()
        for image, mask in zip(imageName, maskName):
            if image[:3] in train_ID:
                train_list.write(os.path.join(imagePath, image) + " ")
                train_list.write(os.path.join( maskPath, mask) + "\n")
            elif image[:3] in test_ID:
                test_list.write(os.path.join(imagePath, image) + " ")
                test_list.write(os.path.join(maskPath, mask) + "\n")
            else:
                val_list.write(os.path.join(imagePath, image) + " ")
                val_list.write(os.path.join(maskPath, mask) + "\n")
            #augmented data
        for augType in augFolder:
            prefix = os.path.join(aug_path, augType)
            imagePath = os.path.join(prefix, "images")
            maskPath = os.path.join(prefix, "masks")
            imageName = os.listdir(imagePath)
            imageName.sort()
            maskName = os.listdir(maskPath)
            maskName.sort()
            for image, mask in zip(imageName, maskName):
                if image[:3] in train_ID:
                    train_list.write(os.path.join(imagePath, image) + " ")
                    train_list.write(os.path.join(maskPath, mask) + "\n")
        train_list.close()
        val_list.close()
        test_list.close()

    def _lumbar(self, from_split = False):
        train_list = open("./dataset/lists/train_list/lumbar_train.txt", "w")
        val_list = open("./dataset/lists/val_list/lumbar_val.txt", "w")
        test_list = open("./dataset/lists/test_list/lumbar_test.txt", "w")
        #patient ID
        if from_split:
            with open('./dataset/lists/split_list/lumbar_split.txt', 'r') as file:
                split = yaml.load(file, Loader=yaml.FullLoader)
            train_ID = split['train']
            val_ID = split['val']
            test_ID = split['test']
        else:
            split_list = open("./dataset/lists/split_list/lumbar_split.txt", "w")
            ID = ["{:0>2d}".format(i) for i in range(11, 26)]
            #randomly split to three set
            random.shuffle(ID)
            num_test = round(len(ID) * self.test)
            num_val = round(len(ID) * self.val)
            num_train = len(ID) - num_test - num_val
            train_ID = ID[:num_train]
            val_ID = ID[num_train: num_train + num_val]
            test_ID = ID[-num_test:]
            split_list.write("train: {}\nval: {}\ntest: {}".format(train_ID, val_ID, test_ID))
            split_list.close()
        data_path = "./dataset/exp_data/lumbar/2DSlice/data"
        aug_path = "./dataset/exp_data/lumbar/2DSlice/augmentation"
        dataFolder = os.listdir(data_path)
        dataFolder.sort()
        augFolder = os.listdir(aug_path)
        augFolder.sort()
        print("lumbar >> Augmentation:", augFolder)
        #write to list
        for subject in dataFolder:
            imagePath = os.path.join(data_path, subject, "images")
            maskPath = os.path.join(data_path, subject, "masks")
            imageName = os.listdir(imagePath)
            imageName.sort()
            maskName = os.listdir(maskPath)
            maskName.sort()
            if subject[-2:] in train_ID:
                for image, mask in zip(imageName, maskName):
                    train_list.write(os.path.join(imagePath, image) + " ")
                    train_list.write(os.path.join(maskPath, mask) + "\n")
            elif subject[-2:] in test_ID:
                for image, mask in zip(imageName, maskName):
                    test_list.write(os.path.join(imagePath, image) + " ")
                    test_list.write(os.path.join(maskPath, mask) + "\n")
            else:
                for image, mask in zip(imageName, maskName):
                    val_list.write(os.path.join(imagePath, image) + " ")
                    val_list.write(os.path.join(maskPath, mask) + "\n")
        for augType in augFolder:
            prefix = os.path.join(aug_path, augType)
            prefixFolder = os.listdir(prefix)
            prefixFolder.sort()
            for subject in prefixFolder:
                imagePath = os.path.join(prefix, subject, "images")
                maskPath = os.path.join(prefix, subject, "masks")
                imageName = os.listdir(imagePath)
                imageName.sort()
                maskName = os.listdir(maskPath)
                maskName.sort()
                if subject[-2:] in train_ID:
                    for image, mask in zip(imageName, maskName):
                        train_list.write(os.path.join(imagePath, image) + " ")
                        train_list.write(os.path.join(maskPath, mask) + "\n")
        train_list.close()
        val_list.close()
        test_list.close()

    def _tumor(self, from_split = False):
        train_list = open("./dataset/lists/train_list/tumor_train.txt", "w")
        val_list = open("./dataset/lists/val_list/tumor_val.txt", "w")
        test_list = open("./dataset/lists/test_list/tumor_test.txt", "w")      
        #patient ID
        if from_split:
            with open('./dataset/lists/split_list/tumor_split.txt', 'r') as file:
                split = yaml.load(file, Loader=yaml.FullLoader)
            train_ID = split['train']
            val_ID = split['val']
            test_ID = split['test']
        else:
            split_list = open("./dataset/lists/split_list/tumor_split.txt", "w")
            ID = os.listdir("./dataset/exp_data/tumor/images")
            for i, j in enumerate(ID):
                ID[i] = j.split('.')[0]
            ID.sort()        
            TCIA_ID = []
            CBICA_ID = []
            for name in ID:
                if "TCIA" in name:
                    TCIA_ID.append(name)
                if "CBICA" in name:
                    CBICA_ID.append(name)
            random.shuffle(TCIA_ID)
            random.shuffle(CBICA_ID)
            TCIA_ID_50 = TCIA_ID[:50]
            CBICA_ID_50 = CBICA_ID[:50]
            num_test = round((len(TCIA_ID_50) + len(CBICA_ID_50)) * self.test)
            num_val = round((len(TCIA_ID_50) + len(CBICA_ID_50)) * self.val)
            num_train = (len(TCIA_ID_50) + len(CBICA_ID_50)) - num_test - num_val
            train_ID = TCIA_ID_50[:num_train//2] + CBICA_ID_50[:num_train-num_train//2]
            val_ID = TCIA_ID_50[num_train//2:num_train//2 + num_val//2] + CBICA_ID_50[num_train//2: num_train//2 + (num_val - num_val//2)]
            test_ID = TCIA_ID_50[-num_test//2:] + CBICA_ID_50[-(num_test - num_test//2):]
            test_ID = test_ID + TCIA_ID[50:] + CBICA_ID[50:]
            random.shuffle(train_ID)
            split_list.write("train: {}\nval: {}\ntest: {}".format(train_ID, val_ID, test_ID))
            split_list.close()
        data_path = "./dataset/exp_data/tumor/2DSlice/data"
        aug_path = "./dataset/exp_data/tumor/2DSlice/augmentation"
        dataFolder = os.listdir(data_path)
        dataFolder.sort()
        p = 0
        for i, name in enumerate(dataFolder):
            if "TCIA" in name or "CBICA" in name:
                dataFolder[p] = name
                p+=1
        dataFolder = dataFolder[:p]
        augFolder = os.listdir(aug_path)
        augFolder.sort()
        print("tumor >> Augmentation:", augFolder)
        #write to list
        for subject in dataFolder:
            imagePath = os.path.join(data_path, subject, "images")
            maskPath = os.path.join(data_path, subject, "masks")
            imageName = os.listdir(imagePath)
            imageName.sort()
            maskName = os.listdir(maskPath)
            maskName.sort()
            if subject in train_ID:
                for image, mask in zip(imageName, maskName):
                    train_list.write(os.path.join(imagePath, image) + " ")
                    train_list.write(os.path.join(maskPath, mask) + "\n")
            elif subject in test_ID:
                for image, mask in zip(imageName, maskName):
                    test_list.write(os.path.join(imagePath, image) + " ")
                    test_list.write(os.path.join(maskPath, mask) + "\n")
            else:
                for image, mask in zip(imageName, maskName):
                    val_list.write(os.path.join(imagePath, image) + " ")
                    val_list.write(os.path.join(maskPath, mask) + "\n")
        for augType in augFolder:
            prefix = os.path.join(aug_path, augType)
            prefixFolder = os.listdir(prefix)
            prefixFolder.sort()
            for subject in prefixFolder:
                imagePath = os.path.join(prefix, subject, "images")
                maskPath = os.path.join(prefix, subject, "masks")
                imageName = os.listdir(imagePath)
                imageName.sort()
                maskName = os.listdir(maskPath)
                maskName.sort()
                if subject in train_ID:
                    for image, mask in zip(imageName, maskName):
                        train_list.write(os.path.join(imagePath, image) + " ")
                        train_list.write(os.path.join(maskPath, mask) + "\n")
        train_list.close()
        val_list.close()
        test_list.close()

    def _lumbar_MR(self, from_split = False):
        train_list = open("./dataset/lists/train_list/lumbar_MR_train.txt", "w")
        val_list = open("./dataset/lists/val_list/lumbar_MR_val.txt", "w")
        test_list = open("./dataset/lists/test_list/lumbar_MR_test.txt", "w")
        if from_split:
            with open('./dataset/lists/split_list/lumbar_MR_split.txt', 'r') as file:
                split = yaml.load(file, Loader=yaml.FullLoader)
            train_ID = split['train']
            val_ID = split['val']
            test_ID = split['test']
        else:
            split_list = open("./dataset/lists/split_list/lumbar_MR_split.txt", "w")
            #patient ID
            ID = ["{:0>2d}".format(i) for i in range(1, 24)]
            #randomly split to three set
            random.shuffle(ID)
            num_test = round(len(ID) * self.test)
            num_val = round(len(ID) * self.val)
            num_train = len(ID) - num_test - num_val
            train_ID = ID[:num_train]
            val_ID = ID[num_train: num_train + num_val]
            test_ID = ID[-num_test:]
            split_list.write("train: {}\nval: {}\ntest: {}".format(train_ID, val_ID, test_ID))
            split_list.close()
        data_path = "./dataset/exp_data/lumbar_MR/2DSlice/data"
        aug_path = "./dataset/exp_data/lumbar_MR/2DSlice/augmentation"
        dataFolder = os.listdir(data_path)
        dataFolder.sort()
        augFolder = os.listdir(aug_path)
        augFolder.sort()
        print("lumbar_MR >> Augmentation:", augFolder)
        #write to list
        for subject in dataFolder:
            imagePath = os.path.join(data_path, subject, "images")
            maskPath = os.path.join(data_path, subject, "masks")
            imageName = os.listdir(imagePath)
            imageName.sort()
            maskName = os.listdir(maskPath)
            maskName.sort()
            if subject[-2:] in train_ID:
                for image, mask in zip(imageName, maskName):
                    train_list.write(os.path.join(imagePath, image) + " ")
                    train_list.write(os.path.join(maskPath, mask) + "\n")
            elif subject[-2:] in test_ID:
                for image, mask in zip(imageName, maskName):
                    test_list.write(os.path.join(imagePath, image) + " ")
                    test_list.write(os.path.join(maskPath, mask) + "\n")
            else:
                for image, mask in zip(imageName, maskName):
                    val_list.write(os.path.join(imagePath, image) + " ")
                    val_list.write(os.path.join(maskPath, mask) + "\n")
        for augType in augFolder:
            prefix = os.path.join(aug_path, augType)
            prefixFolder = os.listdir(prefix)
            prefixFolder.sort()
            for subject in prefixFolder:
                imagePath = os.path.join(prefix, subject, "images")
                maskPath = os.path.join(prefix, subject, "masks")
                imageName = os.listdir(imagePath)
                imageName.sort()
                maskName = os.listdir(maskPath)
                maskName.sort()
                if subject[-2:] in train_ID:
                    for image, mask in zip(imageName, maskName):
                        train_list.write(os.path.join(imagePath, image) + " ")
                        train_list.write(os.path.join(maskPath, mask) + "\n")
        train_list.close()
        val_list.close()
        test_list.close()

    def reallocate(self):
        Tools.makedirs("./dataset/lists")
        Tools.mkdir("./dataset/lists/train_list")
        Tools.mkdir("./dataset/lists/val_list")
        Tools.mkdir("./dataset/lists/test_list")
        Tools.mkdir("./dataset/lists/split_list")
        self._RV()
        self._lumbar()
        self._tumor()
        self._lumbar_MR()

    @classmethod
    def make_from_split_list(cls):
        Tools.mkdir("./dataset/lists/train_list")
        Tools.mkdir("./dataset/lists/val_list")
        Tools.mkdir("./dataset/lists/test_list")
        cls._RV(cls, from_split=True)
        cls._lumbar(cls,from_split=True)
        cls._tumor(cls,from_split=True)
        cls._lumbar_MR(cls,from_split=True)

