import os
import os.path as osp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AffectNet(Dataset):
    def __init__(self, root='/data/AffectNet/', train=True,
                 index_path=None, index=None):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self._pre_operate(self.root)

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)), ], p=0.7),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        image_file = os.path.join(root, 'affectnet_info/images_8cls.txt')
        split_file = os.path.join(root, 'affectnet_info/train_test_split_8cls.txt')
        class_file = os.path.join(root, 'affectnet_info/image_class_labels_8cls.txt')
        id2image = self.list2dict(self.text_read(image_file))
        id2train = self.list2dict(self.text_read(split_file))  # 1: train images; 0: test iamges
        id2class = self.list2dict(self.text_read(class_file))
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self.data = []
        self.targets = []
        self.data2label = {}  # dict:{path:label}
        if self.train:
            for k in train_idx:
                image_path = os.path.join(id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]))
                self.data2label[image_path] = (int(id2class[k]))
            self.class_num = self.get_class_num(self.targets)
            print('test_class_num:', self.class_num)
        else:
            for k in test_idx:
                image_path = os.path.join(id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]))
                self.data2label[image_path] = (int(id2class[k]))
            self.class_num = self.get_class_num(self.targets)
            print('test_class_num:', self.class_num)

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def get_class_num(self, targets):
        class_num = np.zeros(8)
        for i in targets:
            class_num[i] += 1
        return class_num

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]

        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets

