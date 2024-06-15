import numpy as np
import torch
from dataloader.sampler import ImbalancedDatasetSampler


def set_up_datasets(args):
    if args.dataset == 'rafdb':
        import dataloader.rafdb.rafdb as Dataset
    if args.dataset == 'affectnet':
        import dataloader.affectnet.affectnet as Dataset
    if args.dataset == 'affectnet_8':
        import dataloader.affectnet.affectnet_8 as Dataset
    args.Dataset = Dataset
    return args


def get_dataloader(args):
    class_index = np.arange(args.classes)
    if args.dataset == 'rafdb':
        trainset = args.Dataset.RafDB(root=args.data_path, train=True, index=class_index,  args=args)
        testset = args.Dataset.RafDB(root=args.data_path, train=False, index=class_index, args=args)
    if args.dataset == 'affectnet':
        trainset = args.Dataset.AffectNet(root=args.data_path, train=True, index=class_index)
        testset = args.Dataset.AffectNet(root=args.data_path, train=False, index=class_index)
    if args.dataset == 'affectnet_8':
        trainset = args.Dataset.AffectNet(root=args.data_path, train=True, index=class_index)
        testset = args.Dataset.AffectNet(root=args.data_path, train=False, index=class_index)
    print('trainset size: ', len(trainset))
    print('testset size: ', len(testset))

    if args.dataset == 'affectnet' or args.dataset == 'affectnet_8':
        print('Imbalanced Dataset Sampling...')
        trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                                  sampler=ImbalancedDatasetSampler(trainset),
                                                  batch_size=args.batch_size,
                                                  num_workers=args.workers,
                                                  pin_memory=True
                                                  )
        testloader = torch.utils.data.DataLoader(dataset=testset,
                                                 # sampler=ImbalancedDatasetSampler(testset),
                                                 shuffle=False,
                                                 batch_size=args.test_batch_size,
                                                 num_workers=args.workers,
                                                 pin_memory=True
                                                 )
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=args.workers,
                                                  pin_memory=True
                                                  )
        testloader = torch.utils.data.DataLoader(dataset=testset,
                                                 batch_size=args.test_batch_size,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True
                                                 )
    return trainset, testset, trainloader, testloader


def get_labelname(args):
    if args.dataset == 'rafdb':
        label_nms = ['surprise', 'fear', 'disgusted', 'happy', 'sad', 'anger', 'neutral']
    if args.dataset == 'affectnet':
        label_nms = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgusted', 'anger']
    print(args.dataset, 'dataset, label_nms: ', label_nms)
    args.label_nms = label_nms
    return args