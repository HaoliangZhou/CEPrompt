import os
import numpy as np
import scipy
import torch.nn.functional as F
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
from utils.misc import is_main_process, compute_ACC, Averager
import utils.lr_sched as lrs


##  train function ###
def train(model, optimizer, criterion, dataloader, logger, scaler, args, epoch):
    if is_main_process():
        logger.info("=======================TRAINING MODE, Epoch: {}/{}=======================".format(epoch, args.epochs))
        print("=======================TRAINING MODE, Epoch: {}/{}=======================".format(epoch, args.epochs))
        dataloader = tqdm(dataloader)

    ta = Averager()
    mean_rank_loss = 0

    for i, (train_inputs, train_labels) in enumerate(dataloader):

        lrs.adjust_learning_rate(optimizer, i / len(dataloader) + epoch, args)

        with autocast(enabled=True):

            optimizer.zero_grad()

            train_inputs = train_inputs.cuda()
            train_labels = train_labels.cuda()

            logits, _, _ = model.forward_maple(train_inputs)

            rank_loss = criterion(logits, train_labels)
            loss = rank_loss

            acc = compute_ACC(logits, train_labels)
            ta.add(acc)

            mean_rank_loss += rank_loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)

        scaler.update()

    mean_rank_loss /= len(dataloader)

    learning_rate = optimizer.param_groups[-1]['lr']

    if is_main_process():
        ta = ta.item()
        logger.info("FINETUNING Epoch: {}/{} \tRankLoss: {:.4f}\tLearningRate {:.6f}\tTrain Acc: {:.4f} ".format(epoch, args.epochs, mean_rank_loss, learning_rate, ta))
        print("FINETUNING Epoch: {}/{} \tRankLoss: {:.4f}\tLearningRate {:.6f}\tTrain Acc: {:.4f} ".format(epoch, args.epochs, mean_rank_loss, learning_rate, ta))

        # torch.save(model.state_dict(), os.path.join(args.record_path, "model_{}.pth".format(epoch)))


########### TEST FUNC ###########
def test(model, criterion, dataloader, logger):
    if is_main_process():
        logger.info("-----------------------EVALUATION MODE-----------------------")
        print("-----------------------EVALUATION MODE-----------------------")
        dataloader = tqdm(dataloader)

    va = Averager()
    vl = Averager()
    torch.set_grad_enabled(False)

    for i, (features, labels) in enumerate(dataloader):
        with autocast():
            logits, _, _ = model.forward_maple(features.cuda())

        loss = criterion(logits, labels.cuda())
        vl.add(loss.item())

        acc = compute_ACC(logits.cuda(), labels.cuda())
        va.add(acc)

    torch.set_grad_enabled(True)

    if is_main_process():
        logger.info(f"sample_num:{labels.shape}")
        logger.info("completed calculating predictions over all images")
        va = va.item()
        vl = vl.item()
        logger.info("Test Loss: {:.4f} \t Test ACC: {:.4f}".format(vl, va))
        print("Test Loss: {:.4f} \t Test ACC: {:.4f}".format(vl, va))


def eval(model, dataloader):
    print("-----------------------EVALUATION MODE 2-----------------------")
    dataloader = tqdm(dataloader)

    va = Averager()
    predRST = []
    labelRET = []
    torch.set_grad_enabled(False)

    for i, (features, labels) in enumerate(dataloader):

        with autocast():
            logits, _, _ = model.forward_maple(features.cuda())

        acc = compute_ACC(logits.cuda(), labels.cuda())
        va.add(acc)

        preds_np = np.array(torch.argmax(logits, dim=1).cpu())
        labels_np = np.array(labels)
        predRST.append(preds_np)
        labelRET.append(labels_np)
    # 转换成一维数组
    predRST = np.concatenate(predRST, axis=0)
    labelRET = np.concatenate(labelRET, axis=0)
    # 保存成mat文件, predRST为a列, labelRET为b列
    scipy.io.savemat('predRST.mat', {'a': predRST, 'b': labelRET})

    torch.set_grad_enabled(True)
    # import pdb; pdb.set_trace()

    print("completed calculating predictions over all images")
    va = va.item()
    print("Test ACC: {:.4f}".format(va))




