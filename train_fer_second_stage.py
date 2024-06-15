#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import datetime
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from utils.LossFunctions import *
import clip
from utils.optimizer import build_optimizer
from models.clip_vit import CLIPVIT
from utils.misc import *
from dataloader.data_utils import *
from engine_fer_second_stage_maple import train, test, eval


def main(args):

    if not args.eval:
        setup_seed(args.seed)

        args.is_master = is_main_process()
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(args.gpu)
        #########  RECORD SETTING ###########

        time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        record_name = time + "_" + args.ckpt_path.split("/")[-2][len("mm-dd-hh:mm:ss"):]+ "-loss_" + args.loss_function + \
                        "-ep_" + str(args.epochs) + "-lr_" + str(args.lr) + "-bs_" + str(args.batch_size)
        args.record_path = os.path.join("outputs", "second_stage", record_name)

        if not os.path.exists(args.record_path):
            os.makedirs(args.record_path, exist_ok=True)

        if args.is_master:
            logger = init_log(args, args.record_path)
        else:
            logger = None

        cudnn.benchmark = True  # For speed i.e, cudnn autotuner

        # Build Dataloader
        set_up_datasets(args)
        train_dataset, test_dataset, train_dataloader, test_dataloader = get_dataloader(args)
        get_labelname(args)

        # Init Vision Backbone
        clip_model, _ = clip.load(args.clip_path, jit=False)
        print("loading clip from {}".format(args.clip_path))
        model = CLIPVIT(args, clip_model)
        convert_models_to_fp32(model)

        if args.ckpt_path:
            ckpt = torch.load(args.ckpt_path, map_location="cuda")
            msg = model.load_state_dict(ckpt, strict=False)
            print(msg)

        model = model.to(args.device)

        # Build Optimizer
        optimizer = build_optimizer(args, model, stage='stage2')

        if args.loss_function == 'ce':
            criterion = nn.CrossEntropyLoss()
        elif args.loss_function == 'focal':
            criterion = FocalLoss(class_num=args.classes, device=args.device)
        elif args.loss_function == 'balanced':
            criterion = BalancedLoss(class_num=args.classes, device=args.device)
        elif args.loss_function == 'cosine':
            criterion = CosineLoss()

        # Dump Params
        if is_main_process():

            logger.info("------------------------------------------------------------------")
            logger.info("USING LR SCHEDULER")
            logger.info("------------------------------------------------------------------")
            logger.info(("initial learning rate {}".format(args.lr)))
            logger.info(optimizer)

            write_description_to_folder(os.path.join(args.record_path, "params.txt"), args)

        scaler = GradScaler()

        for epoch in range(args.epochs):
            model.train()
            train(model, optimizer, criterion, train_dataloader, logger, scaler, args, epoch)
            model.eval()
            test(model, criterion, test_dataloader, logger)

    else:

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True  # For speed i.e, cudnn autotuner

        # Build Dataloader
        set_up_datasets(args)
        _, test_dataset, _, test_dataloader = get_dataloader(args)
        get_labelname(args)

        # Init Vision Backbone
        clip_model, _ = clip.load(args.clip_path, device=args.device, jit=False)
        model = CLIPVIT(args, clip_model)
        convert_models_to_fp32(model)
        ckpt = torch.load(args.eval_ckpt, map_location="cuda")
        msg = model.load_state_dict(ckpt, strict=False)
        print("Image Encoder Load Info: ", msg)
        model = model.to(args.device)

        scaler = GradScaler()
        model.eval()
        eval(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name",                   type=str,   default=None    )
    parser.add_argument("--seed",                   type=int,   default=42      )
    parser.add_argument("--record_path",            type=str,   default='/home/CEPrompt/record')
    parser.add_argument("--eval",                   action="store_true"         )

    parser.add_argument('--classes',                type=int,   default=7       )
    parser.add_argument('--dataset', type=str,      default='rafdb', choices=['rafdb', 'affectnet', 'affectnet_8'])
    parser.add_argument('--data-path', type=str, default='/data/RAFDB/basic', choices=['/data/RAFDB/basic/', '/data/AffectNet/'])
    
    parser.add_argument("--ckpt-path",              type=str,   default='/data/RAFDB/ckpt/model_epoch_34_acc9221.pth'    )
    
    parser.add_argument("--clip-path", type=str, default='/data/CEPrompt_ckpt/pre-trained_model/ViT-B-16.pt',
                        choices=['/data/CEPrompt_ckpt/pre-trained_model/ViT-B-16.pt', '/data/CEPrompt_ckpt/pre-trained_model/ViT-B-32.pt', '/data/CEPrompt_ckpt/pre-trained_model/ViT-L-14.pt'])

    parser.add_argument("--eval-ckpt",              type=str,   default=None    )
    parser.add_argument("--batch-size",             type=int,   default=128,    )
    parser.add_argument("--test-batch-size",        type=int,   default=30,     )
    parser.add_argument("--epochs",                 type=int,   default=10,     )
    parser.add_argument("--warmup_epochs",          type=int,   default=2,      )
    parser.add_argument("--loss_function",          type=str,   default='ce', choices=['ce', 'focal', 'balanced', 'cosine'])
    parser.add_argument("--lr",                     type=float, default=2e-3,   )
    parser.add_argument("--min_lr",                 type=float, default=1e-8,   )
    parser.add_argument("--weight_decay",           type=float, default=0.0005, )
    parser.add_argument("--workers",                type=int,   default=8,      )
    parser.add_argument("--momentum",               type=float, default=0.9,    )
    parser.add_argument("--input_size",             type=int,   default=224     )
    parser.add_argument("--alpha",                  type=float, default=0.5,    )
    parser.add_argument("--topk",                   type=int,   default=16      )

    parser.add_argument("--stage2_name",            type=str,   default="cat")
    parser.add_argument("--ctxinit",                type=str,   default="a photo of") 
    parser.add_argument("--prompts_depth",          type=int,   default=9      )
    parser.add_argument("--use_class_invariant",    action='store_true'        )
    parser.add_argument("--n_ctx",                  type=int,   default=2      )
    parser.add_argument("--gpu",                    type=int,   default=0      )

    args = parser.parse_args()

    main(args)

