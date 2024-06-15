import os
from torch import nn
import argparse
import torch.backends.cudnn as cudnn
import clip
from utils.misc import *
from dataloader.data_utils import *
from utils.optimizer import build_optimizer
from utils.LossFunctions import *
from models.clip_vit import CLIPVIT
from engine_fer_first_stage import train, test


def main(args):
    setup_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True

    # Init Recoder
    record_name = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + "-" + args.dataset + "-loss_" + args.loss_function + \
                  "-ep_" + str(args.epochs) + "-lr_" + str(args.lr) + "-bs_" + str(args.batch_size) + \
                  "-alp_" + str(args.alpha) + "-lam_" + str(args.lamda) + "-fix_" + str(args.fix_layer) + "-top" + str(args.topk)
    args.record_path = os.path.join("outputs", "first_stage", record_name)
    os.makedirs(args.record_path, exist_ok=True)
    logger = init_log(args, args.record_path)
    write_description_to_folder(os.path.join(args.record_path, "configs.txt"), args)

    # Init DataLoader
    set_up_datasets(args)
    train_dataset, val_dataset, train_dataloader, val_dataloader = get_dataloader(args)
    get_labelname(args)

    # FER
    label_prompts = [f"a photo of a {label_nm} facial expression, a type of facial expression." for label_nm in args.label_nms]
    logger.info("First stage label prompts:{}".format(label_prompts))
    print("First stage label prompts:{}".format(label_prompts))

    label_token = clip.tokenize(label_prompts)

    # Build Model: CLIPVIT
    clip_model, _ = clip.load(args.clip_path, jit=False)
    print("loading clip from {}".format(args.clip_path))
    model = CLIPVIT(args, clip_model)
    convert_models_to_fp32(model)
    model = model.to(args.device)

    # Load CLIP
    clip_model, _ = clip.load(args.clip_path, jit=False)
    clip_model.train()
    clip_model = clip_model.to(args.device)

    # Build Optimizer
    optimizer = build_optimizer(args, model)

    if args.loss_function == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_function == 'focal':
        criterion = FocalLoss(class_num=args.classes, device=args.device)
    elif args.loss_function == 'balanced':
        criterion = BalancedLoss(class_num=args.classes, device=args.device)
    elif args.loss_function == 'cosine':
        criterion = CosineLoss()
    elif args.loss_function == 'gce':
        criterion = GCELoss(args.classes, device=args.device)

    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_acc, train_loss = train(model, clip_model, args, optimizer, criterion, train_dataloader, logger, label_token, epoch)  # raw
        model.eval()
        val_acc, val_loss = test(model, args, criterion, val_dataloader, logger, label_token, epoch)  # raw

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

#norm_pred=None
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",                   type=int,   default=42  )
    parser.add_argument("--record_path",            type=str,   default='/home/CEPrompt/record')

    parser.add_argument('--classes', type=int, default=7)
    parser.add_argument('--dataset', type=str, default='rafdb', choices=['rafdb', 'affectnet', 'affectnet_8'])
    parser.add_argument('--data-path', type=str, default='/data/basic/', choices=['/data/RAFDB/basic/', '/data/AffectNet/'])

    parser.add_argument("--clip-path", type=str, default='/data/CEPrompt_ckpt/pre-trained_model/ViT-B-16.pt',
                        choices=['/data/CEPrompt_ckpt/pre-trained_model/ViT-B-16.pt', '/data/CEPrompt_ckpt/pre-trained_model/ViT-B-32.pt', '/data/CEPrompt_ckpt/pre-trained_model/ViT-L-14.pt'])
    
    parser.add_argument("--batch-size",             type=int,   default=128,    )
    parser.add_argument("--test-batch-size",        type=int,   default=50,     )
    parser.add_argument("--epochs",                 type=int,   default=50,     )
    parser.add_argument("--warmup_epochs",          type=int,   default=5,      )
    parser.add_argument("--loss_function",          type=str,   default='ce', choices=['ce', 'focal', 'balanced', 'cosine', 'gce'])
    parser.add_argument("--lr",                     type=float, default=1e-4,   )
    parser.add_argument("--min_lr",                 type=float, default=1e-8,   )
    parser.add_argument("--weight_decay",           type=float, default=0.0005, )
    parser.add_argument("--workers",                type=int,   default=8,      )
    parser.add_argument("--momentum",               type=float, default=0.90,   )
    parser.add_argument("--alpha",                  type=float, default=0.5,    )
    parser.add_argument("--lamda",                  type=float, default=1,      )
    parser.add_argument("--stage2_name",            type=str,   default=None,   )
    parser.add_argument("--input_size",             type=int,   default=224,    )
    parser.add_argument("--gpu",                    type=int,   default=1      )
    parser.add_argument("--layer_decay",            type=float, default=0.65    )
    parser.add_argument("--fix_layer",              type=int,   default=2      )
    parser.add_argument("--topk",                   type=int,   default=16      )

    args = parser.parse_args()

    main(args)
    
