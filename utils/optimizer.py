import torch

import utils.lr_decay as lrd

def build_optimizer(args, model, stage=None):

    if stage == None:
        params_name = None

        params, param_group_names = lrd.param_groups_lrd(model, args.fix_layer, args.weight_decay,
            layer_decay=args.layer_decay)

        params_name = []
        for k, v in param_group_names.items():
            params_name += v["params"]

        optimizer = torch.optim.AdamW(params, lr=args.lr)

        for name, param in model.named_parameters():
            if name not in params_name:
                param.requires_grad = False

    elif stage == "stage2":
        train_param_stage2 = []
        if args.stage2_name == "coop":  # coop
            for name, param in model.named_parameters():
                if "ctx" in name: 
                    train_param_stage2.append(param)
                    print(name, param.shape)
                else:
                    param.requires_grad = False
        elif args.stage2_name == "cat":  # our cat
            for name, param in model.named_parameters():
                if "prompt_learner" in name:
                    train_param_stage2.append(param)
                    print(name, param.shape)
                else:
                    param.requires_grad = False

        optimizer = torch.optim.SGD(train_param_stage2, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    return optimizer