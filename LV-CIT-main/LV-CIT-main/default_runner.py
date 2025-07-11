import gc
import numpy as np
import torch.nn.functional
import pandas as pd
import os
import torch.nn.parallel
import torch.optim
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

from models import *


def get_model(model_name):
    models = {
        "msrn": MSRN,
        "mlgcn": ML_GCN,
        "asl": ASL,
    }
    return models[model_name]


def create_engine(model_class, args):
    state = {
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'max_epochs': args.epochs,
        'evaluate': True,
        'resume': args.resume,
        'num_classes': args.num_classes,
        'difficult_examples': True,
        'save_model_path': os.path.join(args.save_model_path, args.model_name, args.data_name),
        'workers': args.workers,
        'epoch_step': args.epoch_step,
        'lr': args.lr
    }
    if args.model_name == 'dsdl':
        state["device_ids"] = args.device_ids

    # create model
    if args.model_name == 'msrn':
        model = model_class(args.num_classes, args.pool_ratio, args.backbone, args.graph_file)
        if args.pretrained:
            model = msrn_load_pretrain_model(model, args)
        if args.use_gpu:
            model.cuda()
    elif args.model_name == 'mlgcn':
        model = model_class(args.num_classes, t=0.4, adj_file=args.graph_file)
    else:
        raise NotImplementedError("model {} is not implemented".format(args.model_name))

    # define loss function (criterion)
    if args.model_name in ['msrn', 'mlgcn']:
        criterion = nn.MultiLabelSoftMarginLoss()
    else:
        raise NotImplementedError("model {} is not implemented".format(args.model_name))

    # define optimizer
    if args.model_name == 'msrn':
        optimizer = torch.optim.SGD(model.get_config_optim(args.lr),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.model_name == 'mlgcn':
        optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("model {} is not implemented".format(args.model_name))

    # create engine
    engines = {
        "msrn": MSRNEngine,
        "mlgcn": MLGCNEngine,
    }
    engine = engines[args.model_name](state)
    return model, criterion, optimizer, engine


def runner1(dataloader_class, model_class, args):
    val_dataset = dataloader_class(args.data, phase=args.phase, inp_name=args.inp_name)
    model, criterion, optimizer, engine = create_engine(model_class, args)
    engine.predict(model, criterion, val_dataset, optimizer)

    temp_cat = []
    cat_id = val_dataset.get_cat2id()
    for item in cat_id:
        temp_cat.append(item)
    _, indec = torch.sort(engine.state['ap_meter'].scores, descending=True)
    pb = torch.nn.functional.sigmoid(engine.state['ap_meter'].scores)
    result = []
    for i in range(len(indec)):
        labels = []
        labels_gt = []
        for j in range(len(cat_id)):
            if pb.numpy()[i][indec.numpy()[i][j]] > args.threshold:
                labels.append(temp_cat[indec.numpy()[i][j]])
            if val_dataset.targets[engine.state['names'][i].split(os.sep)[-1]][j] > 0:
                labels_gt.append(temp_cat[j])
        result.append([engine.state['names'][i].split(os.sep)[-1], "|".join(labels), "|".join(labels_gt)])
    result = pd.DataFrame(result)
    result.rename(columns={0: "filename", 1: "labels", 2: "labels_gt"}, inplace=True)
    return result, engine.state['map']


def create_loader_and_model(dataloader_class, model_class, args):
    if args.model_name == 'asl':
        model = model_class(args)
        normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        val_dataset = dataloader_class(
            args.data, phase=args.phase,
            inp_name=args.inp_name,
            transform=transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        raise NotImplementedError("model {} is not implemented".format(args.model_name))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return val_loader, model


def runner2(dataloader_class, model_class, args):
    val_loader, model = create_loader_and_model(dataloader_class, model_class, args)
    if args.model_name == 'asl':
        result, map = asl_validate_multi(val_loader, model, args)
    else:
        raise NotImplementedError("model {} is not implemented".format(args.model_name))
    return result, map


def runner(args):
    args.use_gpu = torch.cuda.is_available()
    model = get_model(args.model_name)
    if args.model_name in ['msrn', 'mlgcn']:
        result, map = runner1(args.dataloader, model, args)
    elif args.model_name in ['asl']:
        result, map = runner2(args.dataloader, model, args)
    else:
        raise NotImplementedError("model {} is not implemented".format(args.model_name))

    result_file = os.path.join(
        args.res_path,
        r'result_{dataname}_{model_name}.xlsx'.format(
            dataname=args.data_name, model_name=args.model_name
        )
    )
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
    result.to_excel(result_file, index=False)
    print("save result to {}".format(result_file))
