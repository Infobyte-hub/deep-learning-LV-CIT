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
from util import cal_score


DO_COMPONENTS = False


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
        'evaluate': args.evaluate,
        'resume': args.resume,
        'num_classes': args.num_classes,
        'difficult_examples': True,
        'save_model_path': os.path.join(args.save_model_path, args.model_name, args.data_name),
        'workers': args.workers,
        'epoch_step': args.epoch_step,
        'lr': args.lr,
        'lrp': args.lrp if args.__contains__("lrp") else 1,
    }

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

    pb = torch.sigmoid(engine.state['ap_meter'].scores)
    cat_id = val_dataset.get_cat2id()
    id_cat = list(cat_id.keys())
    result = []
    for i in range(len(pb)):
        temp = {"filename": engine.state['names'][i]}
        labels = []
        for j in range(args.num_classes):
            if pb.numpy()[i][j] > args.threshold:
                temp[id_cat[j]] = pb.numpy()[i][j]
                labels.append(id_cat[j])
            else:
                temp[id_cat[j]] = -1
        temp["labels_gt"] = "|".join(sorted(
            [id_cat[idx] for idx, value in enumerate(engine.state['ap_meter'].targets[i]) if value == 1]
        ))
        temp["labels"] = "|".join(sorted(labels))
        temp["pass"] = 1 if temp["labels_gt"] == temp["labels"] else 0
        result.append(temp)
    result = pd.DataFrame(result)
    accuracy = result.groupby(by="labels_gt", as_index=False, sort=False)["labels_gt", "pass"].mean()
    result["score"] = result.apply(
        lambda x: cal_score(
            x["labels_gt"], x["labels"], args.num_classes, args.way_num, cat_id
        ), axis=1
    )
    accuracy.rename(columns={"labels_gt": "labels_gt", "pass": "accuracy"}, inplace=True)
    return result, accuracy, engine.state['map']


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
        result, accuracy, map = asl_validate_multi(val_loader, model, args, 1)
    else:
        raise NotImplementedError("model {} is not implemented".format(args.model_name))
    return result, accuracy, map


def runner(args, times=5):
    args.use_gpu = torch.cuda.is_available()
    model = get_model(args.model_name)
    data_root = args.data
    res_root = args.res_path
    for ca_type in args.covering_array_type:
        args.way_num = int(ca_type.split("_")[-1])
        for i in range(0, times):
            args.data = os.path.join(data_root, f"{ca_type}_No{i+1}")
            args.res_path = os.path.join(res_root, f"{ca_type}_No{i+1}")
            if DO_COMPONENTS:
                tasks = [False, True]
            else:
                tasks = [False]
            for components in tasks:
                print(f"covering array type: {ca_type}, No {i+1}, components: {components}")
                if components:
                    args.data = os.path.join(args.data, "components")
                with torch.no_grad():
                    if args.model_name in ['msrn', 'mlgcn']:
                        result, accuracy, map = runner1(args.dataloader, model, args)
                    elif args.model_name in ['asl']:
                        result, accuracy, map = runner2(args.dataloader, model, args)
                    else:
                        raise NotImplementedError("model {} is not implemented".format(args.model_name))

                result_file = os.path.join(
                    args.res_path,
                    f"res_{args.data_name}_{args.model_name}_{ca_type}_No{i+1}{'_components' if components else ''}.csv"
                )
                accuracy_file = os.path.join(
                    args.res_path,
                    f"acc_{args.data_name}_{args.model_name}_{ca_type}_No{i+1}{'_components' if components else ''}.csv"
                )
                if not os.path.exists(os.path.dirname(result_file)):
                    os.makedirs(os.path.dirname(result_file))
                result.to_csv(result_file, index=False)
                # accuracy.to_csv(accuracy_file, index=False)
                print("save result to {}, {}".format(result_file, accuracy_file))
