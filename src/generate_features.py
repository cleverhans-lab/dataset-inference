from __future__ import absolute_import
import sys, time, argparse, ipdb, params, glob, os, json
import numpy as np
import torch
import torch.nn as nn
from funcs import *
from attacks import *
from models import *
from train import epoch_test

'''Threat Models'''
# A) complete model theft 
# --> A.1 Datafree distillation / Zero shot learning
# --> A.2 Fine tuning (on unlabeled data to slightly change decision surface)
# B) Extraction over an API:
# --> B.1 Model extraction using unlabeled data and victim labels
# --> B.2 Model extraction using unlabeled data and victim confidence
# C) Complete data theft:
# --> C.1 Data distillation
# --> C.2 Different architecture/learning rate/optimizer/training epochs
# --> C.3 Coresets
# D) Train a teacher model on a separate dataset (test set)


def get_adversarial_vulnerability(args, loader, model, num_images = 1000):
    train_loss, train_acc, train_n = 0,0,0
    batch_size = 100
    max_iter = num_images/batch_size
    full_dist = []
    ex_skipped = 0
    func = tqdm #if stop == False else lambda x:x
    for i,batch in enumerate(loader):
        if args.regressor_embed == 1: ##We need an extra set of `distinct images for training the confidence regressor
            if(ex_skipped < num_images):
                y = batch[1]
                ex_skipped += y.shape[0]
                continue
        for attack in [pgd_l1, pgd_l2, pgd_linf]:
            for target_i in range(10):
                X,y = batch[0].to(device), batch[1].to(device) 
                delta = attack(model,X,y,args,target = y*0 + target_i) if attack is not None else 0
                yp = model(X+delta) 
                yp = yp[0] if len(yp)==4 else yp
                loss = nn.CrossEntropyLoss()(yp,y)
                train_loss += loss.item()*y.size(0)
                train_acc += (yp.max(1)[1] == y).sum().item()
                train_n += y.size(0)
                distance_dict = {pgd_linf: norms_linf_squeezed, pgd_l1: norms_l1_squeezed, pgd_l2: norms_l2_squeezed}
                distances = distance_dict[attack](delta)
                full_dist.append(distances.cpu().detach())
        if i+1>=max_iter:
            break
    
    full = [x.view(-1,1) for x in full_dist]
    full_d = torch.cat(full, dim = 1)
        
    return train_loss / train_n, train_acc / train_n, full_d

def get_random_label_only(args, loader, model, num_images = 1000):
    print("Getting random attacks")
    batch_size = args.batch_size
    max_iter = num_images/batch_size
    lp_dist = [[],[],[]]
    ex_skipped = 0
    for i,batch in enumerate(loader):
        if args.regressor_embed == 1: ##We need an extra set of `distinct images for training the confidence regressor
            if(ex_skipped < num_images):
                y = batch[1]
                ex_skipped += y.shape[0]
                continue
        for j,distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            for target_i in range(10): #5 random starts
                X,y = batch[0].to(device), batch[1].to(device) 
                args.distance = distance
                # args.lamb = 0.0001
                preds = model(X)
                targets = None
                delta = rand_steps(model, X, y, args, target = targets)
                yp = model(X+delta) 
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim = 1)
            lp_dist[j].append(temp_dist) 
        if i+1>=max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim = 0).unsqueeze(-1) for i in range(3)]    
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim = -1); print(full_d.shape)
        
    return full_d

def get_topgd_vulnerability(args, loader, model, num_images = 1000):
    batch_size = args.batch_size
    max_iter = num_images/batch_size
    lp_dist = [[],[],[]]
    ex_skipped = 0
    for i,batch in enumerate(loader):
        if args.regressor_embed == 1: ##We need an extra set of `distinct images for training the confidence regressor
            if(ex_skipped < num_images):
                y = batch[1]
                ex_skipped += y.shape[0]
                continue
        for j,distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            for target_i in range(10):
                X,y = batch[0].to(device), batch[1].to(device) 
                args.distance = distance
                # args.lamb = 0.0001
                preds = model(X)
                tgt = target_i + 1 if args.dataset == "CIFAR100" else target_i
                targets = torch.argsort(preds, dim=-1, descending=True)[:,tgt]
                delta = mingd(model, X, y, args, target = targets)
                yp = model(X+delta) 
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim = 1)
            lp_dist[j].append(temp_dist) 
        if i+1>=max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim = 0).unsqueeze(-1) for i in range(3)]    
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim = -1); print(full_d.shape)
        
    return full_d

def get_mingd_vulnerability(args, loader, model, num_images = 1000):
    batch_size = args.batch_size
    max_iter = num_images/batch_size
    lp_dist = [[],[],[]]
    ex_skipped = 0
    for i,batch in enumerate(loader):
        if args.regressor_embed == 1: ##We need an extra set of `distinct images for training the confidence regressor
            if(ex_skipped < num_images):
                y = batch[1]
                ex_skipped += y.shape[0]
                continue
        for j,distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            for target_i in range(args.num_classes):
                X,y = batch[0].to(device), batch[1].to(device) 
                args.distance = distance
                # args.lamb = 0.0001
                delta = mingd(model, X, y, args, target = y*0 + target_i)
                yp = model(X+delta) 
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim = 1)
            lp_dist[j].append(temp_dist) 
        if i+1>=max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim = 0).unsqueeze(-1) for i in range(3)]    
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim = -1); print(full_d.shape)
        
    return full_d

def feature_extractor(args):
    if args.dataset != "ImageNet":
        train_loader, test_loader = get_dataloaders(args.dataset, args.batch_size, pseudo_labels = False, train_shuffle = False)
        student, _ = get_student_teacher(args) #teacher is not needed
        location = f"{args.model_dir}/final.pt"
        try:
            student = student.to(args.device)
            student.load_state_dict(torch.load(location, map_location = args.device)) 
        except:
            student = nn.DataParallel(student).to(args.device)
            student.load_state_dict(torch.load(location, map_location = args.device))   
    else:
        train_loader, test_loader = get_dataloaders(args.dataset, args.batch_size, normalize = True, pseudo_labels = False, train_shuffle = False)
        import torchvision.models as models
        imagenet_arch = {"alexnet":models.alexnet, "inception":models.inception_v3,"wrn": models.wide_resnet50_2}
        I_Arch = imagenet_arch[args.imagenet_architecture]
        student = I_Arch(pretrained=True)
        student = nn.DataParallel(student).to(args.device)
    
    student.eval()
    
    # _, train_acc  = epoch_test(args, train_loader, student)
    _, test_acc   = epoch_test(args, test_loader, student, stop = True)
    print(f'Model: {args.model_dir} | \t Test Acc: {test_acc:.3f}')    
    
    mapping = {'pgd':get_adversarial_vulnerability, 'topgd':get_topgd_vulnerability, 'mingd':get_mingd_vulnerability, 'rand': get_random_label_only}

    func = mapping[args.feature_type]

    test_d = func(args, test_loader, student)
    print(test_d)
    torch.save(test_d, f"{args.file_dir}/test_{args.feature_type}_vulnerability_2.pt")
    
    train_d = func(args, train_loader, student)
    print(train_d)
    torch.save(train_d, f"{args.file_dir}/train_{args.feature_type}_vulnerability_2.pt")

    
def get_student_teacher(args):
    w_f = 2 if args.dataset == "CIFAR100" else 1
    net_mapper = {"CIFAR10":WideResNet, "CIFAR100":WideResNet, "AFAD":resnet34, "SVHN":ResNet_8x}
    Net_Arch = net_mapper[args.dataset]
    teacher = None; mode = args.mode
    # ['zero-shot', 'prune', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher']
    deep_full = 34 if args.dataset in ["SVHN", "AFAD"] else 28
    deep_half = 18 if args.dataset in ["SVHN", "AFAD"] else 16
    if mode == 'zero-shot':
        student = Net_Arch(n_classes = args.num_classes, depth=deep_half, widen_factor=1, normalize = args.normalize)
    
    elif mode == "prune":
        raise("Not handled")

    elif mode == "random":
        assert (args.dataset == "SVHN")
        # python generate_features.py --feature_type rand --dataset SVHN --batch_size 500 --mode random --normalize 1 --model_id random_normalized
        student =  Net_Arch(n_classes = args.num_classes, depth=deep_half, widen_factor=8, normalize = args.normalize)
    
    elif mode == "fine-tune":
        # python generate_features.py --batch_size 500 --mode fine-tune --normalize 0 --model_id fine-tune_unnormalized
        # python generate_features.py --batch_size 500 --mode fine-tune --normalize 1 --model_id fine-tune_normalized
        student =  Net_Arch(n_classes = args.num_classes, depth=deep_full, widen_factor=10, normalize = args.normalize)

    elif mode in ["extract-label", "extract-logit"]:
        # python generate_features.py --batch_size 500 --mode extract-label --normalize 0 --model_id extract-label_unnormalized
        # python generate_features.py --batch_size 500 --mode extract-label --normalize 1 --model_id extract-label_normalized
        student =  Net_Arch(n_classes = args.num_classes, depth=deep_half,widen_factor=w_f, normalize = args.normalize)

    elif mode in ["distillation", "independent"]:
        dR = 0.3 if mode == "independent" else 0.0
        # python generate_features.py --batch_size 500 --mode distillation --normalize 0 --model_id distillation_unnormalized
        # python generate_features.py --batch_size 500 --mode distillation --normalize 1 --model_id distillation_normalized
        student =  Net_Arch(n_classes = args.num_classes, depth=deep_half, widen_factor=w_f, normalize = args.normalize, dropRate = dR)
    
    elif mode == "pre-act-18":
        student = PreActResNet18(num_classes = args.num_classes, normalize = args.normalize)
        
    else:
        # python generate_features.py --feature_type rand --dataset SVHN --batch_size 500 --mode teacher --normalize 1 --model_id teacher_normalized
        # python generate_features.py --batch_size 500 --mode teacher --normalize 0 --model_id teacher_unnormalized --dataset CIFAR10
        # python generate_features.py --batch_size 500 --mode teacher --normalize 1 --model_id teacher_normalized --dataset CIFAR10
        student =  Net_Arch(n_classes = args.num_classes, depth=deep_full, widen_factor=10, normalize = args.normalize, dropRate = 0.3)
        
        #Alternate student models: [lr_max = 0.01, epochs = 100], [preactresnet], [dropRate]


    return student, teacher

if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file != None else args
    print(args)
    device = torch.device("cuda:{0}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    root = f"../models/{args.dataset}"
    model_dir = f"{root}/model_{args.model_id}"; print("Model Directory:", model_dir); args.model_dir = model_dir
    root = f"../files/{args.dataset}"
    file_dir = f"{root}/model_{args.model_id}" 
    if args.regressor_embed == 1: 
        file_dir += "_cr" 
    print("File Directory:", file_dir) ; args.file_dir = file_dir
    if(not os.path.exists(file_dir)):
        os.makedirs(file_dir)
       
    if args.dataset != "ImageNet":
        with open(f"{model_dir}/model_info.txt", "w") as f:
            json.dump(args.__dict__, f, indent=2)
    args.device = device
    print(device)
    torch.cuda.set_device(device); torch.manual_seed(args.seed)
    
    n_class = {"CIFAR10":10, "CIFAR100":100,"AFAD":26,"SVHN":10,"ImageNet":1000}
    args.num_classes = n_class[args.dataset]
    feature_extractor(args)

