import argparse
from distutils import util
import yaml
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Adversarial Training')
    ## Basics
    parser.add_argument("--config_file", help="Configuration file containing parameters", type=str)
    parser.add_argument("--dataset", help="MNIST/CIFAR10", type=str, default = "CIFAR10", choices = ["ImageNet","MNIST", "SVHN", "CIFAR10", "CIFAR100","AFAD"])
    parser.add_argument("--model_type", help="cnn/wrn-40-2/wrn-28-10/preactresnet/resnet34",
                            type=str, default = "wrn-28-10", choices = ["cnn","wrn-40-2","wrn-28-10","preactresnet", "resnet34"])
    parser.add_argument("--gpu_id", help="Id of GPU to be used", type=int, default = 0)
    parser.add_argument("--batch_size", help = "Batch Size for Train Set (Default = 100)", type = int, default = 100)
    parser.add_argument("--model_id", help = "For Saving", type = str, default = '0')
    parser.add_argument("--seed", help = "Seed", type = int, default = 0)
    parser.add_argument("--normalize", help = "Normalize training data inside the model", type = int, default = 1, choices = [0,1])
    #Only valid if restarts = 1: #0 -> Always start from Zero, #1-> Always start with 1, #2-> Start from 0/rand with prob = 1/2
    parser.add_argument("--device", help = "To be assigned later", type = str, default = 'cuda:0')
    parser.add_argument("--epochs", help = "Number of Epochs", type = int, default = 50)
    parser.add_argument("--dropRate", help = "DropRate for Teacher model", type = float, default = 0.0)
    parser.add_argument("--imagenet_architecture", help = "Imagenet Architecture", type = str, default = "wrn", choices = ["wrn","alexnet","inception"])
    
    #Threat models    
    parser.add_argument("--mode", help = "Various threat models", type = str, default = 'teacher', choices = ['zero-shot', 'prune', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher','independent','pre-act-18','random'])
    parser.add_argument("--pseudo_labels", help = "Use alternate dataset", type = int, default = 0, choices = [0,1])
    parser.add_argument("--reverse_train_test", help = "Use alternate dataset", type = int, default = 0, choices = [0,1])
    parser.add_argument("--data_path", help = "Use alternate dataset", type = str, default = None)
    parser.add_argument("--concat", help = "For Overlap Exps", type = int, default = 0, choices = [0,1])
    parser.add_argument("--concat_factor", help = "For Overlap Exps", type = float, default = 1.0)

    #LR
    parser.add_argument("--lr_mode", help = "Step wise or Cyclic", type = int, default = 1)
    parser.add_argument("--opt_type", help = "Optimizer", type = str, default = "SGD")
    parser.add_argument("--lr_max", help = "Max LR", type = float, default = 0.1)
    parser.add_argument("--lr_min", help = "Min LR", type = float, default = 0.)

    #Resume
    parser.add_argument("--resume", help = "For Resuming from checkpoint", type = int, default = 0)
    parser.add_argument("--resume_iter", help = "Epoch to resume from", type = int, default = -1)
    

    #Lp Norm Dependent
    parser.add_argument("--distance", help="Type of Adversarial Perturbation", type=str)#, choices = ["linf", "l1", "l2", "vanilla"])
    parser.add_argument("--randomize", help = "For the individual attacks", type = int, default = 0, choices = [0,1,2])
    parser.add_argument("--alpha_l_1", help = "Step Size for L1 attacks", type = float, default = 1.0)
    parser.add_argument("--alpha_l_2", help = "Step Size for L2 attacks", type = float, default = 0.01)
    parser.add_argument("--alpha_l_inf", help = "Step Size for Linf attacks", type = float, default = 0.001)
    parser.add_argument("--num_iter", help = "PGD iterations", type = int, default = 500)

    parser.add_argument("--epsilon_l_1", help = "Step Size for L1 attacks", type = float, default = 12)
    parser.add_argument("--epsilon_l_2", help = "Epsilon Radius for L2 attacks", type = float, default = 0.5)
    parser.add_argument("--epsilon_l_inf", help = "Epsilon Radius for Linf attacks", type = float, default = 8/255.)
    parser.add_argument("--restarts", help = "Random Restarts", type = int, default = 1)
    parser.add_argument("--smallest_adv", help = "Early Stop on finding adv", type = int, default = 1)
    parser.add_argument("--gap", help = "For L1 attack", type = float, default = 0.001)
    parser.add_argument("--k", help = "For L1 attack", type = int, default = 100)
    
    
    #TEST
    parser.add_argument("--path", help = "Path for test model load", type = str, default = None)
    parser.add_argument("--feature_type", help = "Feature type for generation", type = str, default = 'mingd', choices = ['pgd','topgd', 'mingd', 'rand'])

    parser.add_argument("--regressor_embed", help = "Victim Embeddings for training regressor", type = int, default = 0, choices = [0,1])


    return parser

def add_config(args):
    data = yaml.load(open(args.config_file,'r'))
    args_dict = args.__dict__
    for key, value in data.items():
        if('--'+key in sys.argv and args_dict[key] != None): ## Giving higher priority to arguments passed in cli
            continue
        if isinstance(value, list):
            args_dict[key] = []
            args_dict[key].extend(value)
        else:
            args_dict[key] = value
    return args