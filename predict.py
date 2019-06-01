import argparse
from utils import 
import torch

def arg_parser():
    parser = argparse.ArgumentParser("Predict an image class")
    
    parser.add_argument("image_path", type=str, metavar='', help="Input image path")
    parser.add_argument("checkpoint", type=str, metavar='', help="Checkpoint file path")
    parser.add_argument("-t","--top_k", type=int, default=1, metavar='', help="Top K most likely classes")
    parser.add_argument("-c", "--category_names_file ", type=str, metavar='', help="File path for categories mapping to real name")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    args = parser.parse_args()
    return args 


if __name__ == "__main__":
    args = arg_parser()
    image_path = args.image_path
    check_point = args.check_point
    top_k = args.top_k
    category_names = args.category_names_file
    gpu = args.gpu
    device = torch.device("cuda:0" if gpu else "cpu")
    device_name = "GPU" if gpu else "CPU"

    print(f"Using {device_name} for predicting")
   