import argparse
from utils import load_model, process_image, predict
import torch
import json

def arg_parser():
    parser = argparse.ArgumentParser("Predict an image class")
    
    parser.add_argument("image_path", type=str, metavar='', help="Input image path")
    parser.add_argument("checkpoint", type=str, metavar='', help="Checkpoint file path")
    parser.add_argument("-t","--top_k", type=int, default=1, metavar='', help="Top K most likely classes")
    parser.add_argument("-c", "--category_names_file", type=str, metavar='', help="File path for categories mapping to real name")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    
    if args.category_names_file:
        file = open(args.category_names_file, 'r')
        category_names = json.load(file)
    image_path = args.image_path
    checkpoint_file = args.checkpoint
    top_k = args.top_k
    gpu = args.gpu
    device = torch.device("cuda:0" if gpu else "cpu")
    device_name = "GPU" if gpu else "CPU"

    print(f"Using {device_name} for predicting")
   
    model, class_to_idx = load_model(checkpoint_file, device)
    
    idx_to_class = {value: key for key, value in class_to_idx.items()}

    top_p, top_class = predict(image_path, model, top_k, device, idx_to_class, process_image)

    print("Prediction result")
    for p, c in zip(top_p[0].tolist(), top_class):
        if args.category_names_file:
            c = category_names[c].capitalize() 
        print("class {} with probability {:.3f}".format(c, p))