
import argparse
from utils import load_transform_data, build_model, train_model, test_model, save_model
import torch

def arg_parser():
    parser = argparse.ArgumentParser("Train an image classifier")
    
    parser.add_argument("data_dir", type=str, metavar='', help="Data directory should contain /test, /train and /valid")
    parser.add_argument("--save_dir", type=str, default='', metavar='', help="Directory with will contain model checkpoint")
    parser.add_argument("-l","--learning_rate", type=float, default=0.002, metavar='', help="Learning rate")
    parser.add_argument("-H", "--hidden_units", type=int, default=512 , metavar='', help="Hidden units")
    parser.add_argument("-e", "--epochs", type=int, default=15, metavar='', help="Number of epochs")
    parser.add_argument("-a","--arch", type=str, default='densenet121' , metavar='', help="Architecture of the pretrained model")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    args = parser.parse_args()
    
    return args 


if __name__ == "__main__":
    args = arg_parser()
    data_dir = args.data_dir
    check_point_dir = args.save_dir
    gpu = args.gpu
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    device = torch.device("cuda:0" if gpu else "cpu")
    device_name = "GPU" if gpu else "CPU"
    epochs = args.epochs

    print(f"Using {device_name} for training")
    train_data_loader, valid_data_loader, test_data_loader, class_to_idx = load_transform_data(data_dir)
    output_classes = len(class_to_idx)


    model, optimizer, criterion = build_model(args.arch, learning_rate, hidden_units, device, output_classes)

    train_model(device, train_data_loader, valid_data_loader, model, epochs, optimizer, criterion)

    test_model(device, test_data_loader, model)

    save_model(model, optimizer, class_to_idx, epochs, check_point_dir)