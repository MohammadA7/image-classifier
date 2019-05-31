
import argparse
from utils import load_transform_data, build_model, train_model, test_model, save_model


def arg_parser():
    parser = argparse.ArgumentParser("Train an image classifier")
    
    parser.add_argument("data_dir", type=str, metavar='', help="Data directory should contain /test, /train and /valid")
    parser.add_argument("--save_dir", type=str, metavar='', help="Directory with will contain model checkpoint")
    parser.add_argument("-l","--learning_rate", type=float, metavar='', help="Learning rate")
    parser.add_argument("-H", "--hidden_units", type=int, metavar='', help="Hidden units")
    parser.add_argument("-e", "--epochs", type=int, metavar='', help="Number of epochs")
    parser.add_argument("-a","--arch", type=str, metavar='', required=True, help="Architecture of the pretrained model")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    args = parser.parse_args()
    
    return args 


if __name__ == "__main__":
    args = arg_parser()
    data_dir = args.data_dir
    gpu = args.gpu
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    output_classes = args.output_classes
    device = torch.device("cuda:0" if gpu else "cpu")


    train_data_loader, valid_data_loader, test_data_loader, class_to_idx = load_transform_data(data_dir)
    
    model = build_model(args.arch, learning_rate, hidden_units, device, len(class_to_idx))
    
