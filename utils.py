from torch import nn, optim, torch
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import pandas as pd



def load_transform_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop (224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(), 
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    valid_test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_image_datasets,batch_size=64,shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_image_datasets,batch_size=32)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_image_datasets,batch_size=24)

    return train_data_loader, valid_data_loader, test_data_loader, train_image_datasets.class_to_idx

def build_model(arch, learning_rate, hidden_units, device, output_classes):
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024 
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    else:
        raise Exception(f'Sorry we only support densenet121 and vgg16 architectures')

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_units, output_classes),
        nn.LogSoftmax(dim=1))
        
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    model.to(device)
    return model, optimizer, criterion

def train_model(device, train_data_loader, valid_data_loader, model, epochs, optimizer, criterion):
    for i in range(epochs):
        model.train()
        training_loss = 0

        for images, labels in train_data_loader:
            images = images.to(device)
            labels = labels.to(device)
        
            optimizer.zero_grad()    
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        else:
            with torch.no_grad():
                model.eval()
                validation_loss = 0
                accuracy = 0
                for images, labels in valid_data_loader:
                    images, labels = images.to(device), labels.to(device)
        
                    log_ps = model.forward(images)
                    loss = criterion(log_ps, labels)
                    validation_loss += loss.item()
                    
                    ps = torch.exp(log_ps).data
                    
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            
            print("Epoch: {}/{}.. ".format(i+1, epochs),
                "Training Loss: {:.3f}.. ".format(training_loss/len(train_data_loader)),
                "Validation Loss: {:.3f}.. ".format(validation_loss/len(valid_data_loader)),
                "Accuracy: {:.3f}%".format((accuracy/len(valid_data_loader))*100))        


def test_model(device, test_data_loader, model):
    accuracy = 0
    model.eval()
    test_loss = 0
    for images, labels in test_data_loaders:
        images, labels = images.to(device), labels.to(device)
        log_ps = model(images)
        preds = torch.exp(log_ps)
                
        top_p, top_class = preds.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
    print("Model accuracy on test data: {:.3f}".format((accuracy/len(test_data_loader))*100) )

def save_model(model, optimizer, class_to_idx, epochs, dir=''):
    checkpoint = {
              'state_dict': model.state_dict(),
              'class_to_idx': class_to_idx
              }
    torch.save(checkpoint, dir + 'checkpoint.pth')
    return dir + 'checkpoint.pth'

# def load_model(checkpoint_file, arch, learning_rate, hidden_units, device, output_classes):
#     checkpoint = torch.load(checkpoint_file)
    
#     if arch == 'densenet121':
#         model = models.densenet121(pretrained=True)
#         input_size = 1024 
#     elif arch == 'vgg16':
#         model = models.vgg16(pretrained=True)
#         input_size = 25088
#     else:
#         raise Exception(f'Sorry we only support densenet121 and vgg16 architectures')

#     for param in model.parameters():
#         param.requires_grad = False

#     model.classifier = nn.Sequential(
#         nn.Linear(input_size, hidden_units),
#         nn.ReLU(),
#         nn.Dropout(p=0.2),
#         nn.Linear(hidden_units, output_classes),
#         nn.LogSoftmax(dim=1))
    
#     model.load_state_dict(checkpoint['state_dict'])
    
#     optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     model.to(device) 

#     return model, optimizer
