#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import smdebug.pytorch as smd
import torch.optim as optim
import argparse
from torch.optim.lr_scheduler import StepLR
import os

import subprocess
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, test_loader, criterion, device, hook):
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    running_corrects = 0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds==labels.data).item()
    
    average_accuracy = running_corrects/len(test_loader.dataset)
    average_loss = test_loss/len(test_loader.dataset)
    print(f'Test set: Average loss: {average_loss}, Accuracy: {100*average_accuracy}%')


def train(model, train_loader, criterion, optimizer,epochs, device, hook):
    for epoch in range(1, epochs + 1):
        running_loss = 0
        correct=0
        hook.set_mode(smd.modes.TRAIN)
        model.train()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(loss)
            running_loss += loss
            loss.backward()
            optimizer.step()
            outputs = outputs.argmax(dim=1, keepdim=True)
            correct += outputs.eq(labels.view_as(outputs)).sum().item()
        print(f"Loss {running_loss/len(train_loader.dataset)}, \
        Accuracy {100*(correct/len(train_loader.dataset))}%")
    
    return model 

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    return model

def create_data_loaders(data, batch_size):
    print("="*100)
    print(data)
    print("="*100)
    train_path = os.path.join(data, 'train')
    test_path = os.path.join(data, 'test')
    validation_path = os.path.join(data, 'valid')
    
    covert_transforms = transforms.Compose(
        [
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        ]
    )
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=covert_transforms)    
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=covert_transforms)
    validation_dataset = torchvision.datasets.ImageFolder(root=validation_path, transform=covert_transforms)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    return train_data_loader, test_data_loader, validation_data_loader
    
    

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    model.to(args.device)
    '''
    TODO: Create your loss and optimizer
    '''
    
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(loss_criterion)
    
    print(args.data_dir)
    train_dataloader, eval_dataloader, test_dataloader = create_data_loaders(data = args.data_dir, batch_size = args.batch_size)
    
    model=train(model, train_dataloader, loss_criterion, optimizer, args.epochs, args.device, hook)
    
    test(model, test_dataloader, loss_criterion, args.device, hook)    
    
    torch.save(model, os.path.join(args.model_dir, 'dog_classification.pt'))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="PyTorch  Dog Classification")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--device", action="store_true", default="cuda", help="disables CUDA training"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--save_model", action="store_true", default=False, help="For Saving the current Model"
    )
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"], help="training data path in S3")
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"], help="location to save the model to")
    args = parser.parse_args()
    
    main(args)
