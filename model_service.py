import json
import torch

import torch.nn.functional as F
import numpy as np

from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image


architectures_in_features = {
    'alexnet': 9216,
    'densenet161': 2208,
    'vgg16': 25088
}
    
data_transforms = {
    'train_transforms' : transforms.Compose([transforms.RandomRotation(30),
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])]),
    'test_transforms' : transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])]),
    'validation_transforms' : transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], 
                                                                       [0.229, 0.224, 0.225])])
}


def get_dirs(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    return train_dir, valid_dir, test_dir


def get_image_datasets(data_dir):
    return {
        'train_data' : datasets.ImageFolder(data_dir + '/train', transform=data_transforms['train_transforms']),
        'test_data' : datasets.ImageFolder(data_dir + '/test', transform=data_transforms['test_transforms']),
        'valid_data' : datasets.ImageFolder(data_dir + '/valid', transform=data_transforms['validation_transforms'])
    }


def get_dataloaders(image_datasets):
    return {
        'trainloader' : torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
        'testloader' : torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=64),
        'validloader' : torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64)
    }


def model_init(arch, hidden_units):
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print('{} architecture is not available. Please, choose between alexnet, densenet161 or vgg16.'.format(arch))
        return None

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(architectures_in_features[arch], hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    
    return model


def model_train(arch, epochs, env, hidden_units, learning_rate, dataloaders):
    model = model_init(arch, hidden_units)
    
    if not model:
        return None
    
    model.to(env)
    
    steps = 0
    running_loss = 0
    print_every = 20

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for inputs, labels in dataloaders['trainloader']:
            steps += 1

            inputs, labels = inputs.to(env), labels.to(env)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Validation step
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0

                model.eval()

                with torch.no_grad():
                    for images, labels in dataloaders['validloader']:
                        images, labels = images.to(env), labels.to(env)

                        output = model.forward(images)

                        test_loss += criterion(output, labels).item()

                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Training loss: {running_loss/print_every:.3f} - "
                      f"Validation loss: {test_loss/len(dataloaders['validloader']):.3f} - "
                      f"Validation accuracy: {accuracy/len(dataloaders['validloader']):.3f}")

                running_loss = 0

                model.train()
    
    return model


def load_checkpoint(path):
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    
    model = model_init(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image_path):
    with Image.open(image_path) as img:  
        img = data_transforms['test_transforms'](img)
        
    return img


def get_cat_to_name(json_file):
    if json_file:
        with open(json_file, 'r') as f:
            return json.load(f)
        
    return None


def predict(image_path, model, topk, category_names, env): 
    model.to(env)
    model.eval()
    
    img = torch.unsqueeze(process_image(image_path).float().to(env), dim=0)
    
    with torch.no_grad():
        output = model.forward(img)
    
    preds = F.softmax(output.data, dim=1).topk(topk)
    
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    
    probs = ["{0:6.2f}".format(x) for x in preds[0][0].cpu().data.numpy()*100]
    classes = preds[1][0].cpu().data.numpy()
    topk_labels = [idx_to_class[i] for i in classes]
    
    cat_to_name = get_cat_to_name(category_names)
    
    if cat_to_name:
        labels = list()
        
        for label in topk_labels:
            labels.append(cat_to_name[label])
    else:
        labels = topk_labels
    
    return probs, labels
