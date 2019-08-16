import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from timm.models import gen_efficientnet
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
import json
from torch.optim import lr_scheduler
from PIL import ImageFile
from losses import hing
import glob
import os


ImageFile.LOAD_TRUNCATED_IMAGES = True
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(260,scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.2),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(280),
        transforms.CenterCrop(260),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



def run(trainr, test_dr,name,svdr,  checkdrs):
    batch_size = 16
    imagenet_data = ImageFolder(trainr,
                                transform=data_transforms['train'])
    test_data = ImageFolder(test_dr, transform=data_transforms['val'])

    data_loader = DataLoader(imagenet_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    cls_num = len(imagenet_data.class_to_idx)
    data_loader = DataLoader(imagenet_data, batch_size=batch_size, shuffle=True)
    model = gen_efficientnet.efficientnet_b2(drop_rate=0.2,drop_connect_rate=0.2)
    model.load_state_dict(torch.load(checkdrs), strict=False)
    model.classifier = nn.Linear(1408, cls_num)
    model.cuda()

    state = {'learning_rate': 0.01, 'momentum': 0.9, 'decay': 0.0005}
    optimizer = torch.optim.SGD(model.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    state['label_ix'] = imagenet_data.class_to_idx
    state['cls_name'] = name

    state['best_accuracy'] = 0
    sch = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    ll = len(data_loader.dataset)

    test_acc = []
    test_ls = []

    def train():
        model.train()
        loss_avg = 0.0
        ip1_loader = []
        idx_loader = []
        correct = 0
        acc = []
        ls = []
        nm = 0
        for (data, target) in data_loader:
            inputs, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
            output = model(inputs)
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())
            optimizer.zero_grad()
            #loss = mixup_criterion(focal_loss, output, targets_a, targets_b, lam)
            #loss = F.cross_entropy(output, target)
            loss = hing.hing_loss(output,target)
            loss.backward()
            optimizer.step()
            loss_avg = loss_avg * 0.2 + float(loss) * 0.8
            ls.append(loss_avg)
            acc.append(correct/ll)
            nm+=batch_size
            print(state['epoch'],nm,correct, ll, loss_avg)


        state['train_accuracy'] = correct / len(data_loader.dataset)
        state['train_loss'] = loss_avg
    def test():
        with torch.no_grad():
            model.eval()
            loss_avg = 0.0
            correct = 0
            print(test_data_loader)
            for batch_idx, (data, target) in enumerate(test_data_loader):
                data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
                print(target)
                print(target.size())
                output = model(data)
                print(output.size())
                loss = F.cross_entropy(output, target)

                pred = output.data.max(1)[1]

                correct += float(pred.eq(target.data).sum())
                loss_avg += float(loss)
                state['test_loss'] = loss_avg / len(test_data_loader)
                state['test_accuracy'] = correct / len(test_data_loader.dataset)
                test_acc.append(state['test_accuracy'])
                test_ls.append(state['test_loss'])
            print(state['test_accuracy'])


    best_accuracy = 0.0
    for epoch in range(60):
        state['epoch'] = epoch
        train()
        test()
        sch.step(state['train_accuracy'])
        best_accuracy = state['test_accuracy']
        if best_accuracy > state['best_accuracy']:
            state['best_accuracy'] = best_accuracy
            torch.save(model.state_dict(), os.path.join(svdr, name + '.pth'))
            with open(os.path.join(svdr, name + '.json'), 'w') as f:
                f.write(json.dumps(state))
                f.flush()
        print(state)
        print("Best accuracy: %f" % state['best_accuracy'])



if __name__ == '__main__':
    train_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/data/laji/train'
    test_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/data/laji/valid'
    sv = ''
    run(train_dr,test_dr, name='xag',svdr=sv,checkdrs='/home/dsl/all_check/efficientnet_b2-cf78dc4d.pth')