import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from base_model import native_senet
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
import os
from timm.models import gen_efficientnet
ImageFile.LOAD_TRUNCATED_IMAGES = True
from optm_third import over9000
from base_model import stn
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224,scale=(0.7,1.5)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.2),
        transforms.RandomGrayscale(0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
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
    model = gen_efficientnet.efficientnet_b0(num_classes=1000,drop_rate=0.5,drop_connect_rate=0.5)
    model.load_state_dict(torch.load(checkdrs), strict=False)


    model.classifier = nn.Linear(1280, cls_num)
    model.cuda()

    stn_net = stn.Net()
    stn_net.cuda()

    state = {'learning_rate': 0.01, 'momentum': 0.9, 'decay': 0.01}
    #optimizer = torch.optim.SGD(model.parameters(), state['learning_rate'],momentum=0.9,
                                #weight_decay=state['decay'], nesterov=True)
    parms = list(model.parameters())+list(stn_net.parameters())
    optimizer = over9000.Over9000(parms,weight_decay=0.01)

    print(cls_num)
    state['label_ix'] = imagenet_data.class_to_idx
    state['cls_name'] = name

    state['best_accuracy'] = 0
    sch = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    ll = len(data_loader.dataset)

    test_acc = []
    test_ls = []

    def train():
        model.train()
        #model.bn1.eval()
        #model.blocks.eval()
        #model.conv_stem.eval()
        for layers in model.modules():
            if isinstance(layers, nn.BatchNorm2d):
                layers.eval()
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
            loss = F.cross_entropy(output, target)

            #loss = hing.hing_loss(output,target)
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
            for batch_idx, (data, target) in enumerate(test_data_loader):
                data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
                output = model(data)
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
    for epoch in range(30):
        state['epoch'] = epoch
        train()
        test()
        sch.step(state['train_accuracy'])
        best_accuracy = state['test_accuracy']
        if best_accuracy > state['best_accuracy']:
            state['best_accuracy'] = best_accuracy
            torch.save(model.state_dict(), os.path.join(svdr, name + '.pth'))
            torch.save(stn_net.state_dict(), os.path.join(svdr, name + 'stn.pth'))
            with open(os.path.join(svdr, name + '.json'), 'w') as f:
                f.write(json.dumps(state))
                f.flush()
        print(state)
        print("Best accuracy: %f" % state['best_accuracy'])



if __name__ == '__main__':
    import glob
    import shutil
    f1 = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/flower/train/daisy'
    f2 = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/flower/train/roses'
    root = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/flower/test'
    if not os.path.exists(root):
        os.makedirs(root)

    for i in range(1,6,1):
        for x in glob.glob(os.path.join(root,'*','*.jpg')):
            os.remove(x)
        daisy_dr = os.path.join(root,'daisy')
        roses_dr = os.path.join(root, 'roses')
        if not os.path.exists(daisy_dr):
            os.makedirs(daisy_dr)
        if not os.path.exists(roses_dr):
            os.makedirs(roses_dr)

        for x in glob.glob(os.path.join(f1,'*.jpg')):
            if np.random.randint(0,6)==1:
                shutil.copy(x,daisy_dr)
        for x in glob.glob(os.path.join(f2,'*.jpg')):
            if np.random.randint(0,6)<i:
                shutil.copy(x,roses_dr)

        train_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/flower/test'
        test_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/flower/valid'
        sv = ''
        run(train_dr,test_dr, name='xag'+str(i),svdr=sv,checkdrs='/home/dsl/all_check/efficientnet_b0-d6904d92.pth')