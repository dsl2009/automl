from model.efnet import efficientnet
from PIL import ImageFile,Image
from torchvision import transforms
import torch
import glob
import numpy as np
trans = transforms.Compose([
        transforms.Resize(260),
        transforms.CenterCrop(260),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

model = efficientnet.efficientnet_b2(num_classes=1000)
model.load_state_dict(torch.load('/home/dsl/all_check/efficientnet_b2-cf78dc4d.pth'))
model.eval()
model.cuda()
feature = []
label = []
for pth in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/data/laji/valid/*/*.*'):
    ig = Image.open(pth)
    ig = trans(ig)
    ig = ig.unsqueeze(0)
    x, out = model(ig.cuda())
    feature.append(x.detach().cpu().numpy())
    label.append(int(pth.split('/')[-2]))
feature = np.concatenate(feature,axis=0)
np.save('fea_valid',feature)
np.save('label_valid',np.asarray(label))