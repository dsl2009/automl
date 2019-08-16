from model.efnet import efficientnet
from PIL import ImageFile,Image
from torchvision import transforms
import torch
import glob
import numpy as np
trans = transforms.Compose([
        transforms.Resize(280),
        transforms.CenterCrop(260),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
label_id_name_dict = \
    {
        "0": "其他垃圾/一次性快餐盒",
        "1": "其他垃圾/污损塑料",
        "2": "其他垃圾/烟蒂",
        "3": "其他垃圾/牙签",
        "4": "其他垃圾/破碎花盆及碟碗",
        "5": "其他垃圾/竹筷",
        "6": "厨余垃圾/剩饭剩菜",
        "7": "厨余垃圾/大骨头",
        "8": "厨余垃圾/水果果皮",
        "9": "厨余垃圾/水果果肉",
        "10": "厨余垃圾/茶叶渣",
        "11": "厨余垃圾/菜叶菜根",
        "12": "厨余垃圾/蛋壳",
        "13": "厨余垃圾/鱼骨",
        "14": "可回收物/充电宝",
        "15": "可回收物/包",
        "16": "可回收物/化妆品瓶",
        "17": "可回收物/塑料玩具",
        "18": "可回收物/塑料碗盆",
        "19": "可回收物/塑料衣架",
        "20": "可回收物/快递纸袋",
        "21": "可回收物/插头电线",
        "22": "可回收物/旧衣服",
        "23": "可回收物/易拉罐",
        "24": "可回收物/枕头",
        "25": "可回收物/毛绒玩具",
        "26": "可回收物/洗发水瓶",
        "27": "可回收物/玻璃杯",
        "28": "可回收物/皮鞋",
        "29": "可回收物/砧板",
        "30": "可回收物/纸板箱",
        "31": "可回收物/调料瓶",
        "32": "可回收物/酒瓶",
        "33": "可回收物/金属食品罐",
        "34": "可回收物/锅",
        "35": "可回收物/食用油桶",
        "36": "可回收物/饮料瓶",
        "37": "有害垃圾/干电池",
        "38": "有害垃圾/软膏",
        "39": "有害垃圾/过期药物"
    }
lable_ix =\
            {0 : 0 ,
            1 : 1 ,
            2 : 10 ,
            3 : 11 ,
            4 : 12 ,
            5 : 13 ,
            6 : 14 ,
            7 : 15 ,
            8 : 16 ,
            9 : 17 ,
            10 : 18 ,
            11 : 19 ,
            12 : 2 ,
            13 : 20 ,
            14 : 21 ,
            15 : 22 ,
            16 : 23 ,
            17 : 24 ,
            18 : 25 ,
            19 : 26 ,
            20 : 27 ,
            21 : 28 ,
            22 : 29 ,
            23 : 3 ,
            24 : 30 ,
            25 : 31 ,
            26 : 32 ,
            27 : 33 ,
            28 : 34 ,
            29 : 35 ,
            30 : 36 ,
            31 : 37 ,
            32 : 38 ,
            33 : 39 ,
            34 : 4 ,
            35 : 5 ,
            36 : 6 ,
            37 : 7 ,
            38 : 8 ,
            39 : 9}
model = efficientnet.efficientnet_b2(num_classes=40)
model.load_state_dict(torch.load('model/xag.pth'))
model.eval()
model.cuda()
for pth in glob.glob('D:/deep_learn_data/garbage_classify/garbage_classify/train/ 2/*.*'):
    print(pth)
    ig = Image.open(pth)
    ig = trans(ig)
    ig = ig.unsqueeze(0)
    out = model(ig.cuda())
    pred_score = torch.softmax(out, 1)
    pred_score = pred_score.detach().cpu().numpy()
    print(pred_score.shape)
    if pred_score is not None:
        pred_label = np.argmax(pred_score[0])
        print(pred_label)
        ix = label_id_name_dict[str(lable_ix[pred_label])]
        print(ix)