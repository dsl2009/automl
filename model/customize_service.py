import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from model.efnet import efficientnet
from model_service.pytorch_model_service import PTServingBaseService

class garbage_classify_service(PTServingBaseService):
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path
        self.signature_key = 'predict_images'

        self.input_size = 320  # the input image size of the model
        self.trans = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.input_key_1 = 'input_img'
        self.output_key_1 = 'output_score'
        self.model = efficientnet.efficientnet_b0(num_classes=40)
        self.model.load_state_dict(torch.load(self.model_path,map_location='cpu'))
        self.model.eval()
        self.label_id_name_dict = \
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
        self.lable_ix =\
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

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = self.trans(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        img = data[self.input_key_1]
        img = img.unsqueeze(0)  # the input tensor shape of resnet is [?, 224, 224, 3]
        pred_score = self.model(img)
        pred_score = torch.softmax(pred_score,1)
        pred_score = pred_score.detach().cpu().numpy()
        if pred_score is not None:
            pred_label = np.argmax(pred_score[0])
            result = {'result': self.label_id_name_dict[str(self.lable_ix[pred_label])]}
        else:
            result = {'result': 'predict score is None'}
        return result

    def _postprocess(self, data):
        return data
