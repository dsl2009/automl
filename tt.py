import json
import shutil
import glob
import os
import numpy as np

bt = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/data/gd/guangdong1_round1_train1_20190809/valid'

dts = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/data/gd/guangdong1_round1_train1_20190809/train/*/*.jpg')
for k in dts:
    if np.random.randint(0,10)==2:
        sf = k.split('/')[-2]
        am = os.path.join(bt,sf)
        if not os.path.exists(am):
            os.makedirs(am)
        shutil.move(k,am)

