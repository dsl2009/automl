import os
import glob
import shutil
import numpy as np
dr_train = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/data/laji/train'
dr_valid = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/data/laji/valid'
for i in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/data/laji/garbage_classify/train_data/*.txt'):
    x = open(i).read()
    ig_name = x.split(',')[0]
    clss = x.split(',')[1].replace(' ','')
    if np.random.randint(0,10) ==3:
        rt = os.path.join(dr_valid, clss)
        if not os.path.exists(rt):
            os.makedirs(rt)
        shutil.move(i.replace('.txt','.jpg'),rt)
    else:
        rt = os.path.join(dr_train, clss)
        if not os.path.exists(rt):
            os.makedirs(rt)
        shutil.move(i.replace('.txt', '.jpg'), rt)
