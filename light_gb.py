import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score,accuracy_score

iris=datasets.load_iris()

X_train,X_test,y_train,y_test = np.load('fea.npy'),np.load('fea_valid.npy'),np.load('label.npy'),np.load('label_valid.npy')
X_train = PCA(128).fit_transform(X_train)
X_test = PCA(128).fit_transform(X_test)

#X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3)

train_data=lgb.Dataset(X_train,label=y_train)
validation_data=lgb.Dataset(X_test,label=y_test)
params={
    "learning_rate":0.01,
    "lambda_l1":0.01,
    "lambda_l2":0.02,
    "max_depth":12,
    "objective":"multiclass",
    "num_class":40,
}
clf=lgb.train(params,train_data,valid_sets=[validation_data])
y_pred=clf.predict(X_test)
y_pred=[list(x).index(max(x)) for x in y_pred]
print(accuracy_score(y_test,y_pred))

y_pred=clf.predict(X_train)
y_pred=[list(x).index(max(x)) for x in y_pred]
print(accuracy_score(y_train,y_pred))
