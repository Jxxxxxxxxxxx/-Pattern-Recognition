import numpy as np
import pandas as pd
import scipy.io as sio
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,KFold,RandomizedSearchCV,GridSearchCV,StratifiedKFold,cross_validate
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,accuracy_score,f1_score,confusion_matrix

def load_data(path):
    data = []
    count = 0
    for line in open(path):
        single_data = []
        sep_line = line.split()
        #print(line)
        if len(sep_line)<10:
            count+=1
            continue
        ID,_,sex,_,age,_,race,_,face = line.split()[0:9]
        sex=sex.strip(')')
        age = age.strip(')')
        race = race.strip(')')
        face = face.strip(')')

        with open('face/rawdata/'+ID, 'rb') as f:
            img = np.fromfile(f, dtype=np.ubyte).tolist()
        single_data.append(ID)
        single_data.append(sex)
        single_data.append(age)
        single_data.append(race)
        single_data.append(face)
        single_data.append(img)
        data.append(single_data)
    return data
train_path = 'face/faceDR'
test_path = 'face/faceDS'
train_data = np.array(load_data(train_path))
pd_train_data = pd.DataFrame(train_data,columns=['id','sex','age','race','face','img'])
sex_count = pd.value_counts(pd_train_data['sex'], sort=True).sort_index()
print(sex_count)
age_count = pd.value_counts(pd_train_data['age'], sort=True).sort_index()
print(age_count)
race_count = pd.value_counts(pd_train_data['race'], sort=True).sort_index()
print(race_count)
face_count = pd.value_counts(pd_train_data['face'], sort=True).sort_index()
print(face_count)
test_data = np.array(load_data(test_path))
def data_exchange(data):
    label_collection=[]
    raw_data = np.zeros(len(data),dtype=object)
    for index,image_data in enumerate(data):
        label_collection.append(image_data[1])
        raw_data[index] = np.array(image_data[-1]).reshape(128,128,-1)
    return label_collection,raw_data
train_label,train_data = data_exchange(train_data)
test_label,test_data = data_exchange(test_data)
def hog_feature(data):
    hog_train_images = np.zeros(len(data), dtype=object)
    hog_train_features = np.zeros(len(data), dtype=object)
    for i in range(len(data)):
        # 将每张图片放入hog函数得到对应的特征和图片数据
        fd, hog_image = hog(data[i], orientations=6, pixels_per_cell=(16, 16),
                            cells_per_block=(4, 4), visualize=True)
        hog_train_images[i] = hog_image
        hog_train_features[i] = fd
    return hog_train_images.tolist(),hog_train_features.tolist()
train_image_data,train_data = hog_feature(train_data)
test_image_data,test_data = hog_feature(test_data)
train_data = np.array(train_data)
test_data = np.array(test_data)
labelencoder = LabelEncoder()                   #这里将我们的文件名映射为0,1,2,3,4,5,6的数字，代表每个标签
train_label = labelencoder.fit_transform(train_label)
train_label = np.array(train_label)
test_label = labelencoder.fit_transform(test_label)
test_label = np.array(test_label)
print(train_data.shape,train_label.shape)
print(test_data.shape,test_label.shape)


lr = LogisticRegression(solver='newton-cg', penalty='l2', random_state=2020)
XGBClassifier_ = XGBClassifier(n_estimators=1000,random_state=2020,

                                )
def get_lr_classification_result(model,data,label,test_data,test_label):
    class_name = ['female', 'male']
    fold_number = 9   #做多少次交叉验证
    k_fold = StratifiedKFold(fold_number, shuffle=True, random_state=2020)  #收集第一阶段每次的测试结果
    score_value = []
    best_acc_score = float(0)
    best_model = 0
    best_predict=0
    for iteration,(train_index,test_index) in enumerate(k_fold.split(data,label)):
        model_ = model   #初始一个模型
        model_.fit(data[train_index],label[train_index])
        test_predict = model_.predict(data[test_index])
        acc_score_test = accuracy_score(label[test_index], test_predict)
        score_value.append(acc_score_test)
        if best_acc_score<acc_score_test:
            best_acc_score=acc_score_test
            best_model = model_
            best_predict = model_.predict(test_data)
        del model_
    print(best_model.coef_)
    print(best_model.intercept_)
    class_report = classification_report(test_label, best_predict, target_names=class_name, digits=4)
    confu_matirx = confusion_matrix(test_label, best_predict)
    print(max(score_value))
    print(class_report)
    print(confu_matirx)
def get_XGboost_classification_result(model,data,label,test_data,test_label):
    class_name = ['female', 'male']
    fold_number = 9   #做多少次交叉验证
    k_fold = StratifiedKFold(fold_number, shuffle=True, random_state=2020)  #收集第一阶段每次的测试结果
    score_value = []
    best_acc_score = float(0)
    best_predict=0
    for iteration,(train_index,test_index) in enumerate(k_fold.split(data,label)):
        model_ = model   #初始一个模型
        model_.fit(data[train_index],label[train_index])
        test_predict = model_.predict(data[test_index])
        acc_score_test = accuracy_score(label[test_index], test_predict)
        score_value.append(acc_score_test)
        if best_acc_score<acc_score_test:
            best_acc_score=acc_score_test
            best_predict = model_.predict(test_data)
        del model_
    class_report = classification_report(test_label, best_predict, target_names=class_name, digits=4)
    confu_matirx = confusion_matrix(test_label, best_predict)
    print(max(score_value))
    print(class_report)
    print(confu_matirx)
get_lr_classification_result(lr,train_data,train_label,test_data,test_label)
get_XGboost_classification_result(XGBClassifier_,train_data,train_label,test_data,test_label)
