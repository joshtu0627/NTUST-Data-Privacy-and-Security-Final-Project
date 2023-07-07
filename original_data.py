# 這個程式的目標是用原始資料來訓練模型

import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import metrics

# 用pandas以csv檔案格式讀檔
df = pd.read_csv("colorado_original_INCWAGE_dropped.csv")
df=df[['OCCSCORE','EMPSTATD', 'WKSWORK2', 'FARM', 'SEX', 'AGE','INCWAGE']]

# 將<=1000的薪水設為1 >1000的薪水設為2
def salary_label(salary):
    if int(salary) > 1000:
        return 1
    else:
        return 0
df['INCWAGE'] = df['INCWAGE'].apply(salary_label)

# 指定亂數種子
np.random.seed(7)

# 載入資料集
print(df.head(5))
categorical = set((
    #由於沒有非數字型別這邊沒放東西
))

accuracy=[]
precision=[]
recall=[]

plt.figure(figsize=[14,5])
test_time=10
for i in range(test_time):
    # 把dataframe中的資料給dataset 再將dataset suffle
    dataset=df.values
    np.random.shuffle(dataset)

    # 分割資料
    X=dataset[:,0:6].astype(float)
    Y=to_categorical(dataset[:,6])

    # 特徵標準化 避免不同column中數值的大小不一樣導致的權重失衡
    X-=X.mean(axis=0)
    X/=X.std(axis=0)

    # 分割成訓練和測試資料集
    X_train,Y_train=X[:50000],Y[:50000]
    X_test,Y_test=X[50000:55000],Y[50000:55000]

    # 建立模型
    model=Sequential()
    model.add(Dense(20,input_shape=(6,),activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.summary()

    # 編譯模型
    model.compile(loss='categorical_crossentropy',optimizer='adam',
                    metrics=['acc'])

    model.fit(X_train,Y_train,epochs=10,batch_size=64)

    # 評估模型
    result=model.evaluate(X_test,Y_test)

    # 假設<=1000為真
    # <=1000 是 [1 0]
    predict_raw=model.predict(X_test)
    predict=predict_raw.round(0)
    tp=fp=fn=tn=0
    for j in range(len(predict)):
        if Y_test[j][0]==1: # positive
            if predict[j][0]==1: # true positive
                tp+=1
            else: # false negative
                fn+=1
        else:
            if predict[j][0]==1: # false positive
                fp+=1
            else: # true negative
                tn+=1
    print(f'tp:{tp}, fp:{fp}')
    print(f'fn:{fn}, tn:{tn}')
    print('Accuracy: ',(tp+tn)/(tp+fp+fn+tn))
    print('Precision: ', tp/(tp+fp))
    print('Recall: ', tp/(tp+fn))
    accuracy.append((tp+tn)/(tp+fp+fn+tn))
    precision.append(tp/(tp+fp))
    recall.append(tp/(tp+fn))

    # 作auc圖的資料
    p=[]
    for j in range(len(predict_raw)):
        p.append([predict_raw[j][0],Y_test[j][0]])

    def sort_fun(l):
        return l[0]
    p=sorted(p,key=sort_fun,reverse=True)
    p=np.array(p).round(0)

    y_unit=1/(tp+fn)
    x_unit=1/(fp+tn)

    x=[]
    y=[]
    current_x=0
    current_y=0
    for row in p:
        if row[1]==1:
            current_y+=y_unit
        else:
            current_x+=x_unit
        x.append(current_x)
        y.append(current_y)

    x=np.array(x)
    y=np.array(y)

    # 繪圖
    plt.subplot(2,5,i+1) # 有兩列兩行 然後現在在操作的是第一個
    plt.title('AUC curve '+str(i))
    plt.plot(x, y, color = 'orange', label = 'AUC')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
print(f'Average accuracy:{sum(accuracy)/test_time}')
print(f'Average misclassification error:{1-(sum(accuracy)/test_time)}')
print(f'Average precision:{sum(precision)/test_time}')
print(f'Average recall:{sum(recall)/test_time}')

plt.subplots_adjust(
    left=0.1,
    bottom=0.1, 
    right=0.9, 
    top=0.9, 
    wspace=0.5, 
    hspace=0.5
)
plt.show()

# 儲存模型
model.save('model/original_data.h5')