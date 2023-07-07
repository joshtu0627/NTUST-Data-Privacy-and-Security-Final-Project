# 這個程式的目標是用神經網路來分析經過l-diversity後的原始資料

import pandas as pd
import numpy as np

# 有些column裡面的資料不是數字型的 需要特別處理 這邊把那些column提出來
categorical = set((
    #由於沒有非數字型別這邊沒放東西
))

# 檢測該partition是否滿足k匿名
def is_k_anonymous(partition,k):
    if len(partition)<k:
        return False
    return True

def is_l_diversity(df, partition, l):
    if len(set(df["INCWAGE"][partition]))<l:
        return False
    return True    


# 將一個partition 以傳入的column為基準 從中間分成兩半
# 如果該column是非數值型的資料 先取得不重複的值的數量 再從該數量分成左半段跟右半段
# 如果是數值型 則以中位數分成左半段跟右半段
def split(df,partition,column): # df:資料集, partition:該組的index陣列, column:做split用的資料欄
    partition_data=df[column][partition]

    # 如果是以catagorical的column做split
    if column in categorical:
        values=partition_data.unique()
        l=values[:len(values)//2]
        r=values[len(values)//2:]
        l_data=partition_data.isin(l)
        r_data=partition_data.isin(r)
        return partition_data.index[l_data],partition_data.index[r_data]

    # 如果是以numerical的column做split
    else:
        median = partition_data.median()
        l=partition_data.index[partition_data <= median]
        r=partition_data.index[partition_data > median]
        return l,r

# 對dataset做partition
# 先把整個dataset當成一個partition丟進partitions中
# while迴圈中取partitions中的第一個做split split完丟回partitions裡面
# 如果一個partition沒辦法再split 丟進finished_partitions
# 最後把partitions都split完 將finished_partitions回傳
def partition(df, feature_columns,sensitive_column,k, l_d):
    finished_partitions=[]
    partitions=[df.index]
    i=0
    while partitions:
        partition=partitions.pop()
        for column in feature_columns:
            l,r=split(df,partition,column)
            if is_k_anonymous(l,k) and is_k_anonymous(r,k) and is_l_diversity(df, l, l_d) and is_l_diversity(df, r, l_d):
                partitions.extend((l,r))
                break
        else:
            finished_partitions.append(partition)
            i+=1
            if i%100==1:
                print("Finished splitting {} partitions...".format(i))
    return finished_partitions

# 如果是categorical 則將該partition中所有的不重複字串連起來 回傳
# 假設partition中有個name欄位 各個值是 'bob', 'sandy', 'bruh'
# 則該partition中的name欄位會變成'bob, sandy, bruh', 'bob, sandy, bruh', 'bob, sandy, bruh'
def get_anonymize_categorical_column_value(series):
    return ','.join(set(series))

# 如果是numerical 則將該partition中所有值取平均 回傳
def get_anonymize_numerical_column_value(series):
    return series.mean()

# 將每個partition的各個值以上面定義的兩種方式給值
def build_k_anonymity_data(df,partitions, feature_columns, sensitive_column):
    rows=[]
    i=1
    for partition in partitions:
        if i % 100 == 1:
            print("Finished {} partitions...".format(i))
        for column in feature_columns:
            if column in categorical:
                df.loc[partition,column]=get_anonymize_categorical_column_value(df.loc[partition,column])
            else:
                df.loc[partition,column]=get_anonymize_numerical_column_value(df.loc[partition,column])
        i+=1
    return df

# 建立k-anonymity的資料集
def build_data(df,k, l_d):
    feature_columns =['OCCSCORE','EMPSTATD', 'WKSWORK2', 'FARM', 'SEX', 'AGE']
    sensitive_column = ['INCWAGE']
    finished_partitions = partition(df, feature_columns, sensitive_column,k, l_d)
    print('total partitions:', len(finished_partitions))
    dfn=build_k_anonymity_data(df, finished_partitions, feature_columns, sensitive_column)
    return dfn[feature_columns+sensitive_column]

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

# 這邊可以設定要處理哪個區段的資料
# df=df.iloc[0:1000]

# 指定k是多少
k=800
l_d=2

df=build_data(df,k,l_d)
# print(df.head(5))



############################################################# 下面是深度學習部分



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import metrics

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

    # 假設<=50為真
    # <=50 是 [1 0]
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
model.save('model/l-Diversity_data.h5')