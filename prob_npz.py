import pandas as pd
import sys
from libsvm.svmutil import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import preprocessing
import time
from multiprocessing import Pool
import numpy as np

start = time.time()
predict_alignment = sys.argv[1]
model_file = sys.argv[2]

X = pd.read_hdf(predict_alignment)
#Y = X['label']
position = X['position']

#encoder = preprocessing.LabelEncoder()
#encoder.fit(Y)
#Y = encoder.transform(Y)

def preprocessing(data):
    new_draft = []
    draft = data['draft']
    for i in draft:
        if i == 'A':
            new_draft.append(0)
        elif i == 'T':
            new_draft.append(1)
        elif i == 'C':
            new_draft.append(2)
        elif i == 'G':
            new_draft.append(3)
        elif i == '-':
            new_draft.append(4)

    data['draft'] = new_draft
    data['A'] = data['A'].astype(int) / data['coverage'].astype(int)
    data['T'] = data['T'].astype(int) / data['coverage'].astype(int)
    data['C'] = data['C'].astype(int) / data['coverage'].astype(int)
    data['G'] = data['G'].astype(int) / data['coverage'].astype(int)
    data['gap'] = data['gap'].astype(int) / data['coverage'].astype(int)
    data['label'] =1-(data['label'].astype(int)-4)
    
    data['Ins_A'] = data['Ins_A'].astype(int) / data['coverage'].astype(int)
    data['Ins_T'] = data['Ins_T'].astype(int) / data['coverage'].astype(int)
    data['Ins_C'] = data['Ins_C'].astype(int) / data['coverage'].astype(int)
    data['Ins_G'] = data['Ins_G'].astype(int) / data['coverage'].astype(int)
    
    #scaler = MinMaxScaler()
    #data['coverage'] = scaler.fit_transform(data[['coverage']])

    return data

X = preprocessing(X)
Y = X['label']
X = pd.get_dummies(X, columns=['draft', 'homopolymer'])
X = pd.DataFrame(X.drop(['label', 'position'], axis=1))

print('training shape: ', X.shape)
size = 100000
list_of_X = [X.loc[i:i+size-1,:] for i in range(0, len(X),size)]

def predict_class(input_data):
    model = svm_load_model(model_file)
    p_label, p_acc, p_val = svm_predict([], input_data.values.tolist(), model,"-b 1")
    y_score = np.array(p_val)
    #np.savetxt("prob.csv", a, delimiter=",")
    #p_prob = list(y_score[:,4])
    return y_score

pool = Pool(32)
results = pool.map(predict_class, list_of_X)
pool.close()
pool.join()
result = []
for i in results:
    result.extend(i)
print(len(result))
np.savetxt("prob.csv", result, delimiter=",")

end = time.time()
print(end-start)
