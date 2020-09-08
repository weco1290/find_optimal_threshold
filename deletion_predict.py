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
    #data['label'] =1-(data['label'].astype(int)-4)
    
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
    p_prob = list(y_score[:,3])
    return p_prob

pool = Pool(32)
results = pool.map(predict_class, list_of_X)

pool.close()
pool.join()
result = []
for i in results:
    result.extend(i)
#print(len(result))
#pricision recall curve
#Compute the average precision score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
y_test = label_binarize(Y.tolist(), classes=[5,0,6,4,1,2,3])
average_precision = average_precision_score(y_test[:,3], result)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from numpy import argmax
precision, recall, thresholds = precision_recall_curve(y_test[:,3], result)
# convert to f score
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
'''
plt.plot(recall, precision, marker='.', label='libsvm')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show title
plt.title('deletion label Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
# show the plot
#pyplot.show()
plt.savefig('label4_prcurve.png')
'''
'''
ACC, MSE, SCC = evaluations(Y.tolist(), result)
print(ACC)
print(confusion_matrix(Y.tolist(),result))

#Metrics
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(Y.tolist(),result))
print("G-mean: ",geometric_mean_score(Y.tolist(),result, average=None))


sub = X
sub['position'] = position
sub['label'] = Y
sub['predict'] = result
sub.to_hdf('deletion_result.h5', key='sub', mode='w')
# sub.to_csv('result.csv', index=False)

debug = sub[sub['predict'] != sub['label']]
debug.to_csv('deletion_debug.csv', index=False)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm =confusion_matrix(Y.tolist(),result)
plt.figure(figsize=(15,15))
ax= plt.subplot()
sns.heatmap(cm, annot=True, linewidths=.5, square = True, cmap = 'Blues_r', fmt='g', annot_kws={"size":15}, ax=ax);
ax.set_xticklabels( ['deletion','no deletion'])
ax.set_yticklabels( ['deletion','no deletion'])

plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(ACC)
plt.title(all_sample_title, size = 15);
#ax.xaxis.set_ticklabels(targets); ax.yaxis.set_ticklabels(targets);
plt.savefig('deletion_confusion.png')
'''
end = time.time()
print(end-start)
