from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle



thres = [1,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0]

scores = [0.94,0.93,0.9,0.91,0.77,0.94,0.93,0.95,0.73,0.93,0.69,0.7,0.91,0.7,0.94,0.95,0.93,0.95,0.89,0.87,0.68,0.51,0.88,0.93,0.94,0.96,0.91,0.94,0.93,0.85,0.96,0.94,0.91,0.63,0.92,0.84,0.93,0.89,0.87,0.93,0.95,0.73,0.85,0.87,0.88,0.93,0.8,0.91,0.84,0.5,0.94,0.82,0.74,0.91,0.9,0.84,0.84,0.89,0.68,0.89,0.84,0.93,0.88,0.79,0.71,0.82,0.85,0.9,0.67,0.92]
y_true = [1,1,1,0,1,0,1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,1,1,0,1,0,0,1,0,0,1,0]

TP = np.zeros(len(thres), dtype = int)
FN = np.zeros(len(thres), dtype = int)
TN = np.zeros(len(thres), dtype = int)
FP = np.zeros(len(thres), dtype = int)
a = 0
b = 0

n0 = 0
n1 = 1

for x in thres:
    for i in scores:
        if (y_true[b]==n1):
            if (i >= x):
                TP[a] += 1
            else: FN[a] += 1
        if (y_true[b]!=n1):
            if (1-i >= x):
                FP[a] += 1
            else: TN[a] += 1
        b += 1
    # print('TP', x, ' ', TP[a])
    # print('FP', x,' ', FP[a])
    # print('TN', x, ' ', TN[a])
    # print('FN', x, ' ', FN[a])
    a += 1
    b = 0
print('TP =',TP)
print('FP =',FP)
print('FN =',FN)
print('TN =',TN)



# print('True labels:')
# print(y_true)
# print('\nScores:')
# print(scores)

# fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label = 1)
# print('\nThreshold:')
# print(thresholds)
# print('True Positive Rate:')
# print(tpr)
# print('False Positive Rate:')
# print(fpr)

# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()






