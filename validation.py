from collections import Counter
from numpy import negative, positive
import pandas as pd

from test import test
trainData = pd.read_csv('data/val.txt', sep=";", header=None)
uniqueLabelsDict = {}
uniqueLables = set(list(trainData[1]))
count = 0
for key, labels in enumerate(uniqueLables):
    uniqueLabelsDict[labels] = count
    count += 1
correctLabels = [uniqueLabelsDict[w] for w in trainData[1]]
# print(correctLabels)
positive = 0
negative = 0
predictions = test()
print(Counter(predictions))
for i in range(len(predictions)):
    if correctLabels[i] == predictions[i]:
        positive += 1
    else:
        negative += 1
print(positive, " ", negative)
print((positive/2000)*100)