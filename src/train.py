from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer
from wordcloud import WordCloud
import re
from scipy.sparse import csr_matrix
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def transformer(data):
    vectorizer = CountVectorizer()
    vectorized = vectorizer.fit_transform(data)
    transformer = TfidfTransformer()
    transformed = transformer.fit_transform(vectorized)
    return vectorizer, transformed

def trainModel():
    print("Training")
    from nltk.corpus import stopwords
    verbs = ['wasnt','didnt','doesnt','dont', 'wont', 'wouldnt', 'also']
    fileData = pd.read_csv('data/train.txt', sep=";", header=None)
    trainData = fileData
    stopwords = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()

    # Word Cloud Start 
    # text = " ".join(review for review in trainData[0])
    # wordcloud = WordCloud(stopwords=stopwords).generate(text)
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.savefig('wordcloud11.png')
    # plt.show()
    # Word Cloud End

    uniqueLabelsDict = {}
    uniqueLables = set(list(trainData[1]))
    count = 0
    for key, labels in enumerate(uniqueLables):
        uniqueLabelsDict[labels] = count
        count += 1
    markedLabels = [uniqueLabelsDict[w] for w in trainData[1]]
    sentimentList = []
    for label in markedLabels:
        if (label < 3):
            sentimentList.append(0)
        else:
            sentimentList.append(1)
    trainData[0] = trainData[0].map(lambda x: re.sub('[,\.!?]', '', x))
    trainData[0] = trainData[0].map(lambda x: x.lower())
    tokenizedWord = [nltk.word_tokenize(line) for line in trainData[0]]
    cleanedData = []
    finalData = []
    for line in tokenizedWord:
        cleanedData.append([wordnet_lemmatizer.lemmatize(word) for word in line if word not in stopwords and word not in verbs and len(word) > 3])
    for line in cleanedData:  
        finalData.append(" ".join(w for w in line))
    vectorizer, trainingData = transformer(finalData)
    trainingData = sp.sparse.hstack(( trainingData, csr_matrix(markedLabels).T ))
    # X_train, X_test, y_train, y_test = train_test_split(trainingData, sentimentList, test_size=0.33, random_state=1)
    lrModel = LogisticRegression(max_iter=1000)
    trainedModel = lrModel.fit(trainingData, sentimentList)
    Pkl_Filename = "submission/sentiment_Model.pkl"
    with open(Pkl_Filename, 'wb') as file:  
        pickle.dump(trainedModel, file)
    # prediction = trainedModel.predict(X_test)
    # accuracy = accuracy_score(y_test, prediction)
    # print(accuracy)
    namedPrediction = []
    # for i in sentimentList:
    #     if int(i) == 0:
    #         namedPrediction.append("Negative")
    #     else:
    #         namedPrediction.append("Positive")
    # #creating Data frame of required
    # textIds = range(len(trainData))
    # df = pd.DataFrame({
    #     'textId': textIds,
    #     'text': list(fileData[0]),
    #     'selectedText': finalData,
    #     'mood': list(fileData[1]),
    #     'sentiment': namedPrediction
    # })
    # df.to_csv('submission/train.csv', index=False)
    return trainedModel, vectorizer

if __name__ == "__main__":
    trainedModel = trainModel()
    # print(trainedModel)