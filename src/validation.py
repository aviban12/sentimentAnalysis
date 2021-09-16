from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from train import trainModel
from wordcloud import WordCloud
from scipy.sparse import csr_matrix
import scipy as sp
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

def transformer(data, vectorizer):
    vectorizer = CountVectorizer(vocabulary=vectorizer.vocabulary_)
    vectorized = vectorizer.fit_transform(data)
    transformer = TfidfTransformer()
    transformed = transformer.fit_transform(vectorized)
    return transformed

def test():
    print("Validation")
    from nltk.corpus import stopwords
    verbs = ['wasnt','didnt','doesnt','dont', 'wont', 'wouldnt', 'also']
    fileData = pd.read_csv('data/val.txt', sep=";", header=None)
    testData = fileData
    stopwords = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()

    uniqueLabelsDict = {}
    uniqueLables = set(list(testData[1]))
    count = 0
    for key, labels in enumerate(uniqueLables):
        uniqueLabelsDict[labels] = count
        count += 1
    markedLabels = [uniqueLabelsDict[w] for w in testData[1]]
    testData[0] = testData[0].map(lambda x: re.sub('[,\.!?]', '', x))
    testData[0] = testData[0].map(lambda x: x.lower())
    tokenizedWord = [nltk.word_tokenize(line) for line in testData[0]]
    cleanedData = []
    finalData = []
    for line in tokenizedWord:
        cleanedData.append([wordnet_lemmatizer.lemmatize(word) for word in line if word not in stopwords and word not in verbs and len(word) > 3])
    for line in cleanedData:    
        finalData.append(" ".join(w for w in line))
    trainedModel, vectorizer = trainModel()
    testData = transformer(finalData, vectorizer)
    testingData = sp.sparse.hstack((testData, csr_matrix(markedLabels).T ))
    prediction = trainedModel.predict(testingData)
    finalPrediction = [abs(round(value)) for value in prediction]
    textIds = range(len(fileData))
    namedPrediction = []
    for i in prediction:
        if int(i) == 0:
            namedPrediction.append("Negative")
        else:
            namedPrediction.append("Positive")
    df = pd.DataFrame({
        'textId': textIds,
        'text': list(fileData[0]),
        'selectedText': finalData,
        'mood': list(fileData[1]),
        'sentiment': namedPrediction
    })
    df.to_csv('submission/val.csv', index=False)
    return finalPrediction

if __name__ == "__main__":
    prediction = test()
    