from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer #converts running, run, ran to run since they mean the same thing
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
numLines = 10000
minCount, maxCount = 50, 1000

def createLexicon(pos, neg):
    lexicon = []
    for file in [pos, neg]:
        with open(file, 'r') as f:
            contents = f.read()
            for line in contents[:numLines]:
                words = word_tokenize(line.lower())
                lexicon += list(words)
                
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    counts = Counter(lexicon)
    
    lex2 = []
    
    for word in counts:
        if maxCount > counts[word] > minCount:
            lex2.append(word)
    
    return lex2
            
def handleSample(sample, lexicon, classification):
    allFeats = []
    
    with open(sample, 'r') as file:
        contents = file.read()
        for line in contents[:numLines]:
            currentWords = word_tokenize(line.lower())
            currentWords = [lemmatizer.lemmatize(i) for i in currentWords]
            features = np.zeros(len(lexicon))
            for word in currentWords:
                if word.lower() in lexicon:
                    index = lexicon.index(word.lower())
                    features[index] += 1
            features = list(features)
            allFeats.append([features, classification])
            
    return allFeats

def createFeaturesAndLabels(pos, neg, testSize=0.1):
    lexicon = createLexicon(pos, neg)
    features = []
    features += handleSample('pos.txt', lexicon, [1, 0])
    features += handleSample('neg.txt', lexicon, [0, 1])
    random.shuffle(features)
    #Rewatch videos to take notes, crunched for time right now
    features = np.array(features)
    
    testSize = int(len(features) * testSize)
    trainX = list(features[:,0][:-testSize])#0th element in each array
    trainY = list(features[:,1][:-testSize])
    testX = list(features[:,0][-testSize:])
    testY = list(features[:,1][-testSize:])
            
    return trainX, trainY, testX, testY

if __name__ == '__main__':
    X, y, testX, testY = createFeaturesAndLabels('pos.txt', 'neg.txt')
    with open('sentimentData.pickle', 'wb') as file:
        pickle.dump([X, y, testX, testY], file)

            
            