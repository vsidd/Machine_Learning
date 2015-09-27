'''
Created on Apr 21, 2015

@author: Siddarthan
'''
import os
from os import walk
from elasticsearch import Elasticsearch
import time
from sklearn import tree
from math import ceil
import ast
import email
import re
import numpy as np
import pyliblinear
from scipy.sparse import dok_matrix
import operator

"""
This program classifies emails by considering all the unigrams available in the dataset as features.
With the feature list coming upto 2.5 million words (both proper english and garbage), 
a sparse matrix representation is required for the machine learning algorithm to work.
Liblinear's linear regression model is used with the sparse representation for training. 
The testing phase gave an accuracy of 99% in this approach.
"""
class emailClassifier:
    def __init__(self):
        self.inmailMap = {}
        self.trainMatrix_X_val = []
        self.trainMatrix_Y_val = []
        self.testMatrix_X_val = []
        self.testQueryIDDocID = []
        self.spamList = []
        self.hamList = []
        self.trainset = []
        self.testset = []
        
    def loadIndexFile(self):
        with open("index") as f:
            for line in f:
                lineList = line.split()
                classification = lineList[0]
                emailID = lineList[1][15:]
                self.inmailMap[emailID] = classification
                if(classification == "spam"):
                    self.spamList.append(emailID)
                else:
                    self.hamList.append(emailID)

    def extractUnigrams(self):
        unigramList = {}
        count = 0
        errcount = 0
        #term : docfreq, ttf
        with open("unigramList.txt","w+") as f:
                for docid in self.inmailMap:
                    try:
                        count = count + 1
                        print count
                        result = es.termvector(index = 'ml_classifier', doc_type='document', id=docid, 
                                               body={
                                                  "fields" : ["content"],
                                                  "offsets" : False,
                                                  "payloads" : False,
                                                  "positions" : False,
                                                  "term_statistics" : True,
                                                  "field_statistics" : False
                                                })
                        for term in result['term_vectors']['content']['terms']:
                            if len(term) >1:
                                if(term not in unigramList):
                                    unigramList[term] =""
#                                     temp = str(term) + " "+ str(result['term_vectors']['content']['terms'][term]['doc_freq']) + " " +str(result['term_vectors']['content']['terms'][term]['ttf']) 
                                    temp = str(term)
                                    print >> f, temp
                    except:
                        print "error"
                        errcount += 1
                        continue     
        print "errorcount "
        print errcount 
    
    def loadUnigrams(self):
        unigrams = {}
        index = 1
        f = open("unigramList.txt", "r")
        for currentLine in iter(f):
            currentLine = currentLine.strip()
            unigrams[currentLine] = str(index)
            index += 1
            print "ld:",str(index)
        return unigrams
    
    def createMatrix(self):          
        unigrams = self.loadUnigrams();
        self.createSparseMatrix(unigrams, "SparsetrainMatrix.txt",self.trainset)
        self.createSparseMatrix(unigrams, "SparsetestMatrix.txt",self.testset)
        
    def createSparseMatrix(self, unigrams, datasetName, dataset):
        count = 0
        with open(datasetName,"w+") as f:
            for docid in dataset:
                try:
                    count += 1
                    print datasetName+" : "+str(count)
                    classification = self.inmailMap[docid]
                    label = ""
                    if(classification == "spam"):
                        label = "1"
                    else:
                        label = "0"
                    temp = label + " "
                    result = es.termvector(index = 'ml_classifier', doc_type='document', id=docid, 
                                                   body={
                                                      "fields" : ["content"],
                                                      "offsets" : False,
                                                      "payloads" : False,
                                                      "positions" : False,
                                                      "term_statistics" : True,
                                                      "field_statistics" : False
                                                    })
                    for term in result['term_vectors']['content']['terms']:
                            if(term in unigrams):
                                temp = temp + unigrams[term]+":"+ str(result['term_vectors']['content']['terms'][term]['term_freq']) +" "
                    print >> f, temp
                except:
                    print >> f, temp


    def splitData(self):
        #spam: 50199
        #ham: 25220
        #80%,20% => 60335, 15084
        #train-60335 => 40100 spam, 20100 ham
        #test-15084 => 10099 spam, 5120 ham
        self.trainset = self.trainset + self.spamList[:40100]
        self.trainset = self.trainset + self.hamList[:20100]
        self.testset = self.testset + self.spamList[40100:]
        self.testset = self.testset + self.hamList[20100:]
        

    def trainSparseMatrix(self):
        fmobj = pyliblinear._liblinear.FeatureMatrix
        fmatrixTrain = fmobj.load("SparsetrainMatrix.txt")
        model = pyliblinear.Model
        print "training"
        modelinst = model.train(fmatrixTrain)        
        fmatrixTest = fmobj.load("SparsetestMatrix.txt")
        print "predicting"
        predict = model.predict(modelinst,fmatrixTest)
        modelinst.save("trainingSet.model")
        predictValues = list()
        print "loading predict values to a list"
        for key in predict:
            predictValues.append(key)
        testPredict = dict(zip(self.testset, predictValues))
        self.testAccuracy(testPredict)
        self.topFeatures()
    
    def testAccuracy(self, testPredict):
        correct = 0
        wrong = 0
        for key in testPredict:
            classification = 0
            if(self.inmailMap[key] == "spam"):
                classification = 1
            if(testPredict[key] == classification):
                correct += 1
            else:
                wrong += 1
        print "correct : ",correct
        print "wrong : ", wrong
        print "accuracy : ",(correct/float(correct + wrong))
        
    def topFeatures(self):
        featurefile = open("unigramList.txt", "r")
        featureList = []
        weightList = []
        for currentLine in iter(featurefile):
            currentLine = currentLine.strip()
            featureList.append(currentLine)
        f = open("trainingSet.model", "r")
        flag = False
        for currentLine in iter(f):
            currentLine = currentLine.strip()
            if(flag):
                weightList.append(float(currentLine))
#                 weightList.append((currentLine))
            if(currentLine == "w"):
                flag = True
        featureMap = dict(zip(featureList, weightList))
        featuresSorted = sorted(featureMap.items(), key = operator.itemgetter(1))
        print "Top features: "
        featuresSorted.reverse()
        print featuresSorted[:200]
        
    
        

start = time.time()
es = Elasticsearch(['localhost:9203'],timeout=120, cluster = 'lasttry')
temp = emailClassifier()

# indexedDocuments = es.search(index='team_apple_merge',body={"query": {"match_all": {}},"fields": ["author"]}, size=24819)
# docIDs = [doc['_id'] for doc in indexedDocuments['hits']['hits']]

# indexedDocuments = es.search(index='ml_classifier',doc_type = "document",body={"query": {"match_all": {}},"fields": ["spam"]}, size=75500)
# docIDs = [doc['_id'] for doc in indexedDocuments['hits']['hits']]

print "loading index"
temp.loadIndexFile()
print "forming unigram list"
temp.extractUnigrams()
print "creating train and set ids"
temp.splitData()

print "forming the sparse matrix"
temp.createMatrix()

print "train matrix"
temp.trainSparseMatrix()
print "creating index"
temp.createIndex(es)
temp.removeMissedFiles()    #12916 files are missed
 

print "spliting training and testing data"
temp.splitData()

print "creating training matrix"
temp.createTrainingMatrix()
  
end = time.time()
print "TOTAL TIME : ",end - start

#spam: 50199
#ham: 25220
#80%,20% => 60335, 15084
#train-60335 => 40100 spam, 20100 ham
#test-15084 => 10099 spam, 5120 ham

#predicted ham 2309
#predicted spam 12910
 