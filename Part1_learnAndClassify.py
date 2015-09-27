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
import xml.etree.ElementTree as ET


"""
This program classifies the emails based on user defined features.
"""

class emailClassifier:
    def __init__(self):
        self.inmailMap = {}
        self.features = ["weight loss","click here","act now", "at home", "your money","credit cards",
                         "find out","for free","growth hormone","click below","human growth","join millions","viagra and",
                         "apply now","lowest price","limited time","money back","online pharmacy","no obligation"]
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

    def removeMissedFiles(self):
        with open("missedFiles.txt") as f:
            content = f.read()
            self.missedFileNames = ast.literal_eval(content)
            for key in self.missedFileNames:
                if key in self.inmailMap:
                    del self.inmailMap[key]
    
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
        
    
    def createIndex(self,es):
        filename = os.path.join(os.path.dirname(__file__), '/College/IR/HW6/trec07clean/clean/')
#         filename = os.path.join(os.path.dirname(__file__), '/College/projects/HW6/ml_classifier/cleanData/')
        self.missedFiles = 0
        count = 0
        self.missedFileNames = []
        print "reading files"
        for (dirpath, dirnames, filenames) in walk(filename): 
            for file in filenames:
                print dirpath+file
                docFile = ET.parse(dirpath+file)
                docs = docFile.findall("DOC")
                for doc in docs:
                    fName = doc.find("DOCNO").text
                    fName = fName[7:]
                    content = doc.find("SUBJECT").text
                    if content is None:
                        content = ""
                    textField = doc.find("TEXT").text
                    if(textField is not None):
                        content = content + " "+  textField
                    try:    
                        content = self.extractBody(content,fName)
                        content.encode('utf-8')
                        spam = "no"
                        if self.inmailMap[fName] == "spam":
                            spam = "yes"
                        es.index(index = 'ml_classifier', doc_type='document', body={'content':content,'spam':spam}, id=fName)
                        count += 1
                        print fName," ",spam," ",count
                    except Exception,e:
                        print e
                        print "missed file count : "+str(self.missedFiles)
                        self.missedFileNames.append(fName)
                        self.missedFiles += 1
                        continue
        print "missed files : ",self.missedFiles   
        with open("missedFiles.txt", 'w+') as f:
                print >> f, self.missedFileNames
                    
    def extractBody(self,content,fname):
        mailContent = ""
        msg = email.message_from_string(content)
        if (msg['Subject']) != None:
            mailContent = (msg['Subject'])
        if msg.is_multipart():
            for payload in msg.get_payload():
                mailContent += " " + ''.join(str(v) for v in payload.get_payload())
        else:
            mailContent += " " + msg.get_payload()
        mailContent = self.processText(mailContent)
        finalString = ""
        for word in mailContent.split():
            if(len(word) < 15 and len(word) > 2):
                finalString = finalString + word+" "
        return finalString
    
    def processText(self,currentLine):
        currentLine = re.sub(r"http\S+", "", currentLine)
        currentLine = re.sub("<.*?>", " ", currentLine)
        currentLine = re.sub('[,"\';:=^+/]',' ',currentLine)
        currentLine = re.sub('[-.:@?!#&<>()~`*]',' ',currentLine)
#         currentLine = currentLine.replace("."," ")
        currentLine = re.sub('[^0-9a-zA-Z]+', ' ', currentLine)
        currentLine = ' '.join(currentLine.split())
        currentLine = currentLine.strip()
        currentLine = currentLine.lower()
        return currentLine
    
    
    def createTrainingMatrix(self):
        self.featureMaps = []
        for feature in self.features:
                scriptResult = es.search(index='ml_classifier',body={"query": {"function_score": {"query": {"match_phrase": {"content": feature}},"functions": [{"script_score": {"script_id": "getTF","lang" : "groovy","params": {"term": feature,"field": "content"}}}],"boost_mode": "replace"}},"size": 75500,"fields" : ["spam"]})
                self.featureMaps.append(self.processScriptResult(scriptResult))
        
        for mailID in self.trainset:
            featureRowXval = []
            for featureMap in self.featureMaps:
                if mailID in featureMap:
                    featureRowXval.append(featureMap[mailID])
                else:
                    featureRowXval.append(0.0)
            featureRowYval = self.inmailMap[mailID]
            self.trainMatrix_X_val.append(featureRowXval)
            self.trainMatrix_Y_val.append(featureRowYval)
    
    def train_algorithm(self):
        self.clf = tree.DecisionTreeClassifier()
        with open("tempFile.txt","w+") as f:
            for i in range(len(self.trainMatrix_X_val)):
                temp = str(self.trainMatrix_X_val[i]) + " " + str(self.trainMatrix_Y_val[i])
                print >> f, temp
        self.clf = self.clf.fit(self.trainMatrix_X_val, self.trainMatrix_Y_val)
        self.predict_result = self.clf.predict(self.trainMatrix_X_val)
        self.print_result()

    def print_result(self):
        correct = 0
        wrong = 0
        i = 0
        for mailID in self.trainset:
            if self.inmailMap[mailID] == self.predict_result[i]:
                correct += 1
            else:
                wrong += 1
            i += 1
        print "correct : ",correct
        print "wrong : ",wrong
        accuracy = correct/float(correct+wrong)
        print "accuracy : ",ceil(accuracy*100),"%"
    
    def createTestMatrix(self):
        self.TestFeatureMaps = []
        for feature in self.features:
                scriptResult = es.search(index='ml_classifier',body={"query": {"function_score": {"query": {"match_phrase": {"content": feature}},"functions": [{"script_score": {"script_id": "getTF","lang" : "groovy","params": {"term": feature,"field": "content"}}}],"boost_mode": "replace"}},"size": 75500,"fields" : ["spam"]})
                self.TestFeatureMaps.append(self.processScriptResult(scriptResult))
        for doc in self.testset:
            featureRowXval = []
            for featureMap in self.TestFeatureMaps:
                if doc in featureMap:
                    featureRowXval.append(featureMap[doc])
                else:
                    featureRowXval.append(0.0)
            self.testMatrix_X_val.append(featureRowXval)
        
    def testCrawlData(self):
        crawl_classification = self.clf.predict(self.testMatrix_X_val)
        i = 0
        j = 0
        print "printing spam ids"
        for id in self.testset:
            if(crawl_classification[i] == "spam"):
                j = j+1
                print id+" : " + crawl_classification[i] + ": "+str(j)
            i = i+1
        #predicted ham 2309
        #predicted spam 12910
        
    def processScriptResult(self, scriptResult):
        docScore = {}
        for doc in scriptResult['hits']['hits']:
            docScore[doc['_id']] =  1
        return docScore
    
    def dumpSpamFiles(self):
        searchResult = es.search(index="hw3_merged_data", doc_type="document", body={"query":{"match":{"spam":"spam"}}}, size=10)
        with open("someSpamFiles.txt", 'w+') as f:
            for doc in searchResult['hits']['hits']:
                url = doc['_id']
                print >> f, url
            



start = time.time()
es = Elasticsearch(['localhost:9203'],timeout=120, cluster = 'lasttry')
temp = emailClassifier()

# indexedDocuments = es.search(index='team_apple_merge',body={"query": {"match_all": {}},"fields": ["author"]}, size=24819)
# docIDs = [doc['_id'] for doc in indexedDocuments['hits']['hits']]

# indexedDocuments = es.search(index='ml_classifier',doc_type = "document",body={"query": {"match_all": {}},"fields": ["spam"]}, size=75500)
# docIDs = [doc['_id'] for doc in indexedDocuments['hits']['hits']]

print "loading index"
temp.loadIndexFile()
print "creating index"
temp.createIndex(es)

print "spliting training and testing data"
temp.splitData()
 
print "creating training matrix"
temp.createTrainingMatrix()
 
print "train algo"
temp.train_algorithm()
  
print "create test matrix"
temp.createTestMatrix()
  
print "test crawl data"
temp.testCrawlData()
 
print "dump spam files"
temp.dumpSpamFiles()
end = time.time()
print "TOTAL TIME : ",end - start

#spam: 50199
#ham: 25220
#80%,20% => 60335, 15084
#train-60335 => 40100 spam, 20100 ham
#test-15084 => 10099 spam, 5120 ham

#predicted ham 2309
#predicted spam 12910
 














    
