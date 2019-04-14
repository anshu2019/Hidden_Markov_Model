# Import the required librray
import sys
import numpy as np
import math
import re
import csv
import json
eps = np.finfo(float).eps
from numpy import log2 as log
import time
import decimal

decimal.getcontext().prec = 20

#commandline param
path1=sys.argv[1]
path2=sys.argv[2]
path3=sys.argv[3]
path4=sys.argv[4]
path5=sys.argv[5]
path6=sys.argv[6]
path7=sys.argv[7]
path8=sys.argv[8]

start = time.time()
#variables
word_dict =[]
tag_dict = []
dict_w={}
dict_t={}
train_data =[]
obsrv_cnt = 0
tag_count=0
wrd_count=0
initl_prob=[]
trans_prob =[]
emit_prob=[]
accuracy_count=np.zeros(1)
seq_count=np.zeros(1)
log_likelyhood=np.zeros(1)
avg_lg_liklyhood =0
model_accuracy =0

#load all words
predictFile = open(path7,"w")
metricsFile = open(path8,"w")

#Normalize array
def normalize_arry(lis):
    outp=[]
    for m in lis:
        prob= float(m)/float(np.sum(lis))
        outp.append(prob)
    return outp 

#make dict
def make_dictionary(lis):
    d={}
    for i in range(len(lis)):
        d[lis[i]]=i
    return d

# Make string data from files as float
def str_to_float(lis):
    return [[float(j.strip()) for j in i] for i in lis]

#minimum Bayes Risk Prediction
def bayes_prection(lis):
    idx = np.argmax(lis)
    tag=tag_dict[idx]
    return tag
    
#return tags and words
def preapretestdata(line):
    tokn = re.split('[_  ]',line) 
    X=[]
    Y=[]
    for i in range(len(tokn)):
        if((i%2)==0):
            X.append(tokn[i]) #word
        else:
            Y.append(tokn[i]) #tags
    return X, Y

#predic tag
def predicttag(alpha,beeta,Y,X):
    fileline=""
    o= np.multiply(alpha,beeta)
    
    for u in range(len(o)):
        tag = bayes_prection(o[u])        
        if tag == Y[u].strip():            
            accuracy_count[0]+=1
            fileline+=  X[u].strip()+"_"+tag+" "
        else:
            fileline+=  X[u].strip()+"_"+tag+" "
    #write data to file
    predictionFile.writelines(fileline.strip()+"\n")
        
           
            
#forward backward algorithm
def frwd_bckwd_algorithm(line,seq_count,log_likelyhood):
    X,Y = preapretestdata(line)
    X_rev = X[::-1]    
        
    fwd=[]
    prev_fwd=[]  
    # forward part of the algorithm
    for i in range(len(X)):
        seq_count[0]+=1
        curr_fwd=np.zeros(tag_count)
        for st in tag_dict:
            state_idx = dict_t.get(st)
            wrd_idx   = dict_w.get(X[i])
            if i==0: # starting state
                 prev_fwd_sum = (initl_prob[state_idx])
            else:
                 prev_fwd_sum =(np.sum((prev_fwd[dict_t.get(k)])*trans_prob[dict_t.get(k)][state_idx] for k in tag_dict))
            
            curr_fwd[state_idx] = (emit_prob[state_idx][wrd_idx]) * (prev_fwd_sum)
        
        if(i != len(X)-1): # dont normalize last column            
             curr_fwd=normalize_arry(curr_fwd)                 
                     
        fwd.append(curr_fwd)
        prev_fwd = curr_fwd  
    #calculate log likelyhood at sequence T(use un normalized alpha column)
    log_likelyhood[0] +=math.log((np.sum(fwd[len(X)-1])))
    alpha = fwd
    #print(fwd)    
    
    # backward part of the algorithm
    bkw = []
    prev_bkw = []
    for i in range(len(X_rev)):
        curr_bkw=np.zeros(tag_count)
        for st in tag_dict:
            state_idx = dict_t.get(st)
            wrd_idx   = dict_w.get(X_rev[i])
            if i==0: # end state
                curr_bkw[state_idx]=1
            else:
                curr_bkw[state_idx]=np.sum(trans_prob[state_idx][dict_t.get(l)] * emit_prob[dict_t.get(l)][dict_w.get(X_rev[i-1])] * (prev_bkw[dict_t.get(l)]) for l in tag_dict)
        
        curr_bkw=normalize_arry(curr_bkw) 
        bkw.append(curr_bkw)
        prev_bkw = curr_bkw 
    # reverse data because we , have iterated from T to 1 sequence backward
    beeta =bkw[::-1]
    #write output to file
    predicttag(alpha,beeta,Y,X)
    #print(beeta)
    
    
#------------------------------------------------------------------------------------------
accuracy_count[0]=0
seq_count[0]=0
log_likelyhood[0]=0

predictionFile = open(path7,"w")
metricsFile = open(path8,"w")
#load all words
with open(path2, 'r') as txt1:
    for r in txt1:
        word_dict.append(r.strip())

# load all tags
with open(path3, 'r') as txt2:
    for r in txt2:
        tag_dict.append(r.strip())        

dict_w=make_dictionary(word_dict)             # word dictionary
dict_t=make_dictionary(tag_dict)              # tag dictionary
tag_count=len(tag_dict)                       # number of tags
wrd_count =len(word_dict)                     # number of words
        
#load initial probablity
with open(path4, 'r') as txt4:
    for r in txt4:
        r=float(r.strip())
        initl_prob.append(r)   

#load emission probablity
with open(path5, 'r') as txt5:
    for r in txt5:
        row = r.strip().split()
        emit_prob.append(row)
                          
#load transition probablity
with open(path6, 'r') as txt6:
    for r in txt6:
        row = r.strip().split()        
        trans_prob.append(row)
                          
emit_prob =str_to_float(emit_prob)      #emission probablity
trans_prob =str_to_float(trans_prob)    #transition probablity

# load test data
with open(path1, 'r') as txt3:
    for r in txt3:         
        train_data.append(r)
        frwd_bckwd_algorithm(r,seq_count,log_likelyhood)

obsrv_cnt= len(train_data)  # number of observation/sequences

#create data for metrics file
avg_lg_liklyhood = float(log_likelyhood[0])/obsrv_cnt              # calculate log likelyhood
model_accuracy = float(accuracy_count[0])/float(seq_count[0])      # calculate the accuracy
metricsFile.writelines("Average Log-Likelihood: "+str(avg_lg_liklyhood)+"\n")
metricsFile.writelines("Accuracy: "+str(model_accuracy)+"\n")

print(avg_lg_liklyhood)
print(model_accuracy)

#close the files
predictionFile.close()
metricsFile.close()
# run your code
end = time.time()
elapsed = end - start
print("done in : %f" %(elapsed))
