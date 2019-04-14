
# coding: utf-8

# In[118]:


# Import the required librray
import sys
import numpy as np
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
tag_cntr=[]
trans_mat =[]
emit_mat=[]
wrd_cntr=[]

#make dict
def make_dictionary(lis):
    d={}
    for i in range(len(lis)):
        d[lis[i]]=i
    return d
        

#function to prepare training file. We will replace word and tag with their respective indices from dictionary
def processdata(line):
    tokn = re.split('[_  ]',line)    
    # get first tag in sequence to calcualte prior
    tag1 =tokn[1].strip()    
    if tag1 in tag_dict:        
        tag_cntr[dict_t.get(tag1)]=tag_cntr[dict_t.get(tag1)]+ 1
    
    # get data for transition matrix
    for i in range(len(tokn)):
        if(i%2 ==0 and i > 0):
            tag1_idx=dict_t.get(tokn[i-1].strip())
            tag2_idx=dict_t.get(tokn[i+1].strip())
            trans_mat[tag1_idx][tag2_idx]+=1
    
    # get adta for emission matrix
    for i in range(len(tokn)):
        if(i%2 !=0):
            tag_idx=dict_t.get(tokn[i].strip())
            wrd_idx=dict_w.get(tokn[i-1].strip())
            emit_mat[tag_idx][wrd_idx]+=1
    
    
#function to calculate probablity    
def get_Prob(file1,file2,file3):
    #initial probablity
    for rw in tag_cntr:
        prob = float(rw +1)/(np.sum(tag_cntr)+ 1*len(tag_cntr))
        file1.writelines(format(prob, '.20f') + " "+'\n')
    
    #transition probability   
    for t_t in trans_mat:
        trns_res=""
        for t in t_t:
            prob_trns = float(t +1)/(np.sum(t_t)+ 1*len(t_t))
            trns_res+= format(prob_trns, '.20f') + " "           
        file2.writelines(trns_res.strip()+'\n')
    
    #emission probablity
    for e_t in emit_mat:
        emit_res=""
        for t in e_t:
            prob_emit = float(t +1)/(np.sum(e_t)+ 1*len(e_t))
            emit_res+= format(prob_emit, '.20f') + " "           
        file3.writelines(emit_res.strip()+'\n')
        
        
#load all words
piFile = open(path4,"w")
trnsFile = open(path6,"w")
emitFile = open(path5,"w")

#load all words
with open(path2, 'r') as txt1:
    for r in txt1:
        word_dict.append(r.strip())

# load all tags
with open(path3, 'r') as txt2:
    for r in txt2:
        tag_dict.append(r.strip())

dict_w=make_dictionary(word_dict) 
dict_t=make_dictionary(tag_dict)
tag_count=len(tag_dict)                       # number of tags 
tag_cntr= np.zeros(tag_count)                 #initialize a tag counter
trans_mat =np.zeros([tag_count,tag_count])    # create a transition matrix
wrd_count =len(word_dict)                     # number of words
emit_mat=np.zeros([tag_count,wrd_count])      #create a emission matrix

# load training data
with open(path1, 'r') as txt3:
    for r in txt3:         
        train_data.append(r)
        processdata(r)

obsrv_cnt= len(train_data)

# get the probablities for HMM model
get_Prob(piFile,trnsFile,emitFile)

#close all files
piFile.close()
trnsFile.close()
emitFile.close()
# run your code
end = time.time()

elapsed = end - start
print("done in : %f" %(elapsed))