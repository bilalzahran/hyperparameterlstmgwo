import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import random

import data
import model
import languagemodel

torch.manual_seed(1111)
device = torch.device("cuda")

if __name__ == "__main__":
    ###############################################################################
    # Grey Wolf Optimizer Parameter, Lower Bound & Lower Bound for Search Agent
    ###############################################################################
    SearchAgents_no = 1
    Max_iter = 1
    a = 2
    ub = [800,0.5,0.5,0.5] #This is upper bound for the n of neurons, recurrent dropout, embedding dropout, and output dropout
    lb = [300,0.3,0.3,0.3] #This is lower bound for the n of neurons, recurrent dropout, embedding dropout, and output dropout
    dim = 4 #The dimension of the search agent

    ###############################################################################
    # Hyperparameters for LSTM (Constant)
    ###############################################################################
    batch_size = 10
    lr = 10
    batch_size = 20
    bptt = 30
    nlayers = 2  
    epochs = 1

    ###############################################################################
    # Load data
    ###############################################################################
    data_path = ".\data"
    corpus = data.Corpus(data_path)

    print("Number of tokens:")
    print("Train: ", len(corpus.train))
    print("Valid: ", len(corpus.valid))
    print("Test:  ", len(corpus.test))

    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)

    train_data = batchify(corpus.train, batch_size)
    val_data = batchify(corpus.valid, batch_size)
    test_data = batchify(corpus.test, batch_size)

    ###############################################################################
    # Defining the mapping function for the search agents and the fitness function for GWO
    ###############################################################################
    def map_search_agent(value,index):
        search_agent = (value * (ub[index] - lb[index])) + lb[index]
        if index == 0:
            return int(np.rint(search_agent))    
        else:
            return search_agent
        
    def objective(nhid,dropout,odropout,edropout):
        print("Training LSTM Model with",map_search_agent(nhid,0),"Neurons;", map_search_agent(dropout,1),"Recurrent Dropout-rate;", map_search_agent(odropout,2),"Output Dropout-rate;", map_search_agent(edropout,3),"Embedding Dropout-rate;")
        lm = languagemodel.LanguageModel(map_search_agent(dropout,1),map_search_agent(odropout,2),map_search_agent(edropout,3),map_search_agent(nhid,0),map_search_agent(nhid,0),corpus,lr,nlayers,bptt,batch_size)
        fitness = lm.training(train_data,val_data,epochs,lr,batch_size)
        return fitness

    # Grey Wolf Optimizer Algorithm
    Alpha_pos=np.zeros(dim)
    Alpha_score=float("inf")

    Beta_pos=np.zeros(dim)
    Beta_score=float("inf")

    Delta_pos=np.zeros(dim)
    Delta_score=float("inf")
    
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
        
    #Initialize the positions of search agents
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.uniform(0,1, SearchAgents_no)
    print("\nSearch Agents:")
    print(Positions,"\n")
    
    for m in range(Max_iter):
        for i in range(SearchAgents_no):
            print('-' * 99)
            print("ITERATION: ",m)
            print('-' * 99)
            print(i," Of",SearchAgents_no," Agent")
            # Return back the search agents that go beyond the boundaries of the search space
            
            for j in range(dim):
                Positions[i,j]=np.clip(Positions[i,j], 0, 1)
                
            fitness=objective(Positions[i,0],Positions[i,1],Positions[i,2],Positions[i,3])
            print("Fitness:",fitness)
            
            #Updating the Alpha, Beta, Delta Positions
            if fitness<Alpha_score :
                Alpha_score=fitness; # Update alpha
                Alpha_pos=Positions[i,:].copy()
        
            if (fitness>Alpha_score and fitness<Beta_score ):
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i,:].copy()
            
            if (fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score): 
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i,:].copy()
                
        a=2-m*((2)/Max_iter); # a decreases linearly fron 2 to 0
        
        for k in range(SearchAgents_no):
            for l in range (0,dim):     
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A1=2*a*r1-a; # Equation (3.3)
                C1=2*r2; # Equation (3.4)
                
                D_alpha=abs(C1*Alpha_pos[l]-Positions[k,l]); # Equation (3.5)-part 1
                X1=Alpha_pos[l]-A1*D_alpha; # Equation (3.6)-part 1
                            
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)
                
                D_beta=abs(C2*Beta_pos[l]-Positions[k,l]); # Equation (3.5)-part 2
                X2=Beta_pos[l]-A2*D_beta; # Equation (3.6)-part 2       
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a; # Equation (3.3)
                C3=2*r2; # Equation (3.4)
                
                D_delta=abs(C3*Delta_pos[l]-Positions[k,l]); # Equation (3.5)-part 3
                X3=Delta_pos[l]-A3*D_delta; # Equation (3.5)-part 3             
                
                Positions[k,l]=(X1+X2+X3)/3  # Equation (3.7)
        
        print("\nIn Iteration ",m," Best Score is ",Alpha_score)
    
    print("\nThe Best Parameter Setting:")
    print("Neurons: ",map_search_agent(Alpha_pos[0],0),",Recurrent Dropout: ",map_search_agent(Alpha_pos[1],1), ",Output Dropout: ", map_search_agent(Alpha_pos[2],2), ",Embedding Dropout: ",map_search_agent(Alpha_pos[3],3))
    
    print('-' * 99)
    print("\nTraining and testing LSTM with the hyperparameters from GWO...\n")
    print('-' * 99)
    data_path = ".\data\\full"
    corpus = data.Corpus(data_path)
    train_data = batchify(corpus.train, batch_size)
    val_data = batchify(corpus.valid, batch_size)
    test_data = batchify(corpus.test, batch_size)
    
    lm = languagemodel.LanguageModel(map_search_agent(Alpha_pos[1],1),map_search_agent(Alpha_pos[2],2),map_search_agent(Alpha_pos[3],3),map_search_agent(Alpha_pos[0],0),map_search_agent(Alpha_pos[0],0),corpus,lr,nlayers,bptt,batch_size)
    lm.training(train_data,val_data,epochs,lr,batch_size)