import sys
# from geopy.distance import geodesic
import numpy as np
import csv

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math
from itertools import combinations
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import networkx as nx

import pdb


#randomwalk based curvature regularization
#generate random walks for entire nodes

#For theorem.1

class abs_curvature_regularization:
    def __init__(self, walks, num_walks, num_nodes, dimension, original, file_format):
        self.D_in = dimension#64(karate.dataset)
        self.D_out = dimension
        self.H = 64
        self.X_train = [] 
        self.dim = dimension
        self.y = torch.ones([1, self.dim], dtype=torch.float64)
        self.V = num_nodes#34(karate.dataset)
        self.criterion = nn.CosineEmbeddingLoss()
        self.cos = nn.CosineSimilarity(eps=1e-6)
        self.original_embeddings = torch.tensor(original)#shape(34,multi_dim(in here 34))
        self.dtype = torch.DoubleTensor
        self.learning_rate = 0.001#1e-6
        self.walks = walks#list 
#         self.vectorized_walks = node_to_vector(self.original_embeddings, self.walks)#each item = dictionary 
#         self.vectorized_edges = embeddings_of_edges(self.vectorized_walks)
        self.format = file_format
    
    #-------------------------------------------------------------------------------#
    #                       meet the condition of Theorem 1.                        #
    #-------------------------------------------------------------------------------#
    
        #--absolute value of summation of curvatures along any part of polygonal curve is less than pi/2--#
    def optimization(self):     
        #--minimize sum of entire_abs til meet theorem 1.--#
        
        self.initialize()
        epoch = 0
        output = self.original_embeddings
        
        cos_tot = cosine_for_nodes(self.original_embeddings, self.walks, self.dim, self.V)
        
#         # minimize til any part of P'_{ij} is less than pi/2(90 degree)(less than cosine 0)
#         while any(entire_abs[i] >= 90 for i in range(len(entire_abs))):
        while any(cos_tot[i] <= 0 for i in range(len(cos_tot))):

            X = Variable(self.original_embeddings.type(self.dtype), requires_grad=False)
            
            print("X:{}".format(X.shape))

            print("W:{}".format(self.W1))
            output = self.train(X)

            print("Output:{}".format(output))
            #print("shape:{}".format(output.shape))

            self.original_embeddings = output

            cost_tot = cosine_for_nodes(self.original_embeddings, self.walks, self.dim, self.V)
            epoch += 1
            print("epoch:{}".format(epoch))

        plot_embeddings(self.V, output.detach().numpy())
        print("epoch:{}".format(epoch)) 
        
        return output

      
        
    #-------------------------------------------------------------------------------#
    #                         minimize two terms jointly                            #
    #-------------------------------------------------------------------------------#
    
    #-- get a flat embedding manifold --#
    #def curvature_regularization(self):

       
   
    #-------------------------------------------------------------------------------#
    #                         update embeddings(regularization)                     #
    #-------------------------------------------------------------------------------#
    
    def initialize(self):
        self.W1 = Variable(torch.randn(self.D_in, self.H).type(self.dtype), requires_grad=True)
        self.W2 = Variable(torch.randn(self.H, self.D_out).type(self.dtype), requires_grad=True)
        
    def feed_forward(self, original_embeddings): 
        updated_embeddings = torch.tanh(original_embeddings.mm(self.W1)).mm(self.W2)

        #y_pred = torch.tanh(y_pred)#(34,34)

        return updated_embeddings

    def train(self, X):
        
        updated_embeddings = self.feed_forward(X)#maintain tensor
        vectorized_walks = node_to_vector(updated_embeddings, self.walks)#each item = dictionary 
        vectorized_edges = embeddings_of_edges(vectorized_walks)
        
        optimizer = torch.optim.Adam([self.W1, self.W2], lr=self.learning_rate)
        
        loss_tot = 0
        
        #cosine_function    
        for walk in vectorized_edges:
            if len(walk) > 1:
                for edge_num in range(len(walk)-1):#edge_num == each edge embedding(array)
                    cosine_loss = self.criterion(walk[edge_num].reshape(1,int(self.dim)), 
                                                 walk[edge_num+1].reshape(1,int(self.dim)), self.y)#
                    loss_tot += cosine_loss.item()#scalar value of loss
                    optimizer.zero_grad()
                    cosine_loss.backward(retain_graph=True)#
                    optimizer.step()
                    
        #loss_tot = torch.div(loss_tot,self.V)#average loss
        
        print("loss_tot:{}".format(loss_tot))
        
        return updated_embeddings

#-------------------------------------------------------------------------------#
#                              get_abs functions                                #
#-------------------------------------------------------------------------------#

#convert nodes in randomwalk to n-dim vector
def node_to_vector(original_embeddings, walks):
    node_embeddings = original_embeddings
    walks = walks
    vectorized_walks = []

    for i in range(len(walks)):#iter 340(340 number of random_walks: 10 for each 34 nodes)
        #for each random_walk(current random_walk)
        cur_walk = walks[i]#length:10

        #calculate ABS curvature of each node q in i random_walk 
        walk_dict = {}
        for q in cur_walk:
            #get n-dim embedding of node q
            walk_dict[int(q)] = node_embeddings[int(q)]

        vectorized_walks.append(walk_dict)

    return vectorized_walks

#generate embeddings of edges
def embeddings_of_edges(vectorized_walks):
    
    vectorized_edges = []
    for walk in vectorized_walks:
        if len(walk.keys()) > 1:
            node_list = []
            edge_list = []
            for index, (node, vectorized_node) in enumerate(walk.items()):
                node_list.append(vectorized_node)        
            for i in range(len(node_list)-1):
                edge_list.append(node_list[i+1] - node_list[i])
                
            vectorized_edges.append(edge_list)

    return vectorized_edges
                      
def cosine_for_nodes(original_embeddings, walks, dimension, V):
    cosine_val = []
    
    #print(original_embeddings)
    cos = nn.CosineSimilarity(dim = 0, eps=1e-6)
    for walk in walks:
            if len(walk) > 1:
                for node_num in walk[0:-1]:
                    if int(node_num) < V-1:
                        cos_val = cos(original_embeddings[int(node_num)],
                                      original_embeddings[int(node_num)+1])
                        cosine_val.append(cos_val)
                         
    return cosine_val

def plot_embeddings(num_node, embeddings):
    tsne = TSNE(n_components=2, random_state=0)
    
    X_2d = tsne.fit_transform(embeddings)
    
    target_ids = range(num_node)

    color = []
    group_edges = []
    
    #karate.label
    if num_node == 34:
        with open("dataset/karate.label", 'r') as f:
            colored_labels = []
            labels = []
            for line in f.readlines():
                if int(line) == 1:
                    colored_labels.append("orange")# 1: Mr.Hi
                    labels.append(line)
                else:
                    colored_labels.append("purple")# 34: John A
                    labels.append(line)
                    
        plt.figure(figsize=(6, 5))
    
        for i in target_ids:
            plt.scatter(embeddings[i, 0], embeddings[i, 1], c=colored_labels[i])

    plt.show()
    plt.savefig('embedding_visualization.png')
    
    #blogcatalog (39 groups)
    if num_node == 10312:
        number_of_colors = 39
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(number_of_colors)]
        
        with open("dataset/group-edges.csv", 'r') as f:
            group_edges = [None]*num_node
            for line in f.readlines():
                group_edges[int(line[0])-1] = int(line[1])-1
                
        plt.figure(figsize=(6, 5))
    
        for i in target_ids:
            plt.scatter(embeddings[i, 0], embeddings[i, 1], c=color[group_edges[i]])

    plt.show()
    plt.savefig('embedding_visualization.png')
    #cora         
    if num_node == 2708:
        number_of_colors = 7
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(number_of_colors)]
        
        labels = sp.load_npz("dataset/cora/labels.npz")
        group_edges = labels.indices
        
        plt.figure(figsize=(6, 5))
    
        for i in target_ids:
            plt.scatter(embeddings[i, 0], embeddings[i, 1], c=color[group_edges[i]])
