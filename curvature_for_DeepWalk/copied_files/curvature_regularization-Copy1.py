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
        self.y = torch.ones([1, 64], dtype=torch.float64) 
        self.V = num_nodes#34(karate.dataset)
        self.criterion = nn.CosineEmbeddingLoss()
        self.cos = nn.CosineSimilarity(eps=1e-6)
        self.original_embeddings = original#shape(34,multi_dim(in here 34))
        #self.W1 = W
        self.dtype = torch.FloatTensor
        self.learning_rate = 0.001#1e-6#learning_rate
        self.walks = walks#e.g., [['24', '24', '25', '25', '24', '31', '32', '18', '18', '33'],[...],...] list 
        self.vectorized_walks = node_to_vector(self.original_embeddings, self.walks)#each item = dictionary (340,)
        self.format = file_format
    
    #-------------------------------------------------------------------------------#
    #                       meet the condition of Theorem 1.                        #
    #-------------------------------------------------------------------------------#
    
        #--absolute value of summation of curvatures along any part of polygonal curve is less than pi/2--#
    def optimization(self):     
        #--minimize sum of entire_abs til meet theorem 1.--#

#         # get turning angle of every nodes that appear in each random walk
#         abs_for_nodes = get_abs_for_nodes(self.walks, self.vectorized_walks)
#         # return summation of absolute value of curvatures for each random walk, shape(198,)
#         entire_abs = total_abs_for_each_polygonal(abs_for_nodes)
#         # return cosine summation per each randomwalk, shape(198,)
#         cosine_sum = sum_cosine_of_curvature(abs_for_nodes)
        
#         each_cosine_sum = each_sum_cosine_of_curvature(self.V, abs_for_nodes)
        
#         print(np.array(self.walks).shape)#shape(340,)
#         print(np.array(abs_for_nodes).shape)#shape(198,)
#         print(np.array(cosine_sum).shape)#shape(198,)
        
            
        self.initialize()
        epoch = 0
        
        cos_tot = cosine_for_nodes(self.original_embeddings, self.walks, self.dim, self.V)
        
#         # minimize til any part of P'_{ij} is less than pi/2(90 degree)(less than cosine 0)
#         while any(entire_abs[i] >= 90 for i in range(len(entire_abs))):
        while any(cos_tot[i] <= 0 for i in range(len(cos_tot))):

            X = Variable(torch.tensor(self.original_embeddings).type(self.dtype), requires_grad=False)

            print("X:{}".format(X.shape))

            print("W:{}".format(self.W1))
            output = self.train(X, self.walks)
            
            plot_embeddings(self.V, output.detach().numpy())
            exit()
            print("Output:{}".format(output))
            #print("shape:{}".format(output.shape))

            self.original_embeddings = output
            cost_tot = cosine_for_nodes(self.original_embeddings, self.walks, self.dim, self.V)
            epoch += 1
            print("epoch:{}".format(epoch))

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
        updated_embeddings = torch.tanh(original_embeddings.mm(self.W1)).mm(self.W2)#(34,34)x(34,34)x(34,34) = (34,34)

        #y_pred = torch.tanh(y_pred)#(34,34)

        return updated_embeddings

    def train(self, X, walks):
        
#         y_pred = self.feed_forward(original_embeddings)#(34,34)
        
#         two_dim_embeddings = multi_dim_to_two_dim(y_pred.tolist(), self.D_in, self.V)

#         vectorized_walks = node_to_vector(two_dim_embeddings, self.walks)
#         abs_for_nodes = get_abs_for_nodes(two_dim_embeddings, self.walks, vectorized_walks)#(198,)

        
#         cosine_sum = sum_cosine_of_curvature(abs_for_nodes)#(198,)
#         each_cosine_sum = each_sum_cosine_of_curvature(self.V, abs_for_nodes)#(34,1)
        
#         angles = angles_of_each_node(abs_for_nodes)#(34,) : list of radian
        updated_embeddings = self.feed_forward(X)
    
       
        print("updated_embeddings:{}".format(updated_embeddings))

        loss_tot = 0
        #cosine_function    
        for walk in walks:
            if len(walk) > 1:
                for node_num in walk[0:-1]:
                    if int(node_num) < self.V - 1:
                        cosine_loss = self.criterion(updated_embeddings[int(node_num)].reshape(1,int(self.dim)), 
                                                     updated_embeddings[int(node_num)+1].reshape(1,int(self.dim)), self.y)
                        

                        cosine_loss.backward(retain_graph=True)
                        
                        optimizer = torch.optim.Adam([self.W1, self.W2], lr=self.learning_rate)
                        opitmizer.step()
                        optimizer.zero_grad()

#                         self.W1.data -= self.learning_rate * self.W1.grad.data
#                         self.W2.data -= self.learning_rate * self.W2.grad.data

#                         self.W1.grad.data.zero_()
#                         self.W2.grad.data.zero_()
                        
                        
                        loss_tot += cosine_loss.item()#scalar value of loss
        
        loss_tot = torch.div(loss_tot,self.V)#average loss
        
        print("loss_tot:{}".format(loss_tot))
        
        print("\ypred:{}".format(updated_embeddings))
        
        return updated_embeddings

#-------------------------------------------------------------------------------#
#                              get_abs functions                                #
#-------------------------------------------------------------------------------#

#convert nodes in randomwalk to 2-dim vector
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
            walk_dict[int(q)] = list(node_embeddings[int(q)])

        vectorized_walks.append(walk_dict)

    return vectorized_walks

def cosine_for_nodes(original_embeddings, walks, dimension, V):
    cosine_val = []
    original_embeddings = torch.tensor(original_embeddings)
    
    #print(original_embeddings)
    cos = nn.CosineSimilarity(dim = 0, eps=1e-6)
    for walk in walks:
            if len(walk) > 1:
                for node_num in walk[0:-1]:
                    if int(node_num) < V-1:
#                     print(original_embeddings[int(node_num)].reshape(1,64))
#                     exit()
#                     print(original_embeddings[int(node_num)+1])
                        #print(int(node_num)+1)
                        cos_val = cos(original_embeddings[int(node_num)],
                                      original_embeddings[int(node_num)+1])
                        cosine_val.append(cos_val)
                         
    return cosine_val

def plot_embeddings(num_node, embeddings):
    tsne = TSNE(n_components=2, random_state=0)
    
    X_2d = tsne.fit_transform(embeddings)
    
    target_ids = range(num_node)

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
                    
    #blogcatalog (39 groups)
    if num_node == 10312:
        number_of_colors = 39
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(number_of_colors)]
        
        with open("dataset/group-edges.csv", 'r') as f:
            group_edges = [None]*num_node
            for line in f.readlines():
                group_edges[int(line[0])-1] = int(line[1])-1
    #cora         
    if num_node == 2708:
        number_of_colors = 7
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(number_of_colors)]
        
        labels = sp.load_npz("dataset/cora/labels.npz")
        group_edges = labels.indices

    plt.figure(figsize=(6, 5))
    
    for i in target_ids:
        plt.scatter(embeddings[i, 0], embeddings[i, 1], c=color[group_edges[i]])

    plt.show()
    plt.savefig('embedding_visualization.png')
    
    
    
#calculate angle between two vectors
# def angle(vector1, vector2, vector3):
#     v0 = np.array(vector2) - np.array(vector1)
#     v1 = np.array(vector3) - np.array(vector2)
#     angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))

#     angle_degree = np.degrees(angle)
#     #angle_radian = np.radians(angle)
#     return angle_degree

# def get_abs_for_nodes(two_dim_embeddings, walks, vectorized_walks):
#     node_embeddings = list(two_dim_embeddings)
#     #ABS curvature(turning angle for each point in randomwalk r)
#     #each item : each random_walk / in each item : {key = node_num : val = 2-dim vector}
#     total_turning_angles = []

#     for i in range(len(walks)):#iter 340
#         cur_walk = vectorized_walks[i]

#         #print(cur_walk)#{7: [0.793727189595323, 0.35655434913847506]}

#         turning_angles = {}
#         if len(cur_walk) > 2:
#             keys = cur_walk.keys()
#             for q in range(0, len(cur_walk)-2):#range(0,8)
#                 node_angle = angle(cur_walk[list(keys)[q]], cur_walk[list(keys)[q+1]], cur_walk[list(keys)[q+2]])

#                 node_number = list(keys)[q+1]
#                 turning_angles[node_number] = node_angle

#             total_turning_angles.append(turning_angles)

#     return total_turning_angles

#return summation of cosine value of absolute value of curvatures for each random walk |\sum_{p} K_{p}(P'_{i,j}^{s})|    
# def total_abs_for_each_polygonal(entire_abs):
#     absolute_sum = []
#         #for each set of abs in each random walk
#     for i in entire_abs:
#         sum_of_abs = 0
#         nodes = list(i.keys())
#         for key in nodes:
#             sum_of_abs += i[key]
        
#         sum_of_abs = np.absolute(sum_of_abs)
#         absolute_sum.append(sum_of_abs)
            
#     return absolute_sum

#-------------------------------------------------------------------------------#
#                          curvature regularization term                        #
#-------------------------------------------------------------------------------#
#list of K-q for each node q in every randomwalk
# def angles_of_each_node(total_turning_angles):

#     sum_of_angles = []
    
#     for node_num in range(self.V):#iter 34
#         angle_list = []
#         for i in total_turning_angles:#each dictionary(randomwalk)
#             for j in i.keys():
#                 if j == node_num:
#                     angle = np.radians(i[j])
#                     angle_list.extend(angle)

#         sum_of_angles.append(angle_list)

        
#     return sum_of_angles#(34,)

# #sum of cosine of curvatures in all cases(any part of P'_{ij})
# def sum_cosine_of_curvature(entire_abs):
#     absolute_sum = []
#         #for each set of abs in each random walk
#     for i in entire_abs:
#         sum_of_abs = 0
#         nodes = list(i.keys())
#         for key in nodes:
#             #sum_of_abs += math.cos(np.radians(np.absolute(i[key])))
#             sum_of_abs += math.cos(np.absolute(np.radians(i[key])))
#         absolute_sum.append(sum_of_abs)
            
#     return absolute_sum#

# #sum of cosine of curvatures in all cases(any part of P'_{ij})
# def each_sum_cosine_of_curvature(V, total_turning_angles):
#     sum_of_angles = [0] * V

#     for i in total_turning_angles:
#         turning_angles = i#each dictionary
#         for j in turning_angles.keys():
#             cosine_val = math.cos(np.absolute(np.radians(turning_angles[j])))
#             sum_of_angles[j] += cosine_val

#     return sum_of_angles#(34,1)

# def multi_dim_to_two_dim(embedding_results, dim, num_nodes):
#             # convert n-dimensional embedding to 2-dim(to satisfy Theorem 1.)
#     embedding_dim = dim
#     df = pd.DataFrame(columns = range(0,embedding_dim))

#     for i in range(num_nodes):
#         df.loc[i] = embedding_results[i]

#     #print(df)

#     #Implement PCA to reduce dimensionality of embeddings

#     #vector representation(embeddings) list
#     X = df.values.tolist()
#     #print(X)
#     #Computing correlation of matrix
#     X_corr=df.corr()

#     #Computing eigen values and eigen vectors
#     values,vectors=np.linalg.eig(X_corr)

#     #Sorting the eigen vectors coresponding to eigen values in descending order
#     arg = (-values).argsort()
#     values = vectors[arg]
#     vectors = vectors[:, arg]

#     #Taking first 2 components which explain maximum variance for projecting
#     new_vectors=vectors[:,:2]

#     #Projecting it onto new dimesion with 2 axis
#     neww_X=np.dot(X,new_vectors)
#     neww_X = neww_X.real

#     return neww_X
