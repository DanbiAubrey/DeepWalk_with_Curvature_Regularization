import sys
import scipy.sparse as sp
import networkx as nx

import random
from random import shuffle

class Graph:
    def __init__(self, graph_file):
        g_npz = sp.load_npz(graph_file)
        self.G = nx.from_scipy_sparse_matrix(g_npz)
        self.num_of_nodes = self.G.number_of_nodes()
        self.num_of_edges = self.G.number_of_edges()
        self.edges = self.G.edges(data=True)
        self.nodes = self.G.nodes(data=True)


    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        g = self.G
        
        if start:
            path = [start]
        else:
            path = [rand.choice(list(g.nodes(data=False)))]#uniform in regard to Nodes while not uniform with edges
        
        while len(path) < path_length:
            current = path[-1]# current node(end node)
            if len(g[current]) > 0:# if there is neighbor of current node
                if rand.random() >= alpha: # if probability of restart is less than random probability
                    path.append(rand.choice(list(g[current].keys())))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]
        
    # In[ ]:

    #build random walks list(shuffle nodes in beforehand)
    def build_deep_walk(self, num_paths, path_length, alpha=0, rand=random.Random(0)):
        g = self.G
        
        walks = []

        #print(g)
        nodes = list(g.nodes)

        for cnt in range(num_paths):
            rand.shuffle(nodes)#shuffle #to speed up the convergence
            for node in nodes:
                walks.append(self.random_walk(path_length, alpha=alpha, rand=rand, start=node))

        return walks

    def random_walk_for_abs(self, path_length, alpha=0, rand=random.Random(), start=None):
        g = self.G
        
        if start:
            path = [start]
        else:
            path = [rand.choice(list(g.nodes(data=False)))]#uniform in regard to Nodes while not uniform with edges
        
        while len(path) < path_length:
            current = path[-1]# current node(end node)
            if len(g[current]) > 0:# if there is neighbor of current node
                if rand.random() >= alpha: 
                    if rand.choice(list(g[current].keys())) != current:
                        path.append(rand.choice(list(g[current].keys())))
                    else:
                        break
                else:
                    break
            else:
                break
        return [str(node) for node in path]
    
    def build_deep_walk_for_abs(self, num_paths, path_length, alpha=0, rand=random.Random(0)):
        g = self.G
        
        walks = []

        #print(g)
        nodes = list(g.nodes)

        for cnt in range(num_paths):
            rand.shuffle(nodes)#shuffle #to speed up the convergence
            for node in nodes:
                walks.append(self.random_walk_for_abs(path_length, alpha=alpha, rand=rand, start=node))

        return walks
