import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse

logger = logging.getLogger("deepwalk")

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class Graph(defaultdict):
   
    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritems()

    def subgraph(self, nodes={}):
        subgraph = Graph()

        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]

        return subgraph

    def make_undirected(self):

        t0 = time()

        for v in list(self):
          for other in self[v]:
            if v != other:
              self[other].append(v)

        t1 = time()
        logger.info('make_directed: added missing edges {}s'.format(t1-t0))

        self.make_consistent()
        return self

    def make_consistent(self):
        t0 = time()
        for k in iterkeys(self):
          self[k] = list(sorted(set(self[k])))

        t1 = time()
        logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

        self.remove_self_loops()

        return self

    def remove_self_loops(self):

        removed = 0
        t0 = time()

        for x in self:
          if x in self[x]: 
            self[x].remove(x)
            removed += 1

        t1 = time()

        logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
        return self

    def check_self_loops(self):
        for x in self:
          for y in self[x]:
            if x == y:
              return True

        return False

    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:
          return True
        return False

    def degree(self, nodes=None):
        if isinstance(nodes, Iterable):
          return {v:len(self[v]) for v in nodes}
        else:
          return len(self[nodes])

    def order(self):
        "Returns the number of nodes in the graph"
        return len(self)    

    def number_of_edges(self):
        "Returns the number of nodes in the graph"
        return sum([self.degree(x) for x in self.keys()])/2

    def number_of_nodes(self):
        "Returns the number of nodes in the graph"
        return self.order()

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):

        G = self
        if start:
          path = [start]
        else:
          # Sampling is uniform w.r.t V, and not w.r.t E
          path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
          cur = path[-1]
          if len(G[cur]) > 0:
            if rand.random() >= alpha:
              path.append(rand.choice(G[cur]))
            else:
              path.append(path[0])
          else:
            break
        return [str(node) for node in path]

#-------For ABS-------#
    def random_walk_for_abs(self, path_length, alpha=0, rand=random.Random(), start=None):

        G = self
        if start:
            path = [start]
        else:
          # Sampling is uniform w.r.t V, and not w.r.t E
          path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    if rand.choice(list(G[cur].keys())) != cur:
                          path.append(rand.choice(G[cur]))
                    #exclude restart
                    else:
                          break
                else:
                    break
            else:
                break
        return [str(node) for node in path]

    # In[ ]:

def load_matfile(file_, variable_name="network", undirected=True):
  mat_varables = loadmat(file_)
  mat_matrix = mat_varables[variable_name]

  return from_numpy(mat_matrix, undirected)
                    
def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
      raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G                    

def build_deep_walk(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())
  
  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
  
  return walks
#-------For ABS-------#
def build_deep_walk_for_abs(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())
  
  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      walks.append(G.random_walk_for_abs(path_length, rand=rand, alpha=alpha, start=node))
  
  return walks
