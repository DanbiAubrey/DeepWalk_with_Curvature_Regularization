DeepWalk_with_Curvature_Regularization
======================================

### Generate Embeddings with curvature regularization

1. karate.adjacency (nodes: 34, edges: 78)
<pre>
<code>
  python main.py --input dataset/karate.adjacency --number-walks 10 --walks-length 40 --model skipgram --output karate_deepwalk.embeddings --epoch 1 
</code>
</pre>

2. blogcatalog.mat (nodes: 10,312, edges: 333, 983)
<pre>
<code>
  python main.py --input dataset/blogcatalog.mat --number-walks 80 --walks-length 40 --model skipgram --output blogcatalog.embeddings --epoch 1 --format mat --dimension 128
</code>
</pre>  

3. cora.npz (nodes: 2708, edges: 5429)
<pre>
<code>
  python main.py --input dataset/cora/network.npz --number-walks 40 --walks-length 40 --model skipgram --output embeddings/cora.embeddings --epoch 1 --format npc --dimension 128
</code>
</pre>

### Evaluate on node classification

1. karate.adjacency (nodes: 34, edges: 78)
<pre>
<code>
  python evaluation.py --emb dataset/blogcatalog.embeddings --network example_graphs/blogcatalog.mat --num-shuffle 10 --all
</code>
</pre>

2. blogcatalog.mat (nodes: 10,312, edges: 333, 983)
<pre>
<code>
  python evaluation.py --emb embeddings/blogcatalog.embeddings --network dataset/blogcatalog.mat --num-shuffle 10 --all
</code>
</pre>

3. cora.npz (nodes: 2708, edges: 5429)
### Reference

<https://github.com/phanein/deepwalk>
