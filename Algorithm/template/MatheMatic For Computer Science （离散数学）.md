# Tree

**Leaf**

**Def.**  A leaf is a node with one degree in a tree.

**Lemma.** Any connected subgraph of the tree is a tree.

一个树的任意连通子图一定是树。

![image-20240407190527422](C:\Users\TsingPig\AppData\Roaming\Typora\typora-user-images\image-20240407190527422.png)

**Proof**. We can essentially use contradiction , let's suppose the connected subgraph is not a tree,  a tree is Acyclic（无环的）, so the the subgraph actually has a circle. But not since it's a subgraph of the bigger graph, so the whole graph must have cycle. But the whole graph is a tree which is Acyclic, so we get a contradiction. So the conntected subgraph must be a tree. +

矛盾法证明树的连通子图必然是树：假设子图不是树，其中必然存在环，与原图是树矛盾。



**Lemma.** If you have a tree that has n vertice, then it must have n minus 1 edges.

**Proof.** We can use Induction (归纳法）. And we start out with an Induction Hypothesis（归纳假设）.

$P(n) :$ there are n minus 1 edges in any n vertices tree. 

$BaseCase:$ if we consider $P(1)$ , so there are zero edge in a 1 vertex tree. 

Let's prove the other part , which is the inductive step. We are going to always assume $P(n) $, and then we want to prove $P(n + 1)$. We take a tree $T$ with $n + 1$ vertices , we want to show that it has $n$ edges.  Let $V$ be a leaf is a tree, (a leaf is the type of vertex which can be delete) , then romove this particular vertex $V$ which creates a connected subgraph which is still a tree, called $T'$ with $n$ vertices. By $P(n)$ , we know $T' $ is a tree with $n - 1$ edges. Then reattached $V$ to the original tree, we know $T$ has $n$ edges, proving $P(n + 1)$ .

数学归纳法证明 n 个节点 的树有 n - 1条边。假设 $P(n)$ ，考虑$P(1)$ 成立；尝试推出 $P(n + 1)$ 。通过对一颗有 n + 1节点的树删去一个叶子得到有n 个节点 n - 1条边（假设$P(n)$）证明 $P(n +  1)$。



**Spanning Tree** （生成树）

**Def**. A spanning tree of a connected graph is actually a subgraph that is a tree with the same vertices as the graph.

连通图的生成树是一个和它有相同点集的子图。

**Theorem.** Every conntected graph has a spanning tree.

**Proof.** By contradiction,  we assume there is a connected graph $G$ has not spannint tree.  Let's $T$ be a connected subgraph of $G$ with the same vertices as $G$ and has a minimum number of edges possible. By contradiction we know $T$ can definitely not a spannint tree. The only difference between the $T$ and the spanning tree is that, $T$ is not a tree. (Well both of them have the same vertices as the $G$). So therefor $T$ must have a cycle. And if we delete one edge of the cycle we are able to construct a smaller subgraph $T$ with a smaller number of edges.

证明每一个连通图都有生成树。假设存在一个没有生成树的连通图$G$ ,  等价于一定有 $T$ 是和 $G$ 有相同顶点，且边数最少的连通图，且$T$不是树。这意味着其存在环，当其去掉环上一条边，得到一个边数更少的连通图（连通性可以针对任意两对节点分情况讨论），那么只要它不是树，一定能够在减少环中边后得到更小的连通图，最终一定会变成一个树。