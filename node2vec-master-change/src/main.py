#encoding:utf-8
#edgelist：边的列表

import argparse
import networkx as nx
import node2vec
from gensim.models import Word2Vec
def parse_args():
	#node2vec中参数设置
	parser = argparse.ArgumentParser(description="Run node2vec.")
#输入interactome.edgelist
	parser.add_argument('--input', nargs='?', default=r'../graph/interactome.edgelist',
	                    help='Input graph path')
#输出interacome。emb
	parser.add_argument('--output', nargs='?', default=r'../emb/interactome.emb',
	                    help='Embeddings path')
#向量维度1*128
	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')
#步长80
	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')
#对图随机游走的次数
	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')
#窗口大小10
	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')
#迭代次数1
	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')
#训练的并行数8
	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')
#p=2.0
	parser.add_argument('--p', type=float, default=2.0,
	                    help='Return hyperparameter. Default is 10.')
#q=0.5
	parser.add_argument('--q', type=float, default=0.5,
	                    help='Inout hyperparameter. Default is 0.5.')
#无权
	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)
#无向图
	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph():
	#读输入的网络
	G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.Graph())

	for edge in G.edges():
		G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()
	return G

def learn_embeddings(walks):

	walks = [map(str, walk) for walk in walks]#把walks都变成str类型

	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.save_word2vec_format(args.output)

	return

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.

	'''

	nx_G = read_graph()
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)

    #得到所有点的概率数组和别名数组
	G.preprocess_transition_probs()

	walks = G.simulate_walks(args.num_walks, args.walk_length)

	learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)

