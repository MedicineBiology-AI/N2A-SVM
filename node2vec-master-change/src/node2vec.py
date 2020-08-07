#encoding:utf-8
import numpy as np
import random

class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed#有向图
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):#步长，起始点
		'''
		从一个开始节点，模拟随机游走
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges
        #walk里存的是随机游走的序列。startnode是起始点。
		walk = [start_node]
        #walk_length 是随机游走的步长
		while len(walk) < walk_length:
			#列表中的-1位置是，列表的最后一个数。
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					#alias_nodes[cur][0]是概率数组
					#alias_nodes[cur][1]是别名数组
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break
		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		从每一个节点重复随机游走
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())

        #num_walks代表对于整个图随机游走的次数
		print ('Walk iteration:')
		for walk_iter in range(num_walks):
			print (str(walk_iter+1), '/', str(num_walks))

			# shuffle() 方法将序列的所有元素随机排序。
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))


		return walks

	def get_alias_edge(self, src, dst):
		#对每个邻居节点求概率。
		G = self.G
		p = self.p
		q = self.q
		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			# print (dst_nbr)
			if dst_nbr == src:
				unnormalized_probs.append(1 / p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(1)
			else:
				unnormalized_probs.append(1/ q)
		# print (unnormalized_probs)#unnormalized_probs相当于πvx
		norm_const = sum(unnormalized_probs)
		# print ("norm_const")
		# print (norm_const)#norm_const正常的常数，相当于Z
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		# print (normalized_probs)#相当于概率

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		# print ("首先对随机游走的概率进行处理，分为对于起始节点和非起始节点")

		G = self.G

		alias_nodes = {}

		for node in G.nodes():
			#对于起始节点，采用完全随机游走的方法来确定下一个节点
			#
			unnormalized_probs = [1 for nbr in sorted(G.neighbors(node))]

			norm_const = sum(unnormalized_probs)

			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]#全部都是一样的

			alias_nodes[node] = alias_setup(normalized_probs)
		#alias_nodes是一个字典，里面的键是，节点，值是由两个数组组成的元祖，第一个数组代表的是概率数组，第二个数组代表的是别名数组
        #{1: (array([0, 0, 0, 0]), array([ 1.,  1.,  1.,  1.])), 2: (array([0, 0, 0, 0]), array([ 1.,  1.,  1.,  1.])),

        #对于非起始节点
		alias_edges = {}

		for edge in G.edges():
			#边的数组类型是元祖
			#前一个节点是edge[0]，当前节点是edge[1]时，如何选择邻居节点
			#alias_edges是一个字典，字典里的键是一个元祖，代表边，字典里的值是一个元祖代表的是两个数组，概率数组和别名数组。
            #{(1, 2): (array([3, 0, 0, 0]), array([ 0.,  1.,  1.,  1.])), (5, 6): (array([0, 0]), array([ 1.,  0.])),

			alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
			alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return

def alias_setup(probs):
	#根据概率，计算得到概率数组和别名数组。
	K = len(probs)
	#概率数组
	q = np.zeros(K)
	#别名数组
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	# 构建方法：
	# 1.找出其中面积小于等于1的列，如i列，这些列说明其一定要被别的事件矩形填上，所以在Prab[i]中填上其面积
	# 2.然后从面积大于1的列中，选出一个，比如j列，用它将第i列填满，然后Alias[i] = j，第j列面积减去填充用掉的面积。
	for kk, prob in enumerate(probs):

		q[kk] = K*prob

		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()
		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)
	return J, q

def alias_draw(J, q):
	'''
	使用别名抽样从非均匀分布的离散分布中抽取样本。
	J,q都是字典类型
	'''
	#产生两个随机数，第一个产生1~N 之间的整数i，决定落在哪一列。
	# 扔第二次骰子，0~1之间的任意数，判断其与Prab[i]大小，如果小于Prab[i]，则采样i，如果大于Prab[i]，则采样Alias[i]
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:#q相当于prab[i]
		return kk
	else:
		return J[kk]#相当于alias[i]
