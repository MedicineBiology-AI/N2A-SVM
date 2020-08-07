#coding=utf-8
#先选取正负样本
from sklearn import metrics
import numpy as np
import random
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
all_gene=set()
for lines in open(r"../interactome.edgelist","r"):
    line=lines.strip().split("\t")
    all_gene.add(line[0])
    all_gene.add(line[1])
positive=set()
for lines in open(r"../parkinson_gene.txt","r"):
	line=lines.strip().split("\t")
	positive.add(line[0])


all_negtive=all_gene-positive

all_negtive=list(all_negtive)

final=[]
for i in range(20):
	random.shuffle(all_negtive)

	negtive=all_negtive[:len(positive)]



	dict_emb={}
	for lines in open("../Autoencoder/autorcode_emb1.txt"):
		line=lines.strip().split("\t")
		list1=[]
		for l in line[1:]:
			list1.append(float(l))
		dict_emb[line[0]]=list1
	

	positive_emb=[]
	for p in positive:
		positive_emb.append(dict_emb[p])


	negtive_emb=[]
	for n in negtive:
		negtive_emb.append(dict_emb[n])

	emb=positive_emb+negtive_emb
	emb=np.array(emb)

	postive_label=[]
	negtive_label=[]
	for i in range(len(positive)):
		postive_label.append(1)
		negtive_label.append(0)

	label=postive_label+negtive_label
	label=np.array(label)

	roc=[]
	skf=StratifiedKFold(n_splits=10)
	for train_index,test_index in skf.split(emb,label):
	    x_train,x_test=emb[train_index],emb[test_index]
	    y_train,y_test=label[train_index],label[test_index]
	    clf = svm.SVC(probability=True, gamma="auto")
	    clf.fit(x_train, y_train)
	    y_predict = clf.decision_function(x_test)
	    roc_score = metrics.roc_auc_score(y_test, y_predict)
	    roc.append(roc_score)
	m=np.mean(roc)
	print(m)
	final.append(m)
print(np.mean(final))










