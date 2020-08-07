#coding=utf-8
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
import au_calss as au
import os
file1=open("autorcode_emb.txt","w")
file2=open("gene.txt","w")

def standard_scale(X_train):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    return X_train

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]

geneSet=set()
data=[]
for lines in open(r"interactome.emb","r"):
    line=lines.strip().split()
    geneSet.add(line[0])
    file2.write(line[0])
    file2.write("\n")
    list1=[]
    for l in line[1:]:
        list1.append(float(l))
    data.append(list1)
print (geneSet)
print (len(data))

training_epochs =1000
batch_size = 256
display_step = 1
input_n_size = [512, 350]
hidden_size = [350, 100]
sdne = []

for i in range(1):
    if i== 0:
        ae = au.Autoencoder(n_input = input_n_size[0], n_hidden = hidden_size[0], transfer_function = tf.nn.elu,
                             optimizer = tf.train.AdamOptimizer(learning_rate= 0.0001),
                             scale = 0)
        
        sdne.append(ae)
    else:
        ae = au.Autoencoder(n_input = input_n_size[1], n_hidden = hidden_size[1], transfer_function = tf.nn.sigmoid,
                             optimizer = tf.train.AdagradOptimizer(learning_rate= 0.01),
                             scale = 0)
        
        sdne.append(ae)

W = []
b = []
Hidden_feature = []

for j in range(1):
    if j == 0:
        X_train = standard_scale(data)
    else:
        X_train_pre = X_train
        X_train = sdne[j-1].transform(X_train_pre)
        Hidden_feature.append(X_train)
    epoch=0
    for epoch in range(300):
        total_cost = 0.
        total_batch = int(X_train.shape[0] / batch_size)

        for k in range(total_batch):

            batch_xs = get_random_block_from_data(X_train, batch_size)

            cost = sdne[j].partial_fit(batch_xs)
            total_cost=total_cost+cost
        loss=total_cost/13460

        if epoch % display_step == 0:
            print("Epoch:", "%4d" % (epoch + 1), "每个样本上的误差:", "{:.9f}".format(loss))
            
    if j == 0:
        feat0 = sdne[0].transform(standard_scale(data))
        print (len(feat0))
        for feat in feat0:
            for f in feat:
                file1.write(str(f))
                file1.write("\t")
            file1.write("\n")
file1.close()

file2.close()