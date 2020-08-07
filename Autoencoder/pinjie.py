import numpy as np
gene=[]
for lines in open("gene.txt"):
	line=lines.strip().split("\t")

	gene.append([line[0]])
gene=np.array(gene)
print(gene.shape)

emb=np.loadtxt("autorcode_emb.txt")
print(emb.shape)
# emb1=np.hstack((gene,emb))
emb1=np.concatenate((gene,emb),axis=1)
print(emb1)

# np.savetxt("autorcode_emb1.txt",emb1,fmt="%s",delimiter="\t")
file1=open("autorcode_emb1.txt","w")
for e in emb1:
	for i in e:
		file1.write(i+"\t")
	file1.write("\n")
file1.close()

