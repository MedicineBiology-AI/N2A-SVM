#coding=utf-8
"""
clinvar_result.txt 存储的是与帕金森相关的基因
interactome.edgelist存储的是PPI网络，前两列是基因，每一行代表PPI网络中有关系的基因
GeneID_Name.txt存储 基因的ID和name的对应关系
处理数据的思路如下：
找到在PPI网络中有多少帕金森基因。
1.找到与已知的帕金森基因名字对应的ID
2.PPI的基因ID和帕金森的基因ID求交集，得到在PPI网络中的帕金森基因ID
"""

file1=open("clinvar_result.txt","r")
next(file1)
geneSet=set()
for lines in file1:
    line=lines.strip().split("\t")
    l=line[1].split("|")
    for m in l:
        geneSet.add(m)
        # print(m)
print(len(geneSet))

all_gene=set()
for lines in open(r"interactome.edgelist","r"):
    line=lines.strip().split("\t")
    all_gene.add(line[0])
    all_gene.add(line[1])
print(len(all_gene))

dict_name_ID={}
for lines in open(r"GeneID_Name.txt","r"):
    line=lines.strip().split("\t")
    dict_name_ID[line[1]]=line[0]
print(len(dict_name_ID))

geneID=set()
for gene in geneSet:
    if gene in dict_name_ID.keys():
        geneID.add(dict_name_ID[gene])
print(len(geneID))
Parkinson_gene_in_Network= geneID & all_gene

print(len(Parkinson_gene_in_Network))

file2=open("parkinson_gene.txt","w")
for i in Parkinson_gene_in_Network:
    file2.write(i+"\n")
file2.close()







