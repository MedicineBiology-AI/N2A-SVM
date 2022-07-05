This is the code of article "Predicting Parkinson's Disease Genes Based on Node2vec and Autoencoder".

**This repo is moved to [23AIBox](https://github.com/23AIBox/23AIBox-N2A-SVM)**

"interactome.edgelist" file stores the protein protein interaction network. There are two columns, and each column represents the ID of the gene.
"clinvar_result.txt" is from https://www.ncbi.nlm.nih.gov/clinvar/, which contains the known gene names related to Parkinson's disease.
"GeneID_Name" file contains the correspondence between gene names and IDs. The first column is the gene ID and the second column is the gene name.

you can run the code in the following oreder:
proccessData.py-->node2vec-master-change/main.py-->AutoencoderÖÐ/MyAE.py-->svm/svm.py

If you want to use the above code please use the citation below.
@ARTICLE{10.3389/fgene.2019.00226,
  
AUTHOR={Peng, Jiajie and Guan, Jiaojiao and Shang, Xuequn},   
	 
TITLE={Predicting Parkinson's Disease Genes Based on Node2vec and Autoencoder},      
	
JOURNAL={Frontiers in Genetics},      
	
VOLUME={10},      

PAGES={226},     
	
YEAR={2019},      
	  
URL={https://www.frontiersin.org/article/10.3389/fgene.2019.00226},       
	
DOI={10.3389/fgene.2019.00226},      
	
ISSN={1664-8021},   
   
ABSTRACT={Identifying genes associated with Parkinson's disease plays an extremely important role in the diagnosis and treatment of Parkinson's disease. In recent years, based on the guilt-by-association hypothesis, many methods have been proposed to predict disease-related genes, but few of these methods are designed or used for Parkinson's disease gene prediction. In this paper, we propose a novel prediction method for Parkinson's disease gene prediction, named N2A-SVM. N2A-SVM includes three parts: extracting features of genes based on network, reducing the dimension using deep neural network, and predicting Parkinson's disease genes using a machine learning method. The evaluation test shows that N2A-SVM performs better than existing methods. Furthermore, we evaluate the significance of each step in the N2A-SVM algorithm and the influence of the hyper-parameters on the result. In addition, we train N2A-SVM on the recent dataset and used it to predict Parkinson's disease genes. The predicted top-rank genes can be verified based on literature study.}
}


you can run the code in the following oreder:
proccessData.py-->node2vec-master-change/main.py-->AutoencoderÖÐ/MyAE.py-->svm/svm.py

