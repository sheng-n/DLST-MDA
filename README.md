# DLST-MDA
Submit journal xx
## 1. Overview
The code for paper "Deep learning-based integration of sequence and structure information for efficient predicting miRNA-drug associations". The repository is organized as follows:

+ `data/`
  * `drug_id_smiles.xlsx` contain drug ID (from DrugBank) and SMILES;
  * `miRNA_drug_matrix.xlsx` contain known miRNA-drug association;
  * `miRNA_sequences.xlsx` contain miRNA ID (from miRBase) and sequences;
+ `code/`
  * `process_data.py` is used to calculate lncRNA/miRNA k-mer features and construct knn graph (attribute graph) of lncRNA/miRNA/disease.
  * `utils.py` contains preprocessing function of the data;
  * `data_preprocess.py` contains the preprocess of data;
  * `cnn_gcnmulti.py` contains SSCLMD's model layer;
  * `train.py` contains training and testing code;

## 2. Dependencies
* numpy == 1.21.1
* torch == 2.0.0+cu118
* sklearn == 0.24.1
* torch-geometric == 2.3.0

## 3. Quick Start
Here we provide a example to predict the MDA scores:

1. Download and upzip our data and code files
2. Run data_preparation.py and calculating_similarity.py to obtain lncRNA/miRNA/disease attribute graph and intra_edge of topology graph 
3. Run main.py (in file-- dataset1/LDA.edgelist, neg_sample-- dataset1/non_LDA.edgelist, task_type--LDAl)

## 4. Contacts
If you have any questions, please email Nan Sheng (shengnan@jlu.edu.cn)
