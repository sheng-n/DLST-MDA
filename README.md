# DLST-MDA
Submit journal IEEE Journal of Biomedical and Health Informatics
## 1. Overview
The code for paper "Deep learning-based integration of sequence and structure information for efficient predicting miRNA-drug associations". The repository is organized as follows:

+ `data/`
  * `drug_id_smiles.xlsx` contains drug ID (from DrugBank) and SMILES;
  * `miRNA_drug_matrix.xlsx` contains known miRNA-drug association;
  * `miRNA_sequences.xlsx` contains miRNA ID (from miRBase) and sequences;
+ `code/`
  * `process_data.py` is used to preprocess data and divides the dataset;
  * `utils.py` contains preprocessing function of the data;
  * `cnn_gcnmulti.py` contains DLST-MDA's model layer;
  * `train.py` contains training and testing code;

## 2. Dependencies
* numpy==1.24.3
* torch==2.3.0+cu121
* sklearn==1.3.2
* torch_geometric==2.5.3

## 3. Quick Start
Here we provide a example to predict the MDA scores:

1. Download zip our data and code files
3. Run process_data.py to obtain train and test dataset 
4. Run train.py to calculate the MDA scores

## 4. Contacts
If you have any questions, please email Nan Sheng (shengnan@jlu.edu.cn)
