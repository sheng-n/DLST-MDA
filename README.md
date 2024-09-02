# DLST-MDA
Submit journal xx
## 1. Overview
The code for paper "Deep learning-based integration of sequence and structure information for efficient predicting miRNA-drug associations". The repository is organized as follows:

+ `data/`
  * `drug_id_smiles.xlsx` contain drug ID (from DrugBank) and SMILES;
  * `miRNA_drug_matrix.xlsx` contain known miRNA-drug association;
  * `miRNA_sequences.xlsx` contain miRNA ID (from miRBase) and sequences;
+ `code/`
  * `process_data.py` is used to 
  * `utils.py` contains preprocessing function of the data;
  * `data_preprocess.py` contains the preprocess of data and divides the dataset;
  * `cnn_gcnmulti.py` contains DLST-MDA's model layer;
  * `train.py` contains training and testing code;

## 2. Dependencies
* numpy == 1.21.1
* torch == 2.0.0+cu118
* sklearn == 0.24.1
* torch-geometric == 2.3.0

## 3. Quick Start
Here we provide a example to predict the MDA scores:

1. Download zip our data and code files
3. Run process_data.py to obtain train and test dataset 
4. Run train.py to calculate the MDA scores

## 4. Contacts
If you have any questions, please email Nan Sheng (shengnan@jlu.edu.cn)
