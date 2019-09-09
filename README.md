# BlurMore
## Set-up:
You must have the MovieLens data downloaded (https://grouplens.org/datasets/movielens/) and in your project.

## Gender Inference
The basic gender inference attack can be executed by running the GenderClassification.py. In the main function, you can define what exactly should be executed. I.e., 
* one_million(Classifiers.log_reg) runs a logistic regression model on the MovieLens 1M dataset. 
* If we alter that line to one_million(Classifiers.svm_classifier), it would use a SVM instead of a logistic regression model. You can use any classifier from Clssifiers.py.

## Obfuscate Dataset
In the Obfuscation.py you can find the obfuscation algorithms: BlurMe (named blurMe_1m) and BlurM(or)e (named blurMePP). These algorithms load the MovieLens data and obfuscate it according to the specified parameters at the top of each algorithm. The obfuscated datasets are stored in the project directory in the folder "ml-1m/".

## Gender Inference on Obfuscated Datasets
To see how the inference attack performs on the obfuscated dataset, we have to replace the "one_million(Classifiers.log_reg)" line in GenderClassification.py with "one_million_obfuscated". This function trains a logistic regression model on the original MovieLens 1M data and tests the model on the obfuscated dataset. Note that it is very important to check if the function loads the correct obfuscated dataset: In line "X2 = MD.load_user_item_matrix_1m_masked(file_index=55)", we have to specify a file index. This corresponds to the index of the file array from the function "load_user_item_matrix_1m_masked" in MovieLensData.py. 

### Final Note: 
The BlurM(or)e algorithm was initially named "BlurMe++", but we decised to change that afterwards. This is the reason why some functions or files include "BlurMePP" in their naming. 
