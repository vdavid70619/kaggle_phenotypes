kaggle_phenotypes
=================

Xiyang Dai @UMD CS

Predict phenotypes from genotype.
--------------------------------------



This is for Kaggle competition for predicting phenotypes from genotype @UMD CMSC702. The whole codes are provided AS-IS. 

## Requirements
	-Matlab
	-Libsvm/Libliner (for SVR)
	-DeepLearnToolbox (for NN regression)
	-vl_feat (for gmm)
	
## Proposed Approaches
### Unities
	- logloss.m
	- dataloader.m
	- confusion_matrix.m
	- mynormalize.m
	- k_mer_feature.m
	- gmms_feature.m
	
### Feature Selection 
	- feature_selection.m
	- fs_trian_test.m

### Dimension Reduction
	- dimension_reduct.m
	
### K-mer + Population + LASSO + NN/SVR
	- nfold_validation.m: Driver to run nfold cross validation on proporsed methods
	(SVR)
	- mytrain.m
	- mytest.m
	- mypredict.m
	(NN)
	- mynntrain.m
	- mynntest.m
	- mynnpredict.m
	
	- generate_kaggle_result

### K-mer + Population + LASSO + Boosting/Bag
	- boosting.m
	- generate_kaggle_boost_result.m
	