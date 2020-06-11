# A multi-objective based approach for Fair Principal Component Analysis

### Authors: *Guilherme Dean Pelegrina*, *Renan Del Buono Brotto*, *Leonardo Tomazeli Duarte*, *Romis Attux*, *João Marcos Travassos Romano*. 

## Introduction

This work verifies the compromise between the (total) reconstruction error and the fairnes measure in a dimensional reduction problem
Fairness measure is given by the difference between the reconstruction errors of the two considered classes. 
We use a multi-objective approach (SPEA2 algorithm) and select a single non-dominated solution based on
the minimum weighted sum (with equal importance, but taking the scales of each objective into account).
We consider the Default Credit dataset - see: [Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining
techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with
Applications, 36(2), p. 2473-2480] - and some functions were based on [Samadi et al. (2018). The price of fair pca:
One extra dimension. In Advances in Neural Information Processing Systems, p. 10976-10987].

## Execution Steps

All the files in this repository are in .m format, so it is necessary to execute them in a Matlab (v. 2015a) or Octave Environment

1) Clone the repository 
2) Execute the file "main_moofpca_spea2WS_credit.m"
3) The dataset used is in the folder data/adult
