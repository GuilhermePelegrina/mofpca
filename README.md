# A novel multi-objective-based approach to analyze trade-offs in Fair Principal Component Analysis

### Authors: *Guilherme Dean Pelegrina*, *Renan Del Buono Brotto*, *Leonardo Tomazeli Duarte*, *Romis Attux*, *João Marcos Travassos Romano*. 

## Introduction

This work verifies the compromise between the (total) reconstruction error and the fairnes measure in a dimensional reduction problem.
Fairness measure is given by the difference between the reconstruction errors of the two considered classes. We use a multi-objective approach (SPEA2 algorithm) and select a single non-dominated solution based on the minimum weighted sum (with equal importance, but taking the scales of each objective into account). We consider the Default Credit dataset - see [Yeh, I. C., & Lien, C. H. (2009). Expert Systems with Applications, 36(2), p. 2473-2480] - and the Labeled Faces in the Wild (LFW) - see [Huang, G. B., Mattar, M., Berg, T., & Learned-Miller, E. (2008). Labeled faces in the wild: A database for studying face recognition in unconstrained environments. In Workshop on Faces in 'Real-Life' Images: Detection, Alignment, % and Recognition. Marseille, France].

The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients and some functions were based on [Samadi et al. (2018). The price of fair pca: One extra dimension. In Advances in Neural Information Processing Systems, p. 10976-10987].

To cite this work: Pelegrina, G. D.; Brotto, R. D. B.; Duarte, L. T.; Attux, R. & Romano, J. M. T. (2021). A novel multi-objective-based approach to analyze trade-offs in Fair Principal Component Analysis. ArXiv preprint, arXiv:2006.06137. Available at: https://arxiv.org/abs/2006.06137

## Execution Steps

All the files in this repository are in .m format, so it is necessary to execute them in a Matlab (v. 2015a) or Octave Environment

1) Clone the repository 
2) Execute the file "main_moofpca_spea2WS_credit.m" (modify the Load Data, if necessary)
3) The dataset used is in the folder data/credit and data/images

## Datasets

All data files can be downloaded at https://drive.google.com/drive/folders/1ltUvPAj5rrBZO_pOl4N0QHXD_JdFmsE7?usp=sharing
