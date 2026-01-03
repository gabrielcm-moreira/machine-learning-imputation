# Hybrid ML–MG Multiple Imputation Framework

This repository provides a reference implementation of the methodology described in:

Moreira, G. C., Costa, J. F. C. L., & Deutsch, C. V.  
"A hybrid approach to multiple imputation for geological and geometallurgical datasets using machine learning regression and the multivariate Gaussian distribution"  
(Natural Resources Research)

## Overview
The code implements a hybrid multiple imputation workflow that combines:
- Machine learning regression for conditional mean estimation
- Multivariate Gaussian conditional distributions with calibrated variance
- Bayesian Updating using simple kriging as a spatial prior
- Sequential simulation to generate multiple imputed realizations

All computations are performed in Gaussian units.

## Reproducibility
Scripts and a toy dataset are provided to reproduce the main results and figures presented in the manuscript.
