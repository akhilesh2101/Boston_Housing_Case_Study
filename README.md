# Boston_Housing_Case_Study
A case study into what ML algorithm works best for boston housing dataset
Goal of the case study was to apply 7 different machine learning methods to the data of Boston Housing and compare the results to see which fares better in this case.

Algorithms deployed:

1) LINEAR REGRESSION: Assumption of linear relationship (turned out to be the worst model even with variable selection).
2) REGRESSION TREE: Nonlinear modeling . Better than regression model.
3) RANDOM FOREST: Reducing variance . Lead to overfitting with very low in sample MSE.
4) BOOSTING: Initial in sample MSE was 0.04. Had to reduce the number of tress for better MSPE.
5) GENERALIZED ADDITIVE MODEL: Another non-linear model . Did better than regression trees.
6) K-NEAREST NEIGHBOURS: Close to regression trees and linear regression.
7) NEURAL NETWORKS: Most complex model. Performed better as expected. 


