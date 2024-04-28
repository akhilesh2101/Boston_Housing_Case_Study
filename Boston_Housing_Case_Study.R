
# Setting seed

rm(list = ls())
set.seed(14828693)


library(MASS)
data(Boston)


# Splitting into train and test pre analysis

sample_index <- sample(nrow(Boston),nrow(Boston)*0.80)
Boston_train <- Boston[sample_index,]
Boston_test <- Boston[-sample_index,]

# EDA and summary of the data

str(Boston)

summary(Boston)

hist(Boston$medv, prob = T, col = "grey", breaks = 25, main = "Density plot for medv(Median housing Value)")
lines(density(Boston$medv), col = "black", lwd = 2)




# Linear regression Model

# Full Model
Boston.full.lm <- lm(medv ~ ., data = Boston)
summary(Boston.full.lm)
n <- dim(Boston)[1]   # sample size
p <- dim(Boston)[2]-1

# step wise variable selection ###
# k=2, default AIC
Boston.AIC.step <- step(Boston.full.lm,data=Boston) 
summary(Boston.AIC.step)
AIC(Boston.AIC.step)
BIC(Boston.AIC.step)

# k=ln(n), BIC
Boston.BIC.step <- step(Boston.full.lm,data=Boston,k=log(n)) 
summary(Boston.AIC.step)
AIC(Boston.BIC.step)
BIC(Boston.BIC.step)

# Taking just the 11 predictors from step wise selection 

linear_model <- lm(medv~crim+zn+chas+nox+rm+dis+rad+tax+ptratio+black+lstat, data = Boston_train)

# Model Assessment

# In sample MSE
linear_model_summary <- summary(linear_model)
linear_model_mse <- round((linear_model_summary$sigma)^2,2)
# In sample MSE is 20.52

# Out of sample MSPE
pi <- predict(object = linear_model, newdata = Boston_test)
pi <- predict(linear_model, Boston_test)
lm_model_mspe <- round(mean((pi - Boston_test$medv)^2),2)
# Out of sample MSPE is 31.11



# Regression Tree Model

library(rpart)
library(rpart.plot)

regression_tree_model <- rpart(formula = medv ~ ., data = Boston_train)
regression_tree_model
prp(regression_tree_model,digits = 4, extra = 1)

# Pruning the regression tree

boston_largetree <- rpart(formula = medv ~ ., data = Boston_train, cp = 0.001)
prp(boston_largetree)
plotcp(boston_largetree)
printcp(boston_largetree)
regression_prune_tree_model<-prune(boston_largetree, cp = 0.017)

# Model Assessment

# In sample
boston_train_pred_tree = predict(regression_prune_tree_model)
regtree_model_mse <- round(mean((Boston_train$medv - boston_train_pred_tree)^2),2)
# In sample MSE is 16.16

# Out of sample
boston_test_pred_tree = predict(regression_prune_tree_model, Boston_test)
regtree_model_mspe <- round(mean((Boston_test$medv - boston_test_pred_tree)^2),2)
# Out of sample MSE is 25.83



# Random forest Model

library(randomForest)
Random_forest_model <- randomForest(medv~., data = Boston_train, importance = TRUE)
Random_forest_model

# plot to see how the OOB error changes with different ntree
plot(Random_forest_model$mse, type = 'l', col = 2, lwd = 2, xlab = "ntree", ylab = "OOB Error")

# Model Assessment

# In sample
Random_forest_pred_train <- predict(Random_forest_model, Boston_train)
Random_forest_model_mse <- round(mean((Boston_train$medv - Random_forest_pred_train)^2),2)
# In sample MSE is 2.26

# Out of sample
Random_forest_pred_test <- predict(Random_forest_model, Boston_test)
Random_forest_model_mspe <- round(mean((Boston_test$medv - Random_forest_pred_test)^2),2)
# Out of sample MSE is 16.98



# Boosting Model

library(gbm)
Boosting_model <- gbm(formula = medv~., 
                      data = Boston_train, 
                      distribution = "gaussian", 
                      n.trees = 10000, 
                      shrinkage = 0.01, 
                      interaction.depth = 8)
summary(Boosting_model)

# the original model seems to be over fitting, so adjusted the number of tress based on the testing error graph
ntree <- seq(100, 10000, 100)
test.err <- rep(0, 13)

predmat <- predict(Boosting_model, newdata = Boston_test, n.trees = ntree)
err <- apply((predmat-Boston_test$medv)^2, 2, mean)
plot(ntree, err, type = 'l', col=2, lwd=2, xlab = "n.trees", ylab = "Test MSE")
abline(h=min(test.err), lty=2)

# Model Assessment

# In sample
Boosting_model_pred_train <- predict(Boosting_model, Boston_train, n.trees = 1800)
Boosting_model_mse <- round(mean((Boston_train$medv - Boosting_model_pred_train)^2),2)
# In sample MSE is 1.51

# Out of sample
Boosting_model_pred_test <- predict(Boosting_model, Boston_test, n.trees = 1800)
Boosting_model_mspe <- round(mean((Boston_test$medv - Boosting_model_pred_test)^2),2)
# Out of sample MSE is 12.9



# GAM Model

library(mgcv)

#create gam model
GAM_model <- gam(medv ~ s(crim)+s(zn)+s(indus)+chas+s(nox)
                  +s(rm)+s(age)+s(dis)+rad+s(tax)+s(ptratio)
                  +s(black)+s(lstat),data=Boston_train)

summary(GAM_model)

# Model Assessment

# in-sample 
pi <- predict(GAM_model,Boston_train)
GAM_model_mse<-round(mean((pi - Boston_train$medv)^2),2)
# In sample MSE is 7.56

# Out of sample
pi.out <- predict(GAM_model,Boston_test)
GAM_model_mspe<-round(mean((pi.out - Boston_test$medv)^2),2)
# Out of sample MSE is 17.18



# Neural network model

n <- dim(Boston)[1]
p <- dim(Boston)[2] - 1

# initialize standardized training, testing, and new data frames to originals 
train.norm <- Boston_train
test.norm <- Boston_test

# normalize all numerical variables (X&Y) to 0-1 scale, range [0,1]-standardization
cols <- colnames(train.norm[, ]) #scaling both X and Y
for (j in cols) {
  train.norm[[j]] <- (train.norm[[j]] - min(Boston_train[[j]])) / (max(Boston_train[[j]]) - min(Boston_train[[j]]))
  test.norm[[j]] <- (test.norm[[j]] - min(Boston_train[[j]])) / (max(Boston_train[[j]]) - min(Boston_train[[j]]))
}


# Neural networks on (scaled) Training data and plot 

library(neuralnet)
f <- as.formula("medv ~ .")
nn <- neuralnet(f,data = train.norm, hidden = c(5,3), linear.output = T)
plot(nn)

# Model Assessment

# In sample MSE 

pr_nn <- compute(nn, train.norm[,1:p])

# changing the variables back to scale 
pr_nn_org <- pr_nn$net.result*(max(Boston_train$medv) - min(Boston_train$medv)) + min(Boston_train$medv)
train_r <- (train.norm$medv)*(max(Boston_train$medv) - min(Boston_train$medv)) + min(Boston_train$medv)
nn_mse <- round(sum((train_r - pr_nn_org)^2)/nrow(train.norm),2)
# in samples MSE is 4.95

# out of sample MSPE
pr_nn <- compute(nn, test.norm[,1:p])

# recover the predicted value back to the original response scale
pr_nn_org <- pr_nn$net.result*(max(Boston_train$medv) - min(Boston_train$medv)) + min(Boston_train$medv)
test_r <- (test.norm$medv)*(max(Boston_train$medv) - min(Boston_train$medv)) + min(Boston_train$medv)
nn_mspe <- round(sum((test_r - pr_nn_org)^2)/nrow(test.norm),2)
# Out of sample MSPE is 8.38



# K Nearest Neighbors Model

library(FNN)

Boston.knn.reg <- knn.reg(train = train.norm[, 1:p], 
                          test = test.norm[, 1:p], 
                          y = train.norm$medv, 
                          k = 5)

# compile the actual and predicted values and view the first 20 records
Boston.results <- data.frame(cbind(pred = Boston.knn.reg$pred, actual = Boston_test$medv))

# Out-of-Sample Testing
MSPE <- sum((Boston_test$medv-Boston.results$pred)^2)/length(Boston_test$medv)
RMSE <- sqrt(MSPE)
# Out of sample error = 25.63

# initialize a data frame with two columns: k and accuracy
RMSE.df <- data.frame(k = seq(1, 30, 1), RMSE.k = rep(0, 30))

# compute knn for different k on validation set
for (i in 1:30) {
  knn.reg.pred <- knn.reg(train = train.norm[, c(1:p)], test = test.norm[, c(1:p)], 
                          y = train.norm$medv, k = i)
  RMSE.df[i, 2] <- sqrt(sum((test.norm$medv-knn.reg.pred$pred)^2)/length(test.norm$medv))
}
RMSE.df
k <- which(RMSE.df[,2] == min(RMSE.df[,2]))
k

# Hence, choose "optimal" k=2

# Model Comparison through MSPE

Boston.knn.reg <- knn.reg(train = train.norm[, c(1:p)], 
                          test = test.norm[, c(1:p)], 
                          y = train.norm$medv, 
                          k = k)

# Out-of-Sample Testing
MSPE <- sum((Boston_test$medv-Boston.knn.reg$pred)^2)/length(Boston_test$medv)
Knn_RMSE <- sqrt(MSPE)
Knn_RMSE
# Out of sample error = 25.64

# comparison of results between the models
results_comparison <- data.frame(
  Method = c("Linear Regression", "K-Nearest Neighbors","Regression Tree", "Random Forrest", "Boosting Model", "Generalized Additive Model","Neural Networks"),
  mse = c(linear_model_mse,RMSE, regtree_model_mse, Random_forest_model_mse, Boosting_model_mse, GAM_model_mse,nn_mse),
  mspe = c(lm_model_mspe,Knn_RMSE, regtree_model_mspe, Random_forest_model_mspe, Boosting_model_mspe, GAM_model_mspe,nn_mspe),
  stringsAsFactors = FALSE
)

# Print the data frame
results_comparison
