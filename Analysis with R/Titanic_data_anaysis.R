
#loading the libraries
library(glm2)
library(dplyr)
library(tidyr)
library(caret)
library(rpart)
library(rpart.plot)
library(pROC)

set.seed(132)
options(scipen=999, repr.plot.width = 7, repr.plot.height = 6)
add_factors <- function(df){
  df$Embarked <- factor(df$Embarked)
  df$Survived <- factor(df$Survived, levels = c(0, 1), labels = c("Did Not Survive", "Survived"))
  df$Sex <- factor(df$Sex)
  df$Pclass <- factor(df$Pclass)
  return(df)
}
#removes all rows with missing values
train2_cln <- drop_na(train2)
#turns all of the relevant variables containing strings into categorical variables
train2_cln <- add_factors(train2_cln)
test$Pclass <- as.factor(test$Pclass)
test$Sex <- as.factor(test$Sex)
test$Embarked <- factor(test$Embarked)
test <- mutate(test, Age = ifelse(is.na(Age), mean(drop_na(test)$Age), Age))
#splitting data set into train and verification/test data set
train_df <- train2_cln %>% sample_frac(0.8)
test_df <- train2_cln %>% anti_join(train_df)
#creating a logistic regression model with all the relevant explanatory variables in the data frame
full_model_log <- glm(Survived ~ ., family = "binomial", data=train_df)
summary(full_model_log)
#checking to see the importance of all our variables
varImp(full_model_log)
#Test whether Embarked, Fare and Parch are useful
no_parch_embark_fare <- glm2(Survived ~ . - Parch - Embarked - Fare, family = "binomial", data=train_df)
summary(no_parch_embark_fare)
anova(no_parch_embark_fare, full_model_log)
#Likelihood ratio test (p-value not shown in anova)
test_stat <- -2 * (logLik(no_parch_embark_fare)-logLik(full_model_log))
deg_freedom <- attr(test_stat, "df")
p_val <- pchisq(test_stat[1], deg_freedom, lower.tail = F)
p_val
#Plotting ROC curves
pred_full_log <- predict(full_model_log, test_df, type = "response")
pred_rest_log <- predict(no_parch_embark_fare, test_df, type = "response")
roc_df <- data.frame(pred_full_prob = pred_full_log, survived = test_df$Survived, pred_rest_prob = pred_rest_log)
plot.roc(test_df$Survived, pred_full_log, percent = T, col = "brown", main = "ROC curves for full  and restricted model")
lines.roc(test_df$Survived, pred_rest_log, percent = T, col = "blue")
#getting ROC details (and AUC) from full and restricted model respectively
full_roc <- roc(survived ~ pred_full_prob, data = roc_df)
rest_roc <- roc(survived ~ pred_rest_prob, data = roc_df)
#predicting survival using the best threshold from restricted model
best_threshold_rest <- as.numeric(coords(rest_roc, "best", ret="threshold"))
Survived <- unname(predict(no_parch_embark_fare, test, type = "response"))
Survived <- ifelse((Survived >= best_threshold_rest), 1, 0)
prediction_df <- as.data.frame(cbind(test$PassengerId, Survived), row.names = F)
prediction_df <- rename(prediction_df, PassengerId = V1)
#plotting histogram and box plot of the prediction probabilities of full model
ggplot(data = data.frame(pred = pred_full_log))+geom_histogram(aes(x=pred), col = "blue") +
  labs(title = "histogram of prediction probabilities", xlab= "pred probabilities")
ggplot(data = data.frame(pred = pred_full_log))+geom_boxplot(aes(y=pred), col = "orange") +
  labs(title = "boxplot of prediction probabilities", xlab= "pred probabilities")
#plotting histogram and box plot of the prediction probabilities of restricted model 
#( full model without Embarked, Fare and Parch)
ggplot(data = data.frame(pred = pred_rest_log))+geom_histogram(aes(x=pred), col = "blue") +
  labs(title = "histogram of prediction probabilities", xlab= "pred probabilities")
ggplot(data = data.frame(pred = pred_rest_log))+geom_boxplot(aes(y=pred), col = "orange") +
  labs(title = "boxplot of prediction probabilities", xlab= "pred probabilities")


#running the classification tree
full_model_tree <- rpart(Survived ~ ., data = train_df)
prp(full_model_tree, space = 4, split.cex = 2, nn.border.col=0)
#prediction using the classification tree model
pred_tree <- predict(full_model_tree, test_df, type = "class")
#computing the confusion matrix
table(test_df$Survived, pred_tree)
