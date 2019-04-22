#utilizing required packages
library(pROC)
library(ROCR)
library(emmeans)
library(survivalROC)
library(e1071)
library(boot)
library(nnet)
library(MASS)
library(caret)
require(caTools)
require(ISLR)
library(SDSRegressionR)
library(rms)
library(car)
library(tidyr)
library(ggplot2)
library(psych)
library(car)

#dataset
nba <- read.csv("nba_logreg.csv")

#factoring outcome
nba <- nba %>%
  mutate(TARGET_5Yrs = factor(TARGET_5Yrs, levels=c(0, 1)))

#attaching dataset
attach(nba)

#creating new dataset to display histogram of predictors
newnba <- data.frame(nba$GP, nba$MIN, nba$PTS, nba$REB, nba$AST, nba$STL, nba$BLK, nba$TOV)
names(newnba) <- c("GP", "MIN", "PTS", "REB","AST", "STL", "BLK", "TOV")

#correlation matrix among predictors
pairs(newnba)
  
#creating a histogram of all predictors
dev.off()
newnba %>%
  keep(is.numeric) %>%                     
  gather() %>%                             
  ggplot(aes(value)) +                     
  facet_wrap(~ key, scales = "free") +   
  geom_density()                         

#quantity of outcome
table(TARGET_5Yrs)

#GLM equation
glm.fit<-glm(TARGET_5Yrs ~ GP+MIN+PTS+REB+AST+STL+BLK+TOV, data=nba, family = binomial)
summary(glm.fit)
pseudo <- lrm(TARGET_5Yrs ~ GP+MIN+PTS+REB+AST+STL+BLK+TOV)
pseudo #pseudo r2 - 0.251


#vif
vif(glm.fit)

x2 <- deviance(multinom(TARGET_5Yrs~1, data=nba)) - deviance(glm.fit) #part before "-" sign is -2 log likelihood for null model
deviance(multinom(TARGET_5Yrs~1, data=nba))
x2
pchisq(x2, 6, lower.tail=FALSE)

#makes predictions on training set used to fit the model
glm.probs=predict(glm.fit,type="response") 

#creating a classifier
glm.pred=ifelse(glm.probs>0.5,"1","0")

#confusion matrix of entire dataset - no training/testing
table(glm.pred,TARGET_5Yrs)
mean(glm.pred==TARGET_5Yrs) #70.22% accuracy

#make training and test set
train <- 1:750 
test <- !(1:nrow(nba) %in% train)

#using a training set to hopefully improve the model
glm.fit.train=glm(TARGET_5Yrs~GP+MIN+PTS+REB+AST+STL+BLK+TOV, data=nba, family = binomial, subset=train)
glm.probs.train=predict(glm.fit.train,newdata=nba[test,],type="response") 
glm.pred.train=ifelse(glm.probs.train >0.5,"1","0")

#testing error rate
lol=nba$TARGET_5Yrs[test]
table(glm.pred.train,lol)
mean(glm.pred.train==lol) #68.31% accuracy

# define training control
train_control<- trainControl(method="cv", number=5)

# train the model 
model<- train(TARGET_5Yrs ~ GP+MIN+PTS+REB+AST+STL+BLK+TOV, data=nba, trControl=train_control, method="glm", family=binomial())

# print cv scores
print(model)

#ROC
pred1 <- prediction(predict(glm.fit), TARGET_5Yrs)
perf1 <- performance(pred1,"tpr","fpr")
plot(perf1, main = "ROC")

#AUC
auc.tmp <- performance(pred1,"auc")
auc <- as.numeric(auc.tmp@y.values)
auc
