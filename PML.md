# Practical Machine Learning Course Project
ycdu  
Sunday, November 16, 2014  
## Introduction

This report describes the modelling process for predicting the performance of weight lifting exercise based on the data from accelerometers on the belt, forearm, arm and dumbell of 6 participants. 

The training and testing data are first loaded and cleaned. The training data set is then partitioned into datasets for training and cross-validation. The training data set is used to train the model. The out-of-sample error rate is estimated and used to evaluate the prediction model. Random Forest is chosen as the method of modelling because it is in general robust for classification problems like this one. This decision has been arrived at after many trials on other modelling methods which are not detailed here due to word limit. Random Forest is found to be the best model with the lowest out-of-sample error rate. 

The prediction model has resulted in an estimated out-of-sample error rate of 0.2%. The results submitted for the 20 test cases are also 100% correct. 

## Install and load required packages

```r
packages <- c("caret")
sapply(packages, require, character.only = TRUE, quietly = TRUE)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## caret 
##  TRUE
```

## Load the data

```r
# Download the data files
if (!file.exists("training.csv")){
  fileURL<-"http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(fileURL,destfile="training.csv")
}

if (!file.exists("test.csv")){
  fileURL<-"http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
  download.file(fileURL,destfile="test.csv")
}

# Load the raw train and test data. This will take a while.
rawtrain <- read.csv("training.csv")

rawtest <- read.csv("test.csv")
```

## Get a summary view of the data


```r
summary(rawtrain)
summary(rawtest)
```
The results of the R code is hidden here because it involves 160 variables and is very lengthy to display. You can run it yourself if you want to view the result.

From the results, we can see the followings:

a) Quite a number of variables have large number of missing values in the form of "NA" and #DIV/0!". We need to remove them from our models since data imputation for them will heavily distort the data. 

b) The first column "X" is just a list of ordered series numbers and is meaningless if we include it in the model. Hence, we need to remove it from the model.

c) Same for the second to fifth column. There are 5 users and the user names should have no impact on Classe. The time stamps should also not affect Classe. Hence, we need to remove them from the model.

## Clean the data 


```r
# Reload the data so that all the missing values are marked as NAs
cleantrain <- read.csv("training.csv", na.strings = c("NA","#DIV/0!",""))

cleantest <- read.csv("test.csv", na.strings = c("NA","#DIV/0!",""))

# Remove the columns with NAs
sumNA <- apply(cleantrain, 2, function(x) {
    sum(is.na(x))
})

cleantrain <- cleantrain[, which(sumNA == 0)]
cleantest <- cleantest[, which(sumNA == 0)]

# Remove the first five columns since they are meaningless to this modelling
cleantrain <- cleantrain[,6:ncol(cleantrain)]
cleantest <- cleantest[,6:ncol(cleantest)]
```

## Partition the cleaned training data set for training (70%) and cross-validation (30%) 


```r
# Set the seed so that the study is reproducible
set.seed (1233)

# Do partitioning with 70% in training and 30% in cross-validation
inTrain <- createDataPartition(cleantrain$classe, p = 0.7, list = FALSE)
train <- cleantrain[inTrain, ]
crossvalidation <- cleantrain[-inTrain, ]
```

## Train the data using Random Forest

Random Forest is adopted because this is a classification problem and Random Forest is a robust modelling tool for classification. 


```r
tControl = trainControl(method = "cv", number = 4)
modelFit <- train(train$classe ~ ., data = train, method = "rf", trControl = tControl)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
modelFit
```

```
## Random Forest 
## 
## 13737 samples
##    54 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 10302, 10302, 10303, 10304 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa   Accuracy SD  Kappa SD
##    2    0.9926    0.9907  0.002541     0.003215
##   28    0.9962    0.9952  0.001646     0.002083
##   54    0.9942    0.9926  0.001236     0.001564
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 28.
```
## Evaluate in-sample error rate


```r
# Predict for the training data set
predTrain <- predict(modelFit, train)

# Evaluate the model
confusionMatrix(predTrain,train$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

From the Confusion Matrix above, it can be seen that the accuracy is 1. Given the P-value is smaller than 5%, the accuracy is significant at 95% confidence interval.

Hence the in-sample error rate is 0. 

## Predict for the cross-validation data set and calculate the out-of-sample error rate


```r
# Predict for the cross-validation data set
predCross <- predict(modelFit, crossvalidation)

# Evaluate the model
confusionMatrix(predCross,crossvalidation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    2    0    0    0
##          B    0 1136    2    0    0
##          C    0    1 1024    2    0
##          D    0    0    0  962    2
##          E    0    0    0    0 1080
## 
## Overall Statistics
##                                         
##                Accuracy : 0.998         
##                  95% CI : (0.997, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.998         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.997    0.998    0.998    0.998
## Specificity             1.000    1.000    0.999    1.000    1.000
## Pos Pred Value          0.999    0.998    0.997    0.998    1.000
## Neg Pred Value          1.000    0.999    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.163    0.184
## Detection Prevalence    0.285    0.193    0.175    0.164    0.184
## Balanced Accuracy       1.000    0.998    0.999    0.999    0.999
```

From the Confusion Matrix above, it can be seen that the accuracy is 99.8%. Given the P-value is smaller than 5%, the accuracy is significant at 95% confidence interval. 

**The out-of-sample error rate is 0.2%**

The prediction model works very well on the cross-validation data set given the above accuracy, sensitivity, specificity etc. 

## Apply the model to the 20 different test cases for project submission


```r
# Predict result
result <- predict(modelFit, cleantest)

# Write the result to the files for submission
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(result)
```

## Result

The result submitted shows 100% correct (20/20). This further shows that the prediction model is robust. 
