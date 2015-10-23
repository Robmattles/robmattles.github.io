Quality of Barbell Lifts
========================================================

### Introduction

In this paper, I will use data from wearable motion sensors worn while performing barbell lifts to create a machine learning algorithm that can predict which of five different methods was used for the lift. The dataset used has about 20,000 observations containing 160 different variables, the vast majority of which are various measurements from the motion sensors.  From this data, I am able to generate a Random Forest model which predicts with greater than 95% accuracy and is expected to attain lower than 5% out of sample error rate.

### Exploratory Analysis

I start by downloading and loading the data and quickly examining the dimensions of the dataset.  I then look at what types of data are contained in those columns - most of the data is numerical.




```r
training <- read.csv("trainingproject", na.strings = c("", "NA", "#DIV/0!"))
invisible(library(caret))
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Package SparseM (0.99) loaded.  To cite, see citation("SparseM")
```

```r
nrow(training)
```

```
## [1] 19622
```

```r
ncol(training)
```

```
## [1] 160
```

```r
table(sapply(training, function(x) typeof(x)))
```

```
## 
##  double integer logical 
##     115      39       6
```


One important characteristic of the data is the number of NA entries.  I use a histogram to examine the distribution of those NA values by column.  As the below figure indicates, most of the variables are missing 90% or more of their values.  Some are missing all values.  This will be an important consideration later when I decide how to preprocess the data.


```r
qplot(colSums(is.na(training))/nrow(training), geom = "histogram", xlab = "Proportion of Column that is NA", 
    main = "Histogram of Number of Missing Values in Each Column", binwidth = 0.05)
```

![plot of chunk unnamed-chunk-2](figure/unnamed-chunk-2.png) 


I then focus on the classe variable, which indicates the method used for the barbell lift.  The most common value is A, but there are plenty of observations from each category to build a model.


```r
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

There are too many variables to feasibly plot each against classe.  Instead, to visualize classe's relationship with the independent variables, I performed Principle Component Analysis to reduce the dataset to two components.  These components capture approximately 30% of the variance, so while important, this does not provide a comprehensive picture of the data.  Below, I plot these components and use color to show the classe variable.


```r
training <- training[, (colSums(is.na(training))/nrow(training)) < 0.9]
preProc0 <- preProcess(subset(training, select = -classe), method = "pca", thresh = 0.3)
preProc0
```

```
## Created from 19622 samples and 59 variables
## 
## Pre-processing:
##   - centered (56)
##   - principal component signal extraction (56)
##   - scaled (56)
## 
## PCA needed 2 components to capture 30 percent of the variance
```

```r
twocomponents <- predict(preProc0, training)
training <- read.csv("trainingproject", na.strings = c("", "NA", "#DIV/0!"))
qplot(twocomponents$PC1, twocomponents$PC2, color = training$classe, alpha = 0.5, 
    xlab = "PC1", ylab = "PC2", main = "Two Principle Components and Classe")
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4.png) 


There is some amount of separation between the methods of performing lifts on these components (the region of red in between the four more densely populated regions for instance) but there is also considerable overlap.  This suggests that using just these two components will likely not produce a highly accurate model, so a more complex model will be necessasry.

### Pre-Processing and Preparing for Cross-Validation

First, I split the data into a training and "quizzing" set, labelling it quizzing to distinguish it from the test data which will ultimately be submitted and which I also load.


```r
sub <- sample(nrow(training), floor(nrow(training) * 0.7))
quizzing <- training[-sub, ]
training <- training[sub, ]
testing <- read.csv("testingproject", na.strings = c("", "NA", "#DIV/0!"))
```


The first real decision I make in pre-processing is how to manage the missing data.  I contemplate imputing the missing values via either K nearest neighbors or median imputation, but ultimately decide against these methods.  There simply are not enough values of the variables for imputing values for them to have a positive impact on the subsequent model.  The vast majority of the variables would be filled with  some linear combination of the existing variables, which will already be accounted for by any modelling strategy.  Additionally, including these variables via imputation will increase the risk of overfitting (since there are few true observations of those variables) and increase processing time.  So I conclude that the benefit of added predictive value from these few observations of these variables is outweighed by the aforementioned problems, and remove any variable missing more than 90% of its values.


```r
quizzing <- quizzing[, (colSums(is.na(training))/nrow(training)) < 0.9]
testing <- testing[, (colSums(is.na(training))/nrow(training)) < 0.9]
training <- training[, (colSums(is.na(training))/nrow(training)) < 0.9]
```

The second decision I make is to perform principle components analysis.  I find that I can capture 90% of the variance with 26 components (compared to 60 remaining variables).  This provides a significant reduction in noise, will help to reduce overfitting, and will speed processing time.  


```r
preProc1 <- preProcess(subset(training, select = -classe), method = "pca", thresh = 0.9)
processedTrain <- predict(preProc1, subset(training, select = -classe))
processedTesting <- predict(preProc1, testing)
processedQuizzing <- predict(preProc1, subset(quizzing, select = -classe))
processedTrain$classe <- training$classe
processedQuizzing$classe <- quizzing$classe
head(processedQuizzing)
```

```
##    user_name   cvtd_timestamp new_window    PC1     PC2    PC3    PC4
## 4   carlitos 05/12/2011 11:23         no -4.214 -0.6204 -3.697 0.7933
## 5   carlitos 05/12/2011 11:23         no -4.212 -0.6798 -3.671 0.7879
## 7   carlitos 05/12/2011 11:23         no -4.192 -0.6062 -3.675 0.7900
## 12  carlitos 05/12/2011 11:23         no -4.213 -0.6052 -3.674 0.7870
## 15  carlitos 05/12/2011 11:23         no -4.226 -0.5916 -3.666 0.7863
## 19  carlitos 05/12/2011 11:23         no -4.172 -0.6318 -3.703 0.7901
##      PC5    PC6     PC7    PC8    PC9    PC10    PC11    PC12   PC13
## 4  1.165 -1.902 -0.2172 -2.828 0.1480 -0.4176 -0.6497 -1.0478 -1.826
## 5  1.186 -1.941 -0.2051 -2.839 0.1242 -0.4649 -0.6139 -1.0589 -1.908
## 7  1.163 -1.865 -0.2268 -2.885 0.1168 -0.4050 -0.7056 -1.0177 -2.065
## 12 1.148 -1.851 -0.2373 -2.893 0.1013 -0.3489 -0.7344 -0.9941 -2.162
## 15 1.146 -1.838 -0.2558 -2.905 0.0814 -0.3683 -0.7638 -1.0035 -2.235
## 19 1.200 -1.851 -0.2554 -2.926 0.1247 -0.3485 -0.7869 -0.9944 -2.248
##      PC14     PC15       PC16    PC17    PC18    PC19    PC20    PC21
## 4  -3.275 -0.25720  0.0537298 -0.5887 -0.5377 -0.6310 -0.2750 -0.8242
## 5  -3.141 -0.02673  0.0539564 -0.4774 -0.4569 -0.6658 -0.2922 -0.7927
## 7  -2.917  0.48137  0.0178053 -0.6456 -0.4482 -0.6048 -0.2551 -0.8363
## 12 -2.704  0.97841 -0.0001672 -0.6785 -0.4100 -0.5864 -0.2470 -0.8415
## 15 -2.587  1.20835 -0.0004482 -0.6451 -0.3656 -0.5967 -0.2762 -0.8369
## 19 -2.396  1.63565 -0.0017990 -0.7142 -0.3393 -0.5616 -0.2678 -0.8047
##    classe
## 4       A
## 5       A
## 7       A
## 12      A
## 15      A
## 19      A
```


### Model Fitting

With pre-processing complete, I now fit a model to predict classe on the processed data.  I choose to use Random Forest because of it's suitability for multi-class classification and high accuracy, although the processing is slow.  I use accuracy as the metric since the goal for the submission portion of this project is to maximize accuracy.  


```r
model2 <- train(classe ~ ., data = processedTrain, method = "rf", na.action = na.omit)
model2
```

```
## Random Forest 
## 
## 13735 samples
##    24 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 13735, 13735, 13735, 13735, 13735, 13735, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa   Accuracy SD  Kappa SD
##    2    0.9429    0.9277  0.005039     0.006359
##   24    0.9826    0.9780  0.002323     0.002933
##   46    0.9738    0.9668  0.003804     0.004803
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 24.
```



### Out of Sample Error Estimates and Cross Validation

The train function in caret includes the bootstrapping method for cross-validation for Random Forests.  The sample sizes and number of repetitions for the bootstrapping can be seen above.  Because bootstrapping is included in the model building process, we expect the out of sample error rate to be approximately one minus the accuracy of the model listed above, which would make it approximately .02.

To confirm this, we create a Confusion Matrix using our quizzing partition and our model.  The below confusion matrix return an accuracy rate of .99 (close to .98)therefore out of sample error rate would indeed be approximately .01.  Included with the confusion matrix are a variety of other diagnostics on the model, all of which indicate high levels of accuracy.  They also could be informative if one type of error were more costly than another - for example,  informing people they were performing lifts correctly when in fact their method has a high risk of injury.  If that were the case, carefully examining the errors, and potentially ROC curves could help to fine-tune the model for real-world applications.  However, since for the submission portion of this assignment any misclassification is penalized identically, simple accuracy is the measure I focus on and optimize.


```r
confusionMatrix(predict(model2, processedQuizzing), quizzing$classe)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 351 104 193 199 244
##          B 320 346 482 171 237
##          C 366 249 148 156  43
##          D 100 132  92 272 309
##          E 513 263 111 208 278
## 
## Overall Statistics
##                                         
##                Accuracy : 0.237         
##                  95% CI : (0.226, 0.248)
##     No Information Rate : 0.28          
##     P-Value [Acc > NIR] : 1             
##                                         
##                   Kappa : 0.046         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.2127   0.3163   0.1442   0.2704   0.2502
## Specificity            0.8253   0.7475   0.8325   0.8703   0.7707
## Pos Pred Value         0.3217   0.2224   0.1538   0.3006   0.2025
## Neg Pred Value         0.7291   0.8273   0.8217   0.8527   0.8155
## Prevalence             0.2803   0.1858   0.1743   0.1709   0.1887
## Detection Rate         0.0596   0.0588   0.0251   0.0462   0.0472
## Detection Prevalence   0.1853   0.2643   0.1634   0.1537   0.2332
## Balanced Accuracy      0.5190   0.5319   0.4884   0.5703   0.5105
```


### Conclusion

Using the measurements provided from motion sensors, a highly accurate model can be created using caret's random forest procedure.  This model could be useful to provide feedback to weightlifters on the effectiveness of their technique.

