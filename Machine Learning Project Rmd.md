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
    main = "Histogram of Number of Missing Values in Each Column", binwidth = range/30)
```

```
## Error: non-numeric argument to binary operator
```


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
training <- training[sub, ]
quizzing <- training[-sub, ]
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
```


### Model Fitting

With pre-processing complete, I now fit a model to predict classe on the processed data.  I choose to use Random Forest because of it's suitability for multi-class classification and high accuracy, although the processing is slow.  I use accuracy as the metric since the goal for the submission portion of this project is to maximize accuracy.  


```r
model <- train(classe ~ ., data = processedTrain, method = "rf", na.action = na.omit)
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

```r
model
```

```
## Random Forest 
## 
## 13735 samples
##    25 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 13735, 13735, 13735, 13735, 13735, 13735, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa   Accuracy SD  Kappa SD
##    2    0.9449    0.9303  0.004183     0.005262
##   24    0.9830    0.9785  0.002404     0.003038
##   47    0.9663    0.9573  0.003785     0.004771
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 24.
```



### Out of Sample Error Estimates and Cross Validation

The train function in caret includes the bootstrapping method for cross-validation for Random Forests.  The sample sizes and number of repetitions for the bootstrapping can be seen above.  Because bootstrapping is included in the model building process, we expect the out of sample error rate to be approximately one minus the accuracy of the model listed above, which would make it approximately .03.

To confirm this, we create a Confusion Matrix using our quizzing partition and our model.  The below confusion matrix confirms that accuracy is approximately 0.97, and that therefore out of sample error rate would indeed be approximately .03.  Included with the confusion matrix are a variety of other diagnostics on the model, all of which indicate high levels of accuracy.  They also could be informative if one type of error were more costly than another - for example,  informing people they were performing lifts correctly when in fact their method has a high risk of injury.  If that were the case, carefully examining the errors, and potentially ROC curves could help to fine-tune the model for real-world applications.  However, since for the submission portion of this assignment any misclassification is penalized identically, simple accuracy is the measure I focus on and optimize.


```r
confusionMatrix(predict(model, processedQuizzing), quizzing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1172    0    0    0    0
##          B    0  766    0    0    0
##          C    0    0  745    0    0
##          D    0    0    0  715    0
##          E    0    0    0    0  700
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.286     
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
## Prevalence              0.286    0.187    0.182    0.174    0.171
## Detection Rate          0.286    0.187    0.182    0.174    0.171
## Detection Prevalence    0.286    0.187    0.182    0.174    0.171
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```


### Conclusion

Using the measurements provided from motion sensors, a highly accurate model can be created using caret's random forest procedure.  This model could be useful to provide feedback to weightlifters on the effectiveness of their technique.

