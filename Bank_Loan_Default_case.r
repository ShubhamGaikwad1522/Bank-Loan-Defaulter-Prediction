rm(list=ls(all=T))
setwd("D:/Gaikwad/Data/Live Project 1/Project_1")

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')
#install.packages(x)

lapply(x, require, character.only = TRUE)
rm(x)

#read cvs file
bank_loan = read.csv("bank-loan.csv", header = T, na.strings = c(" ", "", "NA"))

########################################################################################################
##Explore the data

str(bank_loan)
# list types for each attribute
sapply(bank_loan, class)

#Unique values in a column
unique(bank_loan$ed)
summary(bank_loan)
str(bank_loan)
#convert each varaible
bank_loan$age = as.numeric(bank_loan$age)
bank_loan$ed = as.numeric(bank_loan$ed)
bank_loan$employ = as.numeric(bank_loan$employ)
bank_loan$address = as.numeric(bank_loan$address)
bank_loan$debtinc = as.numeric(bank_loan$debtinc)
bank_loan$income = as.numeric(bank_loan$income)
bank_loan$creddebt = as.numeric(bank_loan$creddebt)
bank_loan$othdebtdebt = as.numeric(bank_loan$othdebt)
typeof(bank_loan)



##############################################################################################################
#Missing Values Analysis

#sum of missing values
#sum(is.na(bank_loan))

#create dataframe with missing percentage

missing_val = data.frame(apply(bank_loan,2,function(x){sum(is.na(x))}))
#(explanation of above code: 
#(1) Creating own function of x as I need 
#to calculate missing values from all columns.
#(2) 2 is written as we are doing operations on column.
#(3)Apply: is used for calculating missing values of all the variables)


#convert row names into columns
missing_val$Columns = row.names(missing_val)

#Rename the variable name
names(missing_val)[1] =  "Missing_percentage"

#Calculate percentage
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(bank_data)) * 100

#arranging descending order
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL

#Rearranging the columns
missing_val = missing_val[,c(2,1)]

#writting output result back into disk
write.csv(missing_val, "Missing_perc-r.csv", row.names = F)

#Missing values are present only in "default" so it's not needed to plot bar graph.

###test the missing value
#actual = 45
#mean = 35.01767
#median = 34
#knn =  37.48551
#bank_loan[40,1] = NA
#freez the KNN as the respective value is closer to 45 in KNN method


#Mean Method
#bank_loan$age[is.na(bank_loan$age)] = mean(bank_loan$age, na.rm = T)

#Median Method
#bank_loan$age[is.na(bank_loan$age)] = median(bank_loan$age, na.rm = T)

# kNN Imputation
bank_loan = knnImputation(bank_loan, k = 3)


########################################################################################################
#Outlier Analysis#

#BoxPlots - Distribution and Outlier Check/analysis
numeric_index = sapply(bank_loan,is.numeric) #selecting only numeric
numeric_data = bank_loan[numeric_index]

#numeric_index

cnames = c("age", "employ", "address", "income", "debtinc", "creddebt", "othdebt", "default")

#As there are multiple numeric variable choose the for_loop for plotting the box plot
# Meaning of assign(paste0("gn",i)= assign is used to assign the name to dataset and in this case gn1 to gn8
# will be assign to loops executing in the for loop step by step.

for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "default"), data = subset(bank_loan))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="default")+
           ggtitle(paste("Box plot of default for",cnames[i])))
}

# Plotting plots together
gridExtra::grid.arrange(gn1,gn5,gn2,gn6,gn7,gn8,ncol=3)

#removing outlyer via box plot
# defining new data frame for experiment
df = bank_loan

val = bank_loan$income[bank_loan$income %in% boxplot.stats(bank_loan$income)$out]

#val

# ! is used as not symbol in below code, which is for writting condition 
# don't select that oberservations which contain the value for val(in above code)

bank_loan = bank_loan[which(!bank_loan$income %in% val),]

#Loop to remove from all varibales
for (i in cnames){
  print(i)
  val=bank_loan[,i][bank_loan[,i]%in% boxplot.stats(bank_loan[,i])$out]
  print(length(val))
  bank_loan = bank_loan[which(!bank_loan[,i] %in% val),]
}

#################################################################################################
#Feature Selection

#Correlation Plot 

#bank_loan = df
#Explanation= Order is F don't need to follow any numeric variable order, #upper pannel is pie chart
corrgram(bank_loan[,cnames], order = F, upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

## Explanation of plot: red color indicates that variables are negatively correlated and blue color indicates
# that variables are positively correlated.

#Here we will not take any selection algo on categorial because we have only one categorial variable


#Dimension Reduction
bank_loan = subset(bank_loan, select = -c(creddebt,othdebt))



##############################################################################################################
#Feature Scaling

#Normality check
qqnorm(bank_loan$income)
hist(bank_loan$income) 

#Normalisation
cnames = c("age", "employ", "address", "income", "debtinc")

#(in consol crosscheck cnames)

for(i in cnames){
  print(i)
  bank_loan[,i] = (bank_loan[,i] - min(bank_loan[,i]))/
    (max(bank_loan[,i] - min(bank_loan[,i])))
}

#Standardisation

#reloaded the data which came after feature selection

for(i in cnames){
  print(i)
  bank_loan[,i] = (bank_loan[,i] - mean(bank_loan[,i]))/
                                 sd(bank_loan[,i])
}



#############################################################################################################
#Model Development

#Clean the environment
library(DataCombine)
rmExcept("bank_loan")

#Divide data into train and test using stratified sampling method
set.seed(1234)

train.index = createDataPartition(bank_loan$default, p = .80, list = FALSE)
train = bank_loan[ train.index,]
test  = bank_loan[-train.index,]

train$default=as.factor(train$default)
str(train$default)

#Logistic Regression
logit_model = glm(default ~ ., data = train, family = "binomial")

#explanation of above: . means consider all variable except default
# glm iss built in function for builting ligistic regression
# default is target variable

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = test, type = "response")

#cross check
#logit_Predictions

#convert prob into 1 and 0
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)


##Evaluate the performance of classification model
ConfMatrix_RF = table(test$default, logit_Predictions)

#False Negative rate
FNR = FN/FN+TP 


##Decision tree for classification
#Develop Model on training data
str(bank_loan$default)
bank_loan$default[bank_loan$default %in% "1"] = "yes"
bank_loan$default[bank_loan$default %in% "0"] = "no"

C50_model = C5.0(default ~., train, trials = 100, rules = TRUE)

#Summary of DT model
summary(C50_model)

#write rules into disk
write(capture.output(summary(C50_model)), "c50Rules.txt")

#Lets predict for test cases
C50_Predictions = predict(C50_model, test[,-6], type = "class")

##Evaluate the performance of classification model
ConfMatrix_C50 = table(test$default, C50_Predictions)
confusionMatrix(ConfMatrix_C50)

#False Negative rate
#FNR = FN/FN+TP 


###Random Forest
RF_model = randomForest(default ~ ., train, importance = TRUE, ntree = 500)

#Presdict test data using random forest model
RF_Predictions = predict(RF_model, test[,-6])

##Evaluate the performance of classification model
ConfMatrix_RF = table(test$default, RF_Predictions)
confusionMatrix(ConfMatrix_RF)

#False Negative rate
#FNR = FN/FN+TP 

##KNN Implementation
library(class)

#Predict test data
KNN_Predictions = knn(train[, 1:6], test[, 1:6], train$default, k = 6)

#Confusion matrix
Conf_matrix = table(KNN_Predictions, test$default)

#Accuracy
sum(diag(Conf_matrix))/nrow(test)

#False Negative rate
#FNR = FN/FN+TP 


#naive Bayes
library(e1071)

#Develop model
NB_model = naiveBayes(default ~ ., data = train)

#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,1:6], type = 'class')

#Look at confusion matrix
Conf_matrix = table(observed = test[,6], predicted = NB_Predictions)
confusionMatrix(Conf_matrix)


#statical way
mean(NB_Predictions == test$default)


