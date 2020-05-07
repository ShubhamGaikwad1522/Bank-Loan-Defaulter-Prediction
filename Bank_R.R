rm(list=ls(all=T))
setwd("D:/Gaikwad/Data/Live Project 1/Project_1")
getwd()

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

## Read the data
df1 = read.csv("bank-loan.csv", header = T, na.strings = c(" ", "", "NA"))

###########################################Explore the data##########################################
str(df1)
# list types for each attribute
sapply(df1, class)

#Unique values in a column
unique(df1$ed)
summary(df1)
str(df1)
#convert each varaible
df1$age = as.numeric(df1$age)
df1$ed = as.numeric(df1$ed)
df1$employ = as.numeric(df1$employ)
df1$address = as.numeric(df1$address)
df1$debtinc = as.numeric(df1$debtinc)
df1$income = as.numeric(df1$income)
df1$creddebt = as.numeric(df1$creddebt)
df1$othdebtdebt = as.numeric(df1$othdebt)
typeof(df1)

##################################Missing Values Analysis###############################################

#sum of missing values
#sum(is.na(bank_loan))

#create dataframe with missing percentage

missing_val = data.frame(apply(df1,2,function(x){sum(is.na(x))}))

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
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(df1)) * 100

#arranging descending order
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL

#Rearranging the columns
missing_val = missing_val[,c(2,1)]

#writting output result back into disk
write.csv(missing_val, "Mising_perc_R.csv", row.names = F)
df <- DropNA(df1)

#Missing values are present only in "default" so it's not needed to plot bar graph.
# ggplot(data = missing_val[1:3,], aes(x=reorder(Columns, -Missing_percentage),y = Missing_percentage))+
#   geom_bar(stat = "identity",fill = "grey")+xlab("Parameter")+
#   ggtitle("Missing data percentage (Train)") + theme_bw()
sum(is.na(df1))

write.csv(df1, 'df_missing_R.csv', row.names = F)
summary(df)
str(df)


############################################Outlier Analysis#############################################
# BoxPlots - Distribution and Outlier Check

numeric_index = sapply(df,is.numeric) #selecting only numeric

numeric_data = df[,numeric_index]

cnames = colnames(numeric_data)

cnames

for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "default",group=1), data = subset(df))+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "blue" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="default")+
           ggtitle(paste("Box plot of loan default wrt",cnames[i])))
}

# Plotting plots together
gridExtra::grid.arrange(gn1,gn5,gn2,gn6,gn7,gn8,gn9,ncol=3)



##################################Feature Selection################################################
## Correlation Plot 
corrgram(df[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

###################################Model Development#######################################
#Clean the environment
rmExcept("df")

#Divide data into train and test using stratified sampling method
set.seed(1234)
train.index = createDataPartition(df$default, p = .80, list = FALSE)
train = df[ train.index,]
test  = df[-train.index,]
 
train$default<-as.factor(train$default)
str(train$default)



#Logistic Regression
logit_model = glm(default ~ ., data = train, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = test, type = "response")
logit_Predictions = predict(logit_model, newdata = test, type = "response")
#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)


##Evaluate the performance of classification model
ConfMatrix_lg = table(test$default, logit_Predictions)
confusionMatrix(ConfMatrix_lg)

#tpr=.85
#fpr=.475

#False Negative rate
#FNR = FN/FN+TP 

#ROC Curve
library(ROCR)
data(ROCR.simple)
pred <- prediction( ROCR.simple$predictions, ROCR.simple$labels)
roc <- performance(pred,"tpr","fpr")
plot(roc,
     colorize=T,
     main="ROC -Curve")   
abline(a=0,b=1)

#AUC curve
auc<- performance(pred,"auc")
auc<-unlist(slot(auc,"y.values"))
auc<- round(auc,4)
legend(.6,.2,auc,title="AUC",cex=4)

##Decision tree for classification
#Develop Model on training data

C50_model = C5.0(default ~., train, trials = 100, rules = TRUE)

#Summary of DT model
summary(C50_model)

#Lets predict for test cases
C50_Predictions = predict(C50_model, test[,-6], type = "class")
##Evaluate the performance of classification model
ConfMatrix_C50 = table(test$default, C50_Predictions)
confusionMatrix(ConfMatrix_C50)



#ROC Curve
library(ROCR)
data(ROCR.simple)
pred <- prediction( ROCR.simple$predictions, ROCR.simple$labels)
roc <- performance(pred,"tpr","fpr")
plot(roc,
     colorize=T,
     main="ROC -Curve")   
abline(a=0,b=1)

#AUC curve
auc<- performance(pred,"auc")
auc<-unlist(slot(auc,"y.values"))
auc<- round(auc,4)
# legend(.6,.2,auc,title="AUC",cex=4)
#False Negative rate
#FNR = FN/FN+TP 

###Random Forest
RF_model = randomForest(default ~ ., train, importance = TRUE, ntree = 500)

#Predict test data using random forest model
RF_Predictions = predict(RF_model, test[,-8])

##Evaluate the performance of classification model
ConfMatrix_RF = table(test$default, RF_Predictions)
confusionMatrix(ConfMatrix_RF)

#False Negative rate
#FNR = FN/FN+TP 

library(ROCR)
data(ROCR.simple)
pred <- prediction( ROCR.simple$predictions, ROCR.simple$labels)
perf <- performance(pred,"tpr","fpr")
plot(perf)    
#Accuracy = 76.47%
#FNR = 71.42%
library(ROCR)
data(ROCR.simple)
pred <- prediction( ROCR.simple$predictions, ROCR.simple$labels)
perf <- performance(pred,"tpr","fpr")
plot(perf)


#naive Bayes
library(e1071)

#Develop model
NB_model = naiveBayes(default ~ ., data = train)

#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,1:8], type = 'class')

#Look at confusion matrix
Conf_matrix = table(observed = test[,8], predicted = NB_Predictions)
confusionMatrix(Conf_matrix)

library(ROCR)
data(ROCR.simple)
pred <- prediction( ROCR.simple$predictions, ROCR.simple$labels)
perf <- performance(pred,"tpr","fpr")
plot(perf)
#statical way
mean(NB_Predictions == test$default)


