library(e1071)
library(caTools)
library(tree) # Contains the "tree" function
library(caret)
library(rattle)
library(ggplot2)
library(maptree)
library(ggcorrplot)
library(Amelia)
library(yardstick)
library(visdat)

# Normalization with Z-Score
'%ni%' <- Negate('%in%') # define 'not in' function
# Function for centering a vector
center <- function(v) { v-mean(v) }
# Function for scaling a vector
scale <- function(v) { v/sd(v) }

# Calculate statistics of the ML Models
statistics_of_model <- function(model){
  result <- confusionMatrix(model)
  precision <- result$table[1, 1] / (result$table[1, 1] + result$table[1, 2])
  recall <- result$table[1, 1] / (result$table[1, 1] + result$table[2, 1])
  f1_score <- 2 * ((precision*recall) / (precision + recall))
  print("Confusion Matrix: ")
  print(confusionMatrix(model)$table)
  print(paste0("Precision: ", precision))
  print(paste0("Recall: ", recall))
  print(paste0("F-score: : ", f1_score))
  print(paste0("Accuracy: : ", max(model$results$Accuracy)))
  print(paste0("Error Rate: : ", 1-max(model$results$Accuracy)))
}
dataPreprocess <- function(){
  dataFrame <- read.csv("train.csv", sep=",",dec = ",")
  dataFrame[dataFrame==""] <- NA #replace all empty cells with NA's
  # NA value plot  
  windows()
  plot(vis_dat(dataFrame))
  
  # Delete NA values
  df <- data.frame(na.omit(dataFrame)) #omit all NA rows
  df$income_.50K<- as.factor(df$income_.50K) #convert target feature to a factor
  DF1 <- as.data.frame(df) 
  
   must_convert<-sapply(DF1,is.numeric)  #logical vector telling which variables displayed as numeric
   M2<-sapply(DF1[,!must_convert],unclass)    # data.frame of all categorical variables now displayed as numeric
   merged<-cbind(DF1[,must_convert],M2)        # complete data.frame with all variables put together
   merged<-data.matrix(DF1)
   DF2 <- as.data.frame(merged)
  
   ggplot(DF2) + aes(x=as.numeric(native.country), group=income_.50K, fill=income_.50K, color=income_.50K) +
     geom_histogram(binwidth=1, fill = "white")+
     labs(x="native.country",y="Count",title = "Income vs. native.country")
   
   
  
  # Target feature levels distribution  
  windows()
  barplot(table(DF1[,"income_.50K"]), xlab = "Target Feature Type")
  legend("topright", 
         legend = c("0 - <=$50K" , "1 - >$50K "))
  
  # Correlation Matrix
  nums2 <- unlist(lapply(DF1,is.integer))# takes integer values
  corr_mat=cor(DF1[,nums2],method="s") #spearman
  windows()
  plot(ggcorrplot(corr_mat, hc.order = TRUE, outline.col = "white", lab=TRUE, lab_size=5, type = "upper"))
  
  # Descriptive feature plot
  windows()
  plot(ggplot(DF1) + aes(x=as.numeric(age), group=income_.50K, fill=income_.50K, color=income_.50K) +
         geom_histogram(binwidth=1, fill = "white")+
         labs(x="Age",y="Count",title = "Income w.r.t Age"))
  
  windows()
  plot(ggplot(DF1) + aes(x=as.numeric(educational.num), group=income_.50K, fill=income_.50K, color=income_.50K) +
         geom_histogram(binwidth=1, fill = "white")+
         labs(x="Educational Num",y="Count",title = "Income w.r.t Educational Num"))
  
  windows()
  plot(ggplot(DF1, aes(x=as.numeric(hours.per.week), group=income_.50K, fill=income_.50K, color=income_.50K)) +
         geom_histogram(fill="white", binwidth = 5)+
         labs(x="Hours per week",y="Number of Samples",title = "Income vs Hours per week"))
  
  
  # z-score calculation
  DF1[,c(1,3,5,11,12,13)] <- DF1[,c(1,3,5,11,12,13)] %>%
    apply(MARGIN = 2, FUN = center) %>%
    apply(MARGIN = 2, FUN = scale)
  
  # Income vs. fnlwgt plot
  windows()
  plot(ggplot(DF1, aes(x=as.numeric(fnlwgt), group=income_.50K, fill=income_.50K, color=income_.50K)) +
         geom_histogram(fill="white", binwidth = 1)+
         labs(x="Final Weight",y="Number of Samples",title = "Income vs Final Weight"))
  
  # Delete unused column
  DF1 <- subset(DF1, select=-native.country)
  DF1 <- subset(DF1, select=-race)
  DF1 <- subset(DF1, select=-fnlwgt)
  DF1 <- subset(DF1, select=-education)
  
  # Under-Sampling
  smaller_values <- subset(DF1 , DF1$income_.50K == 0)
  greater_values <- subset(DF1 , DF1$income_.50K == 1)
  smaller_ind <- sample(2, nrow(smaller_values), replace = T, prob = c(0.53, 0.47)) # 30000 var. 0.66 2-1 yapar.
  
  trainSet <- rbind(greater_values, smaller_values[smaller_ind == 1, ])
  set.seed(42)
  rows <- sample(nrow(trainSet))
  shuffleTrain <- trainSet[rows,]
  
  # Descriptive features plot (capital gain-capital loss)
  
  windows()
  plot(ggplot(shuffleTrain) + aes(x=as.numeric(capital.gain), group=income_.50K, fill=income_.50K, color=income_.50K) +
         geom_histogram(binwidth=1, fill = "white")+
         labs(x="Capital Gain",y="Count",title = "Income w.r.t Capital Gain"))
  
  windows()
  plot(ggplot(shuffleTrain) + aes(x=as.numeric(capital.loss), group=income_.50K, fill=income_.50K, color=income_.50K) +
         geom_histogram(binwidth=1, fill = "white")+
         labs(x="Capital Loss",y="Count",title = "Income w.r.t Capital Loss"))
  
  return(shuffleTrain)
}

#ML Models Function
naiveBayesFunc <- function(df, tControl){
  
  naive_model <- train(income_.50K ~ age + workclass + educational.num + marital.status + occupation + relationship + gender + capital.loss+ capital.gain + hours.per.week, data=df,
                       na.action=na.pass, method="naive_bayes", trControl=tControl, tuneLength=13)
  
  return(naive_model)
}

decisionTreeFunc <- function(df, tcontrol){
  
  decisionTreeModel <- train(income_.50K ~ age + workclass + educational.num + marital.status + occupation + relationship + gender + capital.loss+ capital.gain + hours.per.week, data=df,
                             na.action=na.pass, method="rpart2", trControl=tControl, tuneLength=15)
  
  
  return(decisionTreeModel)
}

knnFunc <- function(df, tControl){
  
  knnModel <- train(income_.50K ~ age + workclass + educational.num + marital.status + occupation + relationship + gender + capital.loss+ capital.gain + hours.per.week, data=df,
                    na.action=na.pass, method="knn", trControl=tControl, tuneLength=10)
  
  return(knnModel)
}

svmFunc <- function(df, tControl){
  
  svmModel <- train(income_.50K ~ age + workclass + educational.num + marital.status + occupation + relationship + gender + capital.loss+ capital.gain+ hours.per.week , data=df,
                    na.action=na.pass, method="svmLinearWeights2", trControl=tControl)
  
  return(svmModel)
}

logisticFunc <- function(df, tControl){
  
  logitMod<- train(income_.50K ~ age + workclass + educational.num + marital.status + occupation + relationship + gender + capital.loss+ capital.gain + hours.per.week , data=df, 
                   na.action=na.pass, method="regLogistic", trControl=tControl)
  
  return(logitMod)
}

neuralFunc <- function(df, tControl){
  
  NnMod<- train(income_.50K ~ age + workclass + educational.num + marital.status + occupation + relationship + gender + capital.loss+ capital.gain + hours.per.week , data=df, 
                na.action=na.pass, method="nnet", trControl=tControl)
  
  return(NnMod)
}

set.seed(80654)

trainsets <- dataPreprocess()
# k-fold parameters
tControl <- trainControl(method="repeatedcv", repeats = 5, number=4, verboseIter = TRUE)

# ML models create, tune parameters plot, run time calculation and accuracy plot
#bayes
start_time <- Sys.time()
naive_model <- naiveBayesFunc(trainsets,tControl)
finish_time <- Sys.time()
naive_time <- finish_time - start_time

plot(naive_model)
plot(row.names(naive_model$resample),naive_model$resample[,1], main = "Bayes Accuracy Plot",type="b" , lwd=2 ,col= "red" , ylab="Accuracy" , xlab="Repeat Number" , bty="l" , pch=20 , cex=2, xlim = c(1,25), ylim = c(0.73,0.85))

#decision tree
start_time <- Sys.time()
dtModel <- decisionTreeFunc(trainsets,tControl)
finish_time <- Sys.time()
decision_tree_time <- finish_time - start_time

plot(dtModel)
fancyRpartPlot(dtModel$finalModel)
plot(row.names(dtModel$resample),dtModel$resample[,1], main = "Decision Tree Accuracy Plot",type="b" , lwd=2 ,col= "blue" , ylab="Accuracy" , xlab="Repeat Number" , bty="l" , pch=20, cex=2, xlim = c(1,21), ylim = c(0.73,0.85))

#knn
start_time <- Sys.time()
knnModel <- knnFunc(trainsets,tControl)
finish_time <- Sys.time()
kNN_time <- finish_time - start_time
plot(knnModel ,cex=2)
plot(row.names(knnModel$resample),knnModel$resample[,1], main = "Knn Accuracy Plot",type="b" , lwd=2 ,col= "green" , ylab="Accuracy" , xlab="Repeat Number" , bty="l" , pch=20, cex=2, xlim = c(1,21), ylim = c(0.73,0.85))

#logisticModel
start_time <- Sys.time()
logisticModel <- logisticFunc(trainsets, tControl)
finish_time <- Sys.time()
logistic_time <- finish_time - start_time
plot(logisticModel)
plot(row.names(logisticModel$resample),logisticModel$resample[,1], main = "Logistic Accuracy Plot",type="b" , lwd=2 ,col= "yellow" , ylab="Accuracy" , xlab="Repeat Number" , bty="l" , pch=20, cex=2, xlim = c(1,21), ylim = c(0.7,0.85))

#svmModel
start_time <- Sys.time()
svmModel <- svmFunc(trainsets, tControl)
finish_time <- Sys.time()
svm_time <- finish_time - start_time
plot(row.names(svmModel$resample),svmModel$resample[,1], main = "SVM Accuracy Plot",type="b" , lwd=2 ,col= "purple" , ylab="Accuracy" , xlab="Repeat Number" , bty="l" , pch=20, cex=2, xlim = c(1,21), ylim = c(0.75,0.85))
plot(svmModel)

#NnModel
start_time <- Sys.time()
nnModel <- neuralFunc(trainsets, tControl)
finish_time <- Sys.time()
nn_time <- finish_time - start_time
plot(row.names(nnModel$resample),nnModel$resample[,1], main = "Neural Network Accuracy Plot",type="b" , lwd=2 ,col= "orange" , ylab="Accuracy" , xlab="Repeat Number" , bty="l" , pch=20, cex=2, xlim = c(1,21), ylim = c(0.75,0.85))
plot(nnModel)

statistics_of_model(naive_model)
statistics_of_model(dtModel)
statistics_of_model(knnModel)
statistics_of_model(logisticModel)
statistics_of_model(svmModel)
statistics_of_model(nnModel)

# Accuracy Comparison plot
plot( naive_model$resample[,1] ~row.names(naive_model$resample) , type="b" , bty="l" , xlab="Repeat Number" , ylab="Accuracy" , col= "red" , lwd=2 , pch=20 , ylim=c(0.7,0.93) , bty="l" , cex=2, xlim =c(1,21))
lines(dtModel$resample[,1] ~row.names(dtModel$resample) ,type="b", lwd=2 ,col= "blue", bty="l" , pch=20, cex=2)
lines(knnModel$resample[,1] ~row.names(knnModel$resample) ,type="b", lwd=2 ,col= "green", bty="l" , pch=20, cex=2)
lines(logisticModel$resample[,1] ~row.names(logisticModel$resample) ,type="b", lwd=2 ,col= "yellow", bty="l" , pch=20, cex=2)
lines(svmModel$resample[,1] ~row.names(svmModel$resample) ,type="b", lwd=2 ,col= "purple", bty="l" , pch=20, cex=2)
lines(nnModel$resample[,1] ~row.names(nnModel$resample) ,type="b", lwd=2 ,col= "orange", bty="l" , pch=20, cex=2)

legend("topright", 
       legend = c("Naive Bayes", "Decision Tree", "k-Nearest Neighbor", "Logistic Regression", "Support Vector Machine", "Neural Network"), 
       col = c("red", "blue", "green", "yellow","purple", "orange"),
       pch = c(20,20,20,20,20), 
       bty = "n", 
       text.col = "black", 
       horiz = F,
       cex = 1,y.intersp=0.8)

