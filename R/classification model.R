library(tidyverse)
library(caTools)
library(randomForest)
library(caret)
library(e1071)
library(rpart)
library(tidymodels)
library(schrute)
library(lubridate)
library(quantreg)
library(pROC)
library(klaR)
library(psych)
library(MASS)
library(devtools)
library(ROCR)
library(ElemStatLearn)
library(boot)

dataset = read.csv("C:\\Tal\\Data Engineer\\BSc\\semester 4\\Advanced programming\\project\\children_at_risk\\data\\Children_at_risk.csv")
datasetA <- dataset %>%
  filter(Test == 'A') %>% 
  mutate(Hope_A = Q1+Q2+Q3+Q4+Q5+Q6+Q7+(5-Q8)+Q9+(5-Q10)+(5-Q11)) %>% 
  dplyr::select(Id,Hope_A)

datasetB <- dataset %>%
  filter(Test == 'B') %>% 
  mutate(Hope_B = Q1+Q2+Q3+Q4+Q5+Q6+Q7+(5-Q8)+Q9+(5-Q10)+(5-Q11)) %>% 
  dplyr::select(Id,Hope_B)

datasetC <- dataset %>%
  filter(Test == 'C') %>% 
  mutate(Hope_C = Q1+Q2+Q3+Q4+Q5+Q6+Q7+(5-Q8)+Q9+(5-Q10)+(5-Q11)) %>% 
  dplyr::select(Id,Hope_C)

dataset = left_join(dataset,left_join(datasetA,left_join(datasetB,datasetC)))
dataset <- dataset %>%
  mutate(n_of_t = T1+T2+T3+T4+T5+T6+T7+T8+T9+T10)

data = dataset %>% 
  filter(Test == 'A') %>% 
  mutate(served = ifelse(A4==1|A5==1,1,0)) %>% 
  mutate(delta_hope = ifelse(Hope_B>Hope_A,1,0)) %>% 
  dplyr::select(Religiousness,Permanency,n_of_t,served,Hope_A,delta_hope)

set.seed(123)
training.samples <- data$served %>% 
  createDataPartition(p = 0.7, list = FALSE)
train.data  <- data[training.samples, ]
test.data <- data[-training.samples,]

model <- glm(delta_hope ~. , data = train.data, family = binomial)
# Summarize the model
summary(model)
model

probabilities <- model %>% predict(test.data, type = "response")

#Building ROC Curve

#ACC = TP / (TP + TN)
#TPR = TP / (TP + FN)
#FPR = FP / (FP + TN)

pred = prediction(probabilities, test.data$served)
perf = performance(pred, "acc")

roc = performance(pred,"tpr","fpr")
pr = performance(pred,"acc","tpr")

plot(roc, colorize = T, lwd = 2)
abline(a = 0, b = 1) 

plot(pr, colorize = T, lwd = 2)
abline(a = 0.5, b=0) 


# Making the Confusion Matrix
cm = table(test.data$delta_hope, probabilities > 0.5)
cm

train.data$delta_hope = as.factor(train.data$delta_hope)

trainC = trainControl(method = "cv",number = 5,savePredictions = T )
model <- train(delta_hope ~. , data = train.data,method="glm", family = "binomial",trControl = trainC)
summary(model)
model
probabilities <- model %>% predict(test.data, type = "prob")
cm = table(test.data$delta_hope, probabilities$`1` > 0.5)
cm

pred = prediction(probabilities$'1', test.data$delta_hope)
roc = performance(pred,"tpr","fpr")
pr = performance(pred,"prec","rec")

plot(roc, colorize = T, lwd = 2)
abline(a = 0, b = 1) 

plot(pr, colorize = T, lwd = 2)
abline(a = 0.5, b=0) 


prediction(probabilities[2], test.data$delta_hope) %>%
  performance(measure = "tpr", x.measure = "fpr") -> result

plotdata <- data.frame(x = result@x.values[[1]],
                       y = result@y.values[[1]], 
                       p = result@alpha.values[[1]])

p <- ggplot(data = plotdata) +
  geom_path(aes(x = x, y = y)) + 
  xlab(result@x.name) +
  ylab(result@y.name) +
  theme_bw()

dist_vec <- plotdata$x^2 + (1 - plotdata$y)^2
opt_pos <- which.min(dist_vec)

p + 
  geom_point(data = plotdata[opt_pos, ], 
             aes(x = x, y = y), col = "red") +
  annotate("text", 
           x = plotdata[opt_pos, ]$x,
           y = plotdata[opt_pos, ]$y+0.1,
           label = paste("p =", round(plotdata[opt_pos, ]$p, 3)))


cm = table(test.data$delta_hope, probabilities$`1` > 0.504)
cm

df = data.frame(Religiousness = mean(data$Religiousness),Permanency=mean(data$Permanency),n_of_t = mean(data$n_of_t),served=0,Hope_A=mean(data$Hope_A))
res = model %>% predict(df, type = "prob")

# Calculate the number of true positives, true negatives, false positives, and false negatives
tp <- cm[2, 2]
tn <- cm[1, 1]
fp <- cm[1, 2]
fn <- cm[2, 1]

# Calculate accuracy, recall, precision, and F1 score
accuracy <- (tp + tn) / sum(cm)
recall <- tp / (tp + fn)
precision <- tp / (tp + fp)
f1 <- 2 * precision * recall / (precision + recall)

# Print the results
cat("Accuracy:", round(accuracy, 3), "\n")
cat("Recall:", round(recall, 3), "\n")
cat("Precision:", round(precision, 3), "\n")
cat("F1 Score:", round(f1, 3), "\n")