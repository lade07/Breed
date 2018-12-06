#Dog Breeds


##clear up working space
rm(list = ls())

###load libraries
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("EBImage")

library(EBImage)
library(keras)


###read data in
getwd()
setwd("/Users/lade/Documents/Tosin_R_root/Breed/breed_pics")
pics <- c('h1.jpg', 'h2.jpg', 'h3.jpg', 'h4.jpg', 'h5.jpg', 'h6.jpg', 'h7.jpg', 'h8.jpg', 'h9.jpg', 'r1.jpg', 'r2.jpg', 'r3.jpg', 'r4.jpg', 'r5.jpg', 'r6.jpg', 'r7.jpg', 'r8.jpg', 'r9.jpg')
mypic <- list()
for (i in 1:18) {
  mypic[[i]] <- readImage(pics[i])
}



###explore
print(mypic[[1]])
display(mypic[[11]])
summary(mypic[[4]])
hist(mypic[[6]])
str(mypic)



###Resize
for (i in 1:18) {
  mypic[[i]] <- resize(mypic[[i]], 28, 28)
}
str(mypic)



###reshape
for (i in 1:18) {
  mypic[[i]] <- array_reshape(mypic[[i]], c(28, 28, 3))
}
str(mypic)



###row bind
trainx <- NULL
for (i in 1:7) {
  trainx <- rbind(trainx, mypic[[i]])
}
str(trainx)

testx <- NULL
for (i in 8:9) {
  testx <- rbind(testx, mypic[[i]])
}
str(testx)
trainy <- c(0,0,0,0,0,0,0,1,1,1,1,1,1,1)
testy <- c(0,0,1,1)



###one hot encoding
trainLabels <- to_categorical(trainy)
testLabels <- to_categorical(testy)



###create model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(2352)) %>%
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dense(units = 2, activation = 'softmax')

summary(model)



###compile
model %>% 
  compile(loss = 'binary_crossentropy',
          optimizer = optimizer_rmsprop(),
          metrics = c('accuracy'))



###fit model
history <- model %>%
  fit(trainx,
      trainLabels,
      epochs = 30,
      batch_size = 32,
      validation_split = 0.2)
plot(history)


###evaluation and prediction of train data
model %>% evaluate(trainx, trainLabels)
pred <- model %>% predict_classes(trainx)
table(Predicted = pred, Actual = trainy)
prob <- model %>% predict_proba(trainx)
cbind(prob, Predicted = pred, Actual = trainy)



###evaluation and prediction of test data
model %>% evaluate(testx, testLabels)
pred_test <- model %>% predict_classes(testx)
table(Predicted = pred_test, Actual = testy)