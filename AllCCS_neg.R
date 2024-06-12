library(readxl)

hyd <- read_excel("~/CCS/M-H_Training_Input_SMILES.xlsx")
descriptors <- read.csv("~/CCS/diff_descs.csv")
descriptors$mz <- NULL

#---Common across different % of input data (RFECV.ipynb, RFECV_pos.ipynb)---
fls <- list.files(pattern = "rfecv_[0-9]+\\.csv$")

for(fl in fls){
  name <- gsub("\\.csv", "", fl)
  assign(name, read.csv(fl))
}


descs <- lapply(mget(ls(pattern = "rfecv_[0-9]+")), function(x) colnames(x))
common <- Reduce(intersect, descs)
common[2] <- "m/z"
common <- common[-1]

merged <- merge(hyd, descriptors, by.x = c("METLIN.ID", "smiles"), by.y = c("METLIN_ID", "smiles"))
data <- merged[,common]
train <- data
colnames(train)[which(colnames(train) == "CCS_AVG")] <- "CCS"
testData <- read_excel("~/CCS/M-H_Testing_Input_SMILES.xlsx")
test <- merge(testData, descriptors, by.x = c("METLIN.ID", "smiles"), by.y = c("METLIN_ID", "smiles"))
test <- test[,common]
n <- ncol(test) - 1

#--- SVR ---
library(e1071)
library(caret)
library(foreach)
library(doParallel)
cores <- detectCores()

cl <- makeCluster(cores)
registerDoParallel(cl)

#Default
svmModel <- svm(CCS ~., data = train, kernel = "radial")

#AllCCS grid search
tune_grid <- expand.grid(gamma = 2^(1:15),
                         cost = c(0.001, 0.005, 0.025, 0.05, 0.1, 0.25, 0.5)/n)

model <- foreach(i = 1:nrow(tune_grid), .packages = "e1071") %dopar% {
  svm_model <- tune.svm(CCS ~ . , data = train, kernel = "radial",
                        gamma = tune_grid[i, "gamma"],
                        cost = tune_grid[i, "cost"])
  return(svm_model)
}
#save(model, file = "neg_hyperopt.rda")

combined_results <- do.call(rbind, lapply(model, function(x) x$performances))

best.parameters <- combined_results[which.min(combined_results$error), ]
optimized_SVM <- svm(CCS ~., data = train, kernel = "radial",
                     cost = best.parameters$cost,
                     gamma = best.parameters$gamma)



ytest <- test$CCS_AVG
ypred <- predict(svmModel, newdata = test[,-which(colnames(test) == "CCS_AVG")])
mae <- MAE(ypred, ytest)
rmse <- RMSE(ypred, ytest)
r2 <- R2(ypred, ytest, form = "traditional")
dt <- data.frame(yTest = test[,"CCS_AVG"], yPred = ypred)
dt <- cbind(test,dt)





