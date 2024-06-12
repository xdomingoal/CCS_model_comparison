hyd <- read_excel("~/CCS/M+H_Training_Input_SMILES.xlsx")
na <- read_excel("~/CCS/M+Na_Training_Input_SMILES.xlsx")
pos <- rbind(hyd, na)
descriptors <- read.csv("~/CCS/diff_descs.csv")
descriptors$mz <- NULL

#---Get the common ones across different % of input data (RFECV.ipynb, RFECV_pos.ipynb)---
fls <- list.files(pattern = "rfecv_[0-9]+_pos\\.csv")

for(fl in fls){
  name <- gsub("\\.csv", "", fl)
  assign(name, read.csv(fl))
}


descs <- lapply(mget(ls(pattern = "rfecv_[0-9]+")), function(x) colnames(x))
common <- Reduce(intersect, descs)
common[2] <- "m/z"
common <- common[-1]

merged <- merge(pos, descriptors, by.x = c("METLIN.ID", "smiles"), by.y = c("METLIN_ID", "smiles"))
data <- merged[,common]
train <- data
colnames(train)[which(colnames(train) == "CCS_AVG")] <- "CCS"
hydTest <- read_excel("~/CCS/M+H_Testing_Input_SMILES.xlsx")
naTest <- read_excel("~/CCS/M+Na_Testing_Input_SMILES.xlsx")
testData <- rbind(hydTest, naTest)
test <- merge(testData, descriptors, by.x = c("METLIN.ID", "smiles"), by.y = c("METLIN_ID", "smiles"))

desc100 <- colnames(rfecv_100_pos)[-1]
desc100[1] <- "mz"
# test <- test[,c(desc100, "CCS_AVG")]
test <- test[,common]


#--- SVR ---
library(e1071)
library(caret)
library(foreach)
library(doParallel)
cores <- detectCores()

cl <- makeCluster(cores)
registerDoParallel(cl)

#svmModel <- svm(CCS ~., data = train, kernel = "radial")

t0 <- Sys.time()
#n <-  8
tune_grid <- expand.grid(gamma = 2^(1:15),
                         cost = c(0.001, 0.005, 0.025, 0.05, 0.1, 0.25, 0.5)/n)

model <- foreach(i = 1:nrow(tune_grid), .packages = "e1071") %dopar% {
  svm_model <- tune.svm(CCS ~ . , data = train, kernel = "radial",
                        gamma = tune_grid[i, "gamma"],
                        cost = tune_grid[i, "cost"])
  return(svm_model)
}
Sys.time() - t0
save(model, file = "pos_hyperopt.rda")

stopCluster(cl)
registerDoSEQ()

load("pos_hyperopt.rda")
combined_results <- do.call(rbind, lapply(model, function(x) x$performances))

best.parameters <- combined_results[which.min(combined_results$error), ]
best.parameters
optimized_SVM <- svm(CCS ~., data = train, kernel = "radial",
                     cost = 0.0625,
                     gamma = 2)

ytest <- test$CCS_AVG
#ypred <- predict(svmModel, newdata = test[,-which(colnames(test) == "CCS_AVG")])
ypred <- predict(optimized_SVM, newdata = test[,-which(colnames(test) == "CCS_AVG")])
mae <- MAE(ypred, ytest)
rmse <- RMSE(ypred, ytest)
r2 <- R2(ypred, ytest, form = "traditional")
cat(mae, rmse, r2)




