install.packages("norm")
install.packages("visdat")
install.packages("UpSetR")
install.packages("naniar")
install.packages("mice")
install.packages("tableone")
install.packages("ggplot2")
install.packages("lattice")
install.packages("caret")
install.packages("autoReg")
install.packages("dplyr")
install.packages("Matrix")
install.packages("Boruta")
install.packages("Hmisc")
install.packages("rms")
install.packages("glmnet")

# Load required packages
library(naniar)
library(mice)
library(tableone)
library(ggplot2)
library(lattice)
library(caret)
library(autoReg)
library(dplyr)
library(Matrix)
library(glmnet)
library(Boruta)
library(Hmisc)
library(rms)

# Set the working directory
setwd("C://")
if (.Platform$OS.type == "windows")
  {windowsFonts(Times = windowsFont("Times New Roman"))}
par(family = "Times")

# Read the input data
data = read.csv("origindata.csv",header = T, encoding = "GBK")

# Step 1: Data preprocessing
colnames(data)
data[,1:36] <- lapply(data[,1:36],as.factor)
str(data)
summary(data)

# Missing-data assessment and imputation
data %>%
  gg_miss_var(show_pct = TRUE) +
  theme_bw(base_size = 16, base_family = "Times")

missing_ratio <- colMeans(is.na(data))
missing_ratio_df <- data.frame(Variable = names(missing_ratio), Missing_Ratio = missing_ratio)
write.csv(missing_ratio_df, "Missing_ratio.csv", row.names = FALSE)

data_imputed <- mice(data, seed = 123, print = FALSE, m = 5)
data_imputed$method
data_imputed2 <- mice::complete(data_imputed, action = 2)
write.csv(data_imputed2, "data_complete.csv", row.names = FALSE)
data = read.csv("data_complete.csv",header = T, encoding = "GBK")
colMeans(is.na(data))

# Step 2: Split the training and validation sets
data$Outcome = factor(data$Outcome,levels = c(0,1))
set.seed(123)
inTrain = createDataPartition(y=data[,"Outcome"], p=0.8, list=F)
traindata = data[inTrain,]
valdata = data[-inTrain,]
write.csv(traindata, "traindata.csv", row.names = F)
write.csv(valdata, "valdata.csv", row.names = F)

# Step 3: Create baseline tables
myVars <- colnames(data[, 2:ncol(data)])
catVars <- colnames(data[, 2:36])
contVars <- colnames(data[, 37:ncol(data)])
noncenter <- c()
for (var in contVars) {
  if (is.numeric(data[[var]])) {
    test_result <- shapiro.test(data[[var]])
    if (test_result$p.value < 0.05) {
      noncenter <- c(noncenter, var)}
    } else {print(paste(var, "normally distributed")) }}
print(noncenter)

data[contVars] <- lapply(data[contVars], as.numeric)
table1 <- CreateTableOne(vars = myVars,
                         factorVars = catVars,
                         data = data,
                         strata='Outcome',
                         addOverall = TRUE)
table1 <- print(table1,showAllLevels = T,catDigits = 2,contDigits = 2,pDigits =3,nonnormal =noncenter)
write.csv(table1,file = "Overall_baselinetable.csv")

data[contVars] <- lapply(data[contVars], as.numeric)
table2 = CreateTableOne(vars = myVars,
                        factorVars = catVars,
                        data = traindata,
                        strata='Outcome',
                        addOverall = TRUE)
table2 <- print(table2,showAllLevels = T,catDigits = 2,contDigits = 2,pDigits =3,nonnormal =noncenter)
write.csv(table2,file = "Traning_baselinetable.csv")

# Step 4: Select predictors in the training set
traindata[,1:37] <- lapply(traindata[,1:37],as.factor)

# LASSO feature selection
min_max_scale = function(x){(x-min(x))/(max(x)-min(x))}
lassodata = traindata%>%
  mutate(across(all_of(catVars), as.factor)) %>%
  mutate_if(.predicate = is.numeric,
            .funs = min_max_scale)%>%
  as.data.frame()
set.seed(123)
x <- data.matrix(lassodata[, -1])
y <- lassodata[, 1]
y <- as.numeric(unlist(y))
lasso <- glmnet(x, y, family = "binomial",nlambda = 1000, alpha = 1)
print(lasso)
lasso.cv = cv.glmnet(x, y,alpha = 1,nfolds =10,family="binomial")
plot(lasso.cv, sign.lambda = 1)
abline(v = log(lasso.cv$lambda.min), col = "black", lty = 2)
abline(v = log(lasso.cv$lambda.1se), col = "black", lty = 2)
plot(lasso, xvar = "lambda", sign.lambda = -1)
abline(v = log(lasso.cv$lambda.min), col = "black", lty = 2)
abline(v = log(lasso.cv$lambda.1se), col = "black", lty = 2)
lasso.cv$lambda.min
lasso.cv$lambda.1se
coef(lasso.cv, s = "lambda.1se")
lasso_coefs <- coef(lasso.cv, s = "lambda.1se")
lasso_coefs_matrix <- as.matrix(lasso_coefs)
lasso_coefs_df <- as.data.frame(lasso_coefs_matrix)
rownames(lasso_coefs_df) <- rownames(lasso_coefs_matrix)
write.csv(lasso_coefs_df, "LASSO.csv", row.names = TRUE)

# Boruta feature selection
set.seed(123)
boruta_model <- Boruta(Outcome~.,
                       data=traindata,
                       pValue = 0.01,
                       mcAdj = TRUE,
                       maxRuns = 100,
                       doTrace = 1,
                       holdHistory = TRUE,
                       getImp = getImpRfZ)
print(boruta_model)
plot(boruta_model,
     sort=T,
     whichShadow=c(T,T,T),
     las=2,
     xlab='')
final_model <- TentativeRoughFix(boruta_model)
print(final_model)
selected_features <- getSelectedAttributes(final_model, withTentative = FALSE)
selected_features_df <- data.frame(Variables = selected_features)
write.csv(selected_features_df, "Boruta.csv", row.names = FALSE)
plotImpHistory(boruta_model,type='l')
