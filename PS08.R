library(tidyverse)
library(caret)

# Package for easy timing in R
library(tictoc)



# Get data ----------------------------------------------------------------
# Accelerometer Biometric Competition Kaggle competition data
# https://www.kaggle.com/c/accelerometer-biometric-competition/data
train <- read_csv("~/Downloads/train.csv")

# YOOGE!
dim(train)



# knn modeling ------------------------------------------------------------
model_formula <- as.formula(Device ~ X + Y + Z)

# Values to use:
n_values <- seq(from=1,to=2900000,by=20000)
k_values <- seq(from=1, to=10000,by=2000)

runtime_dataframe <- expand.grid(n_values, k_values) %>%
  as_tibble() %>%
  rename(n=Var1, k=Var2) %>%
  mutate(runtime = n*k)



# Time knn here -----------------------------------------------------------
time<-function(runtime_dataframe=runtime_dataframe){
for (i in 1:nrow(runtime_dataframe) ) {
  sam_size<-runtime_dataframe$n[i]
  train_sub <- slice(train,1:sam_size)
tic()
model_knn <- caret::knn3(model_formula, data=train_sub,
                         k = runtime_dataframe$k[i])
clock<-toc()
runtime_dataframe$runtime[i] <- clock$toc - clock$tic}
  
  
  return(runtime_dataframe)
}

runtime_dataframe1<-time(runtime_dataframe) 
write.csv(runtime_dataframe1,file="runtime.csv") 

# Plot your results ---------------------------------------------------------
# Think of creative ways to improve this barebones plot. Note: you don't have to
# necessarily use geom_point

runtime_plot1<-ggplot(runtime_dataframe1) +
  geom_point(aes(x=n, y=k, col=runtime), size=2)
runtime_plot1
runtime_plot2 <- ggplot(runtime_dataframe1, aes(x=n, y = runtime, group=k, col = k)) +
  geom_line() +
  labs(title = "Runtime with Varying n")
runtime_plot2
runtime_plot3 <- ggplot(runtime_dataframe1, aes(x=k, y = runtime, group=n, col = n)) +
  geom_line() +
  labs(title = "Runtime with Varying k")
runtime_plot3

runtime_plot4<-ggplot(runtime_dataframe1, aes(x=n, y=k))+
  geom_raster(aes(fill = runtime), hjust=0.5, vjust=0.5, interpolate=FALSE)

runtime_plot4
ggsave(filename="Pei_Gong.png", width=16, height = 9)


# Runtime complexity ------------------------------------------------------
# Can you write out the rough Big-O runtime algorithmic complexity as a function
# of:
# -n: number of points in training set 
# -k: number of neighbors to consider
# -d: number of predictors used? In this case d is fixed at 3 

#O(nk+nd) 
#step1, for each of the points we compute all the distance , which is nd 
#step2: after storing the distance nd,we compare the values to find k neighbors, 
#which involve loopoing through the distance, for n points we have complexity nk 
#therefore in total we have O(nd+nk). when d=3, we have O(3n+nk). 

#The graph above supports my hypothesis. 




