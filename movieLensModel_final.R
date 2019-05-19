#First, load the edx and validation data sets created by the provided script. They are saved locally.
#Then, load the necessary packages.
load("MovieLensDefaults.RData")
library(tidyverse)
library(caret)

#Set a seed and randomly create a train/test split (90% train)from the edx data set.
#This will allow us to have a test set to find the RMSE after our script has made predictions.
set.seed(1)
test_index<- createDataPartition(y=edx$rating, times=1, p=0.1, list=FALSE)
test_set=edx[test_index,]
train_set=edx[-test_index,]

#Then, randomly create a validation set out of the train set. We can use this to determine any parameters
#without using the test data (which would wrongly allow information to leak from the test data to our algorithm).
valid_index=createDataPartition(y=train_set$rating,times=1,p=0.1,list=FALSE)
valid_set=train_set[valid_index,]
train_set=train_set[-valid_index,]

#Generates the unique genre combination list, as well as the base list of genres. Note that there are too many unique combinations to
#make the use of the combinations practical; instead, we will have to depend on the individual genres.
genre_combo_list=unique(train_set$genres)
genre_list=unique(flatten(str_split(genre_combo_list,"\\|")))

# Adds columns for each unique genre, with true entries if that appears with the combined genre string
# This is done for both the train and test set
for (ii in 1:20) {
  train_set[as.character(genre_list[ii])]=grepl(genre_list[ii] ,train_set$genres)
  test_set[as.character(genre_list[ii])]=grepl(genre_list[ii] ,test_set$genres)
}

#For efficiency, I now remove the original loaded data as well as garbage collected (i.e., free up) memory
rm(edx,validation)
gc()

#Define two loss functions: RMSE and Accuracy (the original metric; no longer used)
RMSE <- function(predicted_ratings, true_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2,na.rm=TRUE))
}

compute_accuracy <- function(predicted_ratings, true_ratings){
  predicted_ratings=round(predicted_ratings*2)/2
  mean(predicted_ratings==true_ratings,na.rm=TRUE)
}

#Compute the average for all movies (mu) as well as the average residual for each individual movie (b_i)
#This can be interpreted as, how do users rate movies in general (mu) and how good is an individual movie (b_i)
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

#Original code, to demonstrate movie effect only
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i
model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
model_1_accuracy = compute_accuracy(predicted_ratings,test_set$rating)

#Compute the average residual for each user (b_u) after removing the average score and individual movie score.
#This can be interpreted as, "how much of a curmudgeon is this user, on average?" (b_u)
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#Original code, demonstrates movie + user effect
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
model_2_accuracy = compute_accuracy(predicted_ratings,test_set$rating)


#Create a train set that includes the residual left after removing the average, movie effect, and user effect.
train_set_resid <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u,resid=rating-pred)

#Compute the residual and number of movies rated for each genre for each user.
#I.e., how much affinity does each user have towards a particular genre, after removing all previous effects?
genre_resids = train_set_resid %>% filter(.[,7] == TRUE) %>% group_by(userId) %>% summarize('Comedy resid'=mean(resid),'Comedy n' = n())
genre_resids = train_set_resid %>% filter(.[,8] == TRUE) %>% group_by(userId) %>% summarize('Romance resid'=mean(resid),'Romance n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,9] == TRUE) %>% group_by(userId) %>% summarize('Action resid'=mean(resid),'Action n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,10] == TRUE) %>% group_by(userId) %>% summarize('Drama resid'=mean(resid),'Drama n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,11] == TRUE) %>% group_by(userId) %>% summarize('Sci-Fi resid'=mean(resid),'Sci-Fi n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,12] == TRUE) %>% group_by(userId) %>% summarize('Thriller resid'=mean(resid),'Thriller n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,13] == TRUE) %>% group_by(userId) %>% summarize('Adventure resid'=mean(resid),'Adventure n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,14] == TRUE) %>% group_by(userId) %>% summarize('Children resid'=mean(resid),'Children n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,15] == TRUE) %>% group_by(userId) %>% summarize('Fantasy resid'=mean(resid),'Fantasy n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,16] == TRUE) %>% group_by(userId) %>% summarize('War resid'=mean(resid),'War n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,17] == TRUE) %>% group_by(userId) %>% summarize('Animation resid'=mean(resid),'Animation n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,18] == TRUE) %>% group_by(userId) %>% summarize('Musical resid'=mean(resid),'Musical n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,19] == TRUE) %>% group_by(userId) %>% summarize('Crime resid'=mean(resid),'Crime n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,20] == TRUE) %>% group_by(userId) %>% summarize('Mystery resid'=mean(resid),'Mystery n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,21] == TRUE) %>% group_by(userId) %>% summarize('Film-Noir resid'=mean(resid),'Film-Noir n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,22] == TRUE) %>% group_by(userId) %>% summarize('Western resid'=mean(resid),'Western n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,23] == TRUE) %>% group_by(userId) %>% summarize('Horror resid'=mean(resid),'Horror n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,24] == TRUE) %>% group_by(userId) %>% summarize('Documentary resid'=mean(resid),'Documentary n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,25] == TRUE) %>% group_by(userId) %>% summarize('IMAX resid'=mean(resid),'IMAX n' = n()) %>% full_join(genre_resids,.,by='userId')
genre_resids = train_set_resid %>% filter(.[,26] == TRUE) %>% group_by(userId) %>% summarize('(no genres listed) resid'=mean(resid),'(no genres listed) n' = n()) %>% full_join(genre_resids,.,by='userId')


#Join that residual back to the training data, so we can use it to make a more accurate prediction
full_data=full_join(train_set_resid,genre_resids,by='userId')

#SHOULD COMPUTE WITH TRAIN SET WHAT THE RMSE IS


test_set_resid <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred1=mu+b_i,pred2 = mu + b_i + b_u) %>%
  full_join(.,genre_resids,by='userId')


#Generates a correction for a combinate of user + movie, based on the genres of the movie, for the test set.
#It will first check which genres are needed to be used for the prediction.
#Then, it computes a genre correction based on the residual for that user
#based on the genre residuals from the train set.
#These are regularized by the total number of movies that user has reviewed in a given genre n,
#by a factor of (1-1/(n+1)) such that when n is small, the weight is small, and when when n is large, the weight
#is large.
#The desired correction is the average of the weighted residuals for the genres of the movie.
  desired_corrections=0
  for (ii in 1:dim(test_set)[1]){
    needed_indices=seq(1:20)[test_set_resid[ii,c(seq(7,26))]==TRUE]
    genre_correct=0
    for (jj in 1:length(needed_indices)){
           genre_correct[jj]=(test_set_resid[ii,((needed_indices[jj])-1)*2+31]*(1-1/(test_set_resid[ii,((needed_indices[jj]-1)*2)+32]+1)))
    }

    genre_correct_mean=mean(genre_correct,na.rm=TRUE)

    desired_corrections[ii]=genre_correct_mean
    print(ii/900007)
  }
  
  #Demonstrates genre effects

  
  model_3_rmse <-   RMSE(test_set_resid$pred2[1:length(desired_corrections)]+desired_corrections, test_set$rating[1:length(desired_corrections)])
  model_3_accuracy = compute_accuracy(test_set_resid$pred2[1:length(desired_corrections)]+desired_corrections,test_set$rating[1:length(desired_corrections)])

  print(RMSE(test_set_resid$pred2[1:length(desired_corrections)]+desired_corrections, test_set$rating[1:length(desired_corrections)]))


