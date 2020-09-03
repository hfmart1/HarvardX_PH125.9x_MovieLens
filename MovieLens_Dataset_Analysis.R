##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(caret)
library(data.table)
library(randomForest)
library(dplyr)


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")


# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###################################################################################################
# DATA ANALYSIS ON CLEAN DATASET
###################################################################################################

# Dimension edx dataset
dim(edx)

# Number of ratings
table(edx$rating)

# Number of unique movies
n_distinct(edx$movieId)

# Number of different users
n_distinct(edx$userId)

# How many movie ratings are in each genre in the edx dataset?
genrecount <- edx %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarise(number = n()) %>%
  arrange(desc(number))

genrecount

#Barplot genrecount

barplot(genrecount$number,
        main = "Number of movies per genre",
        names.arg = genrecount$genres)


# Which movie has the greatest number of ratings?
num_rating <- edx %>%
  group_by(title) %>%
  summarise(number = n()) %>%
  arrange(desc(number))

# Number of ratings
barplot(table(edx$rating))


#########################################################################################
# PREDICTION RATING
#########################################################################################

# Define loss function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# RMSE (25 points)
# 
# 0 points: No RMSE reported AND/OR code used to generate the RMSE appears to violate the edX Honor Code.
# 5 points: RMSE >= 0.90000 AND/OR the reported RMSE is the result of overtraining (validation set - the final hold-out test set - ratings used for anything except reporting the final RMSE value) AND/OR the reported RMSE is the result of simply copying and running code provided in previous courses in the series.
# 10 points: 0.86550 <= RMSE <= 0.89999
# 15 points: 0.86500 <= RMSE <= 0.86549
# 20 points: 0.86490 <= RMSE <= 0.86499
# 25 points: RMSE < 0.86490

########################
# FIRST APPROACH
########################

# Lets suppose that all movies receive the same rating, what would that be?

# Average rating for movies
mu_hat <- mean(edx$rating)

# RMSE average rating
rmse_1 <- RMSE(validation$rating, mu_hat)

########################
# SECOND APPROACH
########################

# However, we notice that some movies seem to have higher average ratings. Lets improve our prediction using this information.

# Compute difference to average per movie
movie_averages <- edx %>% 
  group_by(movieId) %>% 
  summarise(b_i = mean(rating - mu_hat))

predicted_ratings <- mu_hat + validation %>% 
  left_join(movie_averages, by='movieId') %>%
  pull(b_i)

# RMSE movie effect
rmse_2 <- RMSE(predicted_ratings, validation$rating)
rmse_2


# This is still too high, lets use data to improve our approach

edx %>%
  group_by(userId) %>%
  summarise(a_u = mean(rating)) %>%
  filter(n()>=100) %>%
  ggplot(aes(a_u)) +
  geom_histogram(bins = 30, color = "black")

# We notice that some users tend to give more positive or negative reviews. We will call this phenomenon the user-specific effect


user_averages <- edx %>%
  left_join(movie_averages, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu_hat - b_i))


predicted_ratings <- validation %>% 
  left_join(movie_averages, by='movieId') %>%
  left_join(user_averages, by='userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)



# RMSE user effect
rmse_3 <- RMSE(predicted_ratings, validation$rating)
rmse_3



# Rounding
predicted_ratings_limit <- pmax(pmin(predicted_ratings, 5),     0.5)  #limit values lower than 0.5 & values greater than 5

rmse_4 <- RMSE(predicted_ratings_limit, validation$rating)
rmse_4



# Results
rmse_results <- tibble(method = c("Overall average", "Movie effect","Movie + User effect","With limits"), RMSE = c(rmse_1, rmse_2, rmse_3, rmse_4))

rmse_results


########################
# THIRD APPROACH
########################

#In setting the movie penalty, we will try figures from 0 to 60, increasing by increments of 2.
penalties <- seq(0, 3, 1)

m_rmses <- sapply(penalties, function(p){
  reg_movie_avgs <- edx %>% 
    group_by(movieId) %>%
    summarize(regmoviebias = sum(rating - mu_hat)/(n()+p))
  
  predicted_ratings <- 
    edx %>% 
    left_join(reg_movie_avgs, by = "movieId") %>%
    left_join(user_averages, by = "userId") %>%
    mutate(regmoviebias = mu_hat + b_u + regmoviebias) %>%
    .$regmoviebias
  return(RMSE(predicted_ratings, edx$rating))
})

moviepenalty_optimal <- penalties[which.min(m_rmses)]  #determine which is lowest

reg_movie_avgs <- edx %>% 
  group_by(movieId) %>%
  summarize(regmoviebias = sum(rating - mu_hat)/(n()+moviepenalty_optimal))


#Check movie bias against test set
predicted_ratings <- 
  validation %>% 
  left_join(reg_movie_avgs, by = "movieId") %>%
  left_join(user_averages, by = "userId") %>%
  mutate(regmoviebias = mu_hat + b_u + regmoviebias) %>%
  .$regmoviebias


regularized_movieeffects <- RMSE(predicted_ratings, validation$rating)
regularized_movieeffects


#Now let's do the same for user penalties.
penalties <- seq(0, 1, 0.25)

u_rmses <- sapply(penalties, function(p){
  reg_user_avgs <- edx %>% 
    left_join(reg_movie_avgs, by="movieId") %>%
    group_by(userId) %>%
    summarize(reguserbias = sum(rating - regmoviebias - mu_hat)/(n()+p))
  
  predicted_ratings <- 
    edx %>% 
    left_join(reg_movie_avgs, by = "movieId") %>%
    left_join(reg_user_avgs, by = "userId") %>%
    mutate(regusermoviebias = mu_hat + regmoviebias + reguserbias) %>%
    .$regusermoviebias
  return(RMSE(predicted_ratings, edx$rating))
})

userpenalty_optimal <- penalties[which.min(u_rmses)]  #determine which is lowest


#Let's check both penalty values against the *test* set and see how it affects our RMSE.
#build table with optimal penalty    

reg_user_avgs <- edx %>% 
  left_join(reg_movie_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(reguserbias = sum(rating - regmoviebias - mu_hat)/(n()+userpenalty_optimal))




reg_predicted_ratings <- 
  validation %>% 
  left_join(reg_movie_avgs, by = "movieId") %>%
  left_join(reg_user_avgs, by = "userId") %>%
  mutate(regusermovie = mu_hat + regmoviebias + reguserbias) %>%
  .$regusermovie


regularized_effects <- RMSE(reg_predicted_ratings, validation$rating)


reg_predicted_ratings_limit <- pmax(pmin(predicted_ratings, 5),     0.5)  #limit values lower than 0.5 & values greater than 5

regularized_effects_limit <- RMSE(reg_predicted_ratings_limit, validation$rating)


########################
# FOURTH APPROACH
########################

library(tensorflow)
install_tensorflow()
library(keras)
keras::install_keras()

hello <- tf$constant("Hello")
print(hello)


train_features <- edx%>%
  mutate(
  temp = str_extract(title, regex(   "\\((\\d{4})\\)"   )),   #extract the year of release in brackets
  release_yr = str_extract(temp, regex(   "(\\d{4})"   )),     #remove the brackets and...
  release_yr = as.numeric(release_yr)                          #...convert to a number
) %>%
  select(-everything(), movieId, userId, release_yr)

train_labels <- edx %>%
  select(rating)

test_features <- edx%>%
  mutate(
    temp = str_extract(title, regex(   "\\((\\d{4})\\)"   )),   #extract the year of release in brackets
    release_yr = str_extract(temp, regex(   "(\\d{4})"   )),     #remove the brackets and...
    release_yr = as.numeric(release_yr)                          #...convert to a number
  ) %>%
  select(-everything(), movieId, userId, release_yr)

test_labels <- edx %>%
  select(rating)

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 3, activation = 'relu', input_shape = 3) %>% 
  layer_dense(units = 1, activation = 'softmax')

model %>% summary


model %>% compile(
  loss      = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics   = c('accuracy')
)

history <- model %>% fit(
  x = train_features, y = train_labels,
  epochs           = 200,
  batch_size       = 20,
  validation_split = 0
)
plot(history)














#####################################"


############################ TEST2

clean_edx <- edx %>%
  mutate(
    temp = str_extract(title, regex(   "\\((\\d{4})\\)"   )),   #extract the year of release in brackets
    release_yr = str_extract(temp, regex(   "(\\d{4})"   )),     #remove the brackets and...
    release_yr = as.numeric(release_yr)                          #...convert to a number
  ) %>%
  select(-everything(), movieId, userId, rating, release_yr)


trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

knn_fit <- train(rating ~., 
                 data = clean_edx,
                 method = "knn",
                 trControl=trctrl,
                 preProcess = c("center", "scale"),
                 tuneLength = 10)

confusionMatrix(predict(train_knn, mnist_27$test, type = "raw"),
                mnist_27$test$y)$overall["Accuracy"]



#####################################"



age <- edx %>%
  mutate(
    temp = str_extract(title, regex(   "\\((\\d{4})\\)"   )),   #extract the year of release in brackets
    release_yr = str_extract(temp, regex(   "(\\d{4})"   )),     #remove the brackets and...
    release_yr = as.numeric(release_yr)                          #...convert to a number
  ) %>%
  select(-everything(), movieId, rating, release_yr)

age

age %>%
  group_by(release_yr) %>%
  ggplot (aes(x=release_yr, y=mean(rating))) +
  geom_point()


# Compute difference to average per movie
year_averages <- age %>%
  group_by(movieId) %>%
  summarise(b_y = mean(rating - mu_hat))

predicted_ratings <- validation %>%
  left_join(movie_averages, by='movieId') %>%
  left_join(user_averages, by='userId') %>%
  left_join(year_averages, by="movieId") %>%
  mutate(pred = mu_hat + b_i + b_u + 0*b_y) %>%
  pull(pred)

predicted_ratings_limit <- pmax(pmin(predicted_ratings, 5),     0.5)  #limit values lower than 0.5 & values greater than 5


# RMSE movie effect
rmse_5 <- RMSE(predicted_ratings_limit, validation$rating)
rmse_5






age %>%
  group_by(release_yr) %>%
  summarize(  n = n(), 				sd = sd(rating) ,		se  = sd/sqrt(n) , 		avg = mean(rating) 				) %>%
  
  ggplot (aes(x=release_yr, y=avg)) +
  geom_point() +
  geom_errorbar(aes(ymin=avg-se, ymax=avg+se), width=0.4, colour="red", alpha=0.8, size=1.3) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  geom_hline(yintercept = mu_hat)
















# edx %>% group_by(releaseyear) %>%
#   summarize(rating = mean(rating)) %>%
#   ggplot(aes(releaseyear, rating)) +
#   geom_point() +
#   theme_hc() + 
#   geom_smooth() +
#   ggtitle("Release Year vs. Rating")