#########################################################################################
# IMPORT LIBRARIES
#########################################################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(dplyr, warn.conflicts = FALSE)

# Suppress summarise info
options(dplyr.summarise.inform = FALSE)

#########################################################################################
# Create edx set, validation set (final hold-out test set)
#########################################################################################

# Note: this process could take a couple of minutes

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
set.seed(2, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
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

#########################################################################################
# DATA ANALYSIS ON CLEAN DATASET (edX Quiz)
#########################################################################################

# # Dimension edx dataset
# dim(edx)
# 
# # Number of ratings
# table(edx$rating)
# 
# # Number of unique movies
# n_distinct(edx$movieId)
# 
# # Number of different users
# n_distinct(edx$userId)
# 
# # How many movie ratings are in each genre in the edx dataset?
# genrecount <- edx %>%
#   separate_rows(genres, sep = "\\|") %>%
#   group_by(genres) %>%
#   summarise(number = n()) %>%
#   arrange(desc(number))
# 
# genrecount
# 
# #Barplot genrecount
# 
# barplot(genrecount$number,
#         main = "Number of movies per genre",
#         names.arg = genrecount$genres)
# 
# 
# # Which movie has the greatest number of ratings?
# num_rating <- edx %>%
#   group_by(title) %>%
#   summarise(number = n()) %>%
#   arrange(desc(number))
# 
# # Number of ratings
# barplot(table(edx$rating))

#########################################################################################
# PREDICTION RATING
#########################################################################################

# Define loss function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# RMSE OBJECTTIVES
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
rmse_1

########################
# SECOND APPROACH
########################

# We know that certain movies have higher ratings than others, we will include in our prediction the movie bias.

# Compute difference to average per movie
movie_averages <- edx %>% 
  group_by(movieId) %>% 
  summarise(b_m = mean(rating - mu_hat))

# Compute the predicted rating using the movie bias
predicted_ratings <- mu_hat + validation %>% 
  left_join(movie_averages, by='movieId') %>%
  pull(b_m)

# RMSE movie bias
rmse_2 <- RMSE(predicted_ratings, validation$rating)
rmse_2


# This is still too high, lets use data to improve our approach

# Plot average rating for users with over 100 ratings
edx %>%
  group_by(userId) %>%
  summarise(a_u = mean(rating)) %>%
  filter(n()>=100) %>%
  ggplot(aes(a_u)) +
  geom_histogram(bins = 30, color = "black")

# We notice that some users tend to give more positive or negative reviews, the user bias.

# Calculate user rating average and compute user bias with movie bias for each movie
user_averages <- edx %>%
  left_join(movie_averages, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu_hat - b_m))

# Compute predicted ratings including movie and user bias
predicted_ratings <- validation %>% 
  left_join(movie_averages, by='movieId') %>%
  left_join(user_averages, by='userId') %>%
  mutate(pred = mu_hat + b_m + b_u) %>%
  pull(pred)


# RMSE user effect
rmse_3 <- RMSE(predicted_ratings, validation$rating)
rmse_3

# By observing the predicted ratings we notice that some of them are lower than 0.5 or greater than 5.

# Limit
predicted_ratings_limit <- pmax(pmin(predicted_ratings, 5),     0.5)  #limit values lower than 0.5 & values greater than 5

# Calculate RMSE
rmse_4 <- RMSE(predicted_ratings_limit, validation$rating)
rmse_4



# Results
rmse_results <- tibble(method = c("Overall average", "Movie effect","Movie + User effect","With limits"), RMSE = c(rmse_1, rmse_2, rmse_3, rmse_4))

rmse_results


########################
# THIRD APPROACH
########################

# We now use regularization to improve our prediction

#In setting the movie penalty, we will create a list between 0 and 10 with increments of 1
penalties <- seq(0, 10, 1)


# Compute RMSE on edx dataset to set best penalty
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

# Determine the lowest penalty
moviepenalty_optimal <- penalties[which.min(m_rmses)] 

# Use best penalty to compute movie averages
reg_movie_avgs <- edx %>% 
  group_by(movieId) %>%
  summarize(regmoviebias = sum(rating - mu_hat)/(n()+moviepenalty_optimal))


# Compute prediction for validation dataset using the penalties calculated above
predicted_ratings <- 
  validation %>% 
  left_join(reg_movie_avgs, by = "movieId") %>%
  left_join(user_averages, by = "userId") %>%
  mutate(regmoviebias = mu_hat + b_u + regmoviebias) %>%
  .$regmoviebias

# Compute RMSE for the validation dataset
regularized_movieeffects <- RMSE(predicted_ratings, validation$rating)
regularized_movieeffects


#We do the same for the user bias, we use regularization

# Set penalties to test
penalties <- seq(0, 1, 0.25)

# Compute RMSE on edx dataset to set best penalty
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

# Determine the lowest penalty
userpenalty_optimal <- penalties[which.min(u_rmses)]  #determine which is lowest


# Use best penalty to compute movie averages using both the movie and user bias regularization
reg_user_avgs <- edx %>% 
  left_join(reg_movie_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(reguserbias = sum(rating - regmoviebias - mu_hat)/(n()+userpenalty_optimal))


# Compute prediction for validation dataset using the penalties calculated above
reg_predicted_ratings <- 
  validation %>% 
  left_join(reg_movie_avgs, by = "movieId") %>%
  left_join(reg_user_avgs, by = "userId") %>%
  mutate(regusermovie = mu_hat + regmoviebias + reguserbias) %>%
  .$regusermovie

# Compute RMSE
regularized_effects <- RMSE(reg_predicted_ratings, validation$rating)

# Fix lower and upper limits
reg_predicted_ratings_limit <- pmax(pmin(predicted_ratings, 5),     0.5)  

regularized_effects_limit <- RMSE(reg_predicted_ratings_limit, validation$rating)
regularized_effects_limit