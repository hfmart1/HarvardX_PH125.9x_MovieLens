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



##########################################################
# Neural Network: Data transformation
##########################################################

# Mutate the timestamp to be 0 or 1 depending on the moment ratings start to have 0.5 granularity = 1045526400
edx <- edx %>% mutate(timestamp_binary = ifelse(edx$timestamp > 1045526400, 1, 0))
validation <- validation %>% mutate(timestamp_binary = ifelse(validation$timestamp > 1045526400, 1, 0))

############
# One-hot encoding of genres
############

genres <- as.data.frame(edx$genres, stringsAsFactors=FALSE)
genres_v <- as.data.frame(validation$genres, stringsAsFactors=FALSE)
# n_distinct(edx_copy$genres)
genres2 <- as.data.frame(tstrsplit(genres[,1], '[|]',
                                   type.convert=TRUE),
                         stringsAsFactors=FALSE)
genres2_v <- as.data.frame(tstrsplit(genres_v[,1], '[|]',
                                     type.convert=TRUE),
                           stringsAsFactors=FALSE)


genre_list <- c("Action", "Adventure", "Animation", "Children",
                "Comedy", "Crime","Documentary", "Drama", "Fantasy",
                "Film-Noir", "Horror", "Imax", "Musical", "Mystery","Romance",
                "Sci-Fi", "Thriller", "War", "Western") # There are 19 genres in total

genre_matrix <- matrix(0, length(edx$movieId)+1, n_distinct(genre_list))                       
genre_matrix[1,] <- genre_list #set first row to genre list

genre_matrix_v <- matrix(0, length(validation$movieId)+1, n_distinct(genre_list))                       
genre_matrix_v[1,] <- genre_list #set first row to genre list

colnames(genre_matrix) <- genre_list #set column names to genre list
colnames(genre_matrix_v) <- genre_list #set column names to genre list

#iterate through matrix
for (i in 1:nrow(genres2)) {
  for (c in 1:ncol(genres2)) {
    genmat_col <- which(genre_matrix[1,] == genres2[i,c])
    genre_matrix[i+1,genmat_col] <- 1L
  }
}

for (i in 1:nrow(genres2_v)) {
  for (c in 1:ncol(genres2_v)) {
    genmat_col <- which(genre_matrix_v[1,] == genres2_v[i,c])
    genre_matrix_v[i+1,genmat_col] <- 1L
  }
}
#convert into dataframe
genre_matrix <- as.data.frame(genre_matrix[-1,], stringsAsFactors=FALSE) #remove first row, which was the genre list
genre_matrix_v <- as.data.frame(genre_matrix_v[-1,], stringsAsFactors=FALSE)

edx_by_gen <- cbind(edx[,1:3], genre_matrix, edx$timestamp_binary) 
val_by_gen <- cbind(validation[,1:3], genre_matrix_v, validation$timestamp_binary)
colnames(edx_by_gen) <- c("userId", "movieId", "rating", genre_list, "timestamp_binary")
colnames(val_by_gen) <- c("userId", "movieId", "rating", genre_list, "timestamp_binary")
edx_by_gen <- as.matrix(sapply(edx_by_gen, as.numeric))
val_by_gen <- as.matrix(sapply(val_by_gen, as.numeric))


# remove intermediary matrices
rm(genre_matrix, genre_matrix_v, genres, genres_v, genres2, genres2_v)


# Multiply the rating by the OHE for genre
edx_by_gen_mult <- cbind(edx_by_gen[,1:2], edx_by_gen[,"rating"], sweep(edx_by_gen[,4:22], 1, edx_by_gen[,"rating"], "*"), edx_by_gen[,"timestamp_binary"])
val_by_gen_mult <- cbind(val_by_gen[,1:2], val_by_gen[,"rating"], sweep(val_by_gen[,4:22], 1, val_by_gen[,"rating"], "*"), val_by_gen[,"timestamp_binary"])


colnames(edx_by_gen_mult) <- c("userId", "movieId", "rating", "Action", "Adventure", "Animation", "Children",
                               "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                               "Film.Noir", "Horror", "Imax", "Musical", "Mystery","Romance",
                               "Sci.Fi", "Thriller", "War", "Western", "timestamp_binary")

colnames(val_by_gen_mult) <- c("userId", "movieId", "rating", "Action", "Adventure", "Animation", "Children",
                               "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                               "Film.Noir", "Horror", "Imax", "Musical", "Mystery","Romance",
                               "Sci.Fi", "Thriller", "War", "Western", "timestamp_binary")


# Transform the multiplied one-hot-encoded matrix into a user profile for genre.
user_profiles <- edx_by_gen_mult %>%
  as.data.frame() %>%
  group_by(userId) %>%
  summarise(Action_u = mean(Action),
            Adventure_u = mean(Adventure),
            Animation_u = mean(Animation),
            Children_u = mean(Children),
            Comedy_u = mean(Comedy),
            Crime_u = mean(Crime),
            Documentary_u = mean(Documentary),
            Drama_u = mean(Drama),
            Fantasy_u = mean(Fantasy),
            FilmNoir_u = mean(Film.Noir),
            Horror_u = mean(Horror),
            Imax_u = mean(Imax), 
            Musical_u = mean(Musical),
            Mystery_u = mean(Mystery),
            Romance_u = mean(Romance),
            Sci.Fi_u = mean(Sci.Fi),
            Thriller_u = mean(Thriller),
            War_u = mean(War),
            Western_u = mean(Western)) %>%
  as.data.frame()


user_profiles[is.na(user_profiles)] <- 0

# Transform the Test and Validation datasets to include the user profiles
edx_gen_norm <- edx %>%
  left_join(user_profiles, by="userId") %>%
  select(userId, 
         movieId, 
         rating, 
         Action_u, 
         Adventure_u, 
         Animation_u,
         Children_u, 
         Comedy_u,  
         Crime_u,
         Documentary_u, 
         Drama_u,
         Fantasy_u,
         FilmNoir_u,  
         Horror_u, 
         Imax_u,
         Musical_u, 
         Mystery_u, 
         Romance_u, 
         Sci.Fi_u,  
         Thriller_u,  
         War_u, 
         Western_u, 
         timestamp_binary)

val_gen_norm <- validation %>%
  left_join(user_profiles, by="userId") %>%
  select(userId, 
         movieId, 
         rating, 
         Action_u, 
         Adventure_u, 
         Animation_u,
         Children_u, 
         Comedy_u,  
         Crime_u,
         Documentary_u, 
         Drama_u,
         Fantasy_u,
         FilmNoir_u,  
         Horror_u, 
         Imax_u,
         Musical_u, 
         Mystery_u, 
         Romance_u, 
         Sci.Fi_u,  
         Thriller_u,  
         War_u, 
         Western_u, 
         timestamp_binary)


library(h2o)
h2o.init(nthreads = -1, max_mem_size = "16G")



##################
# Define the model in h2o

# turn the matrices into h2o objects
edx_h2o <- as.h2o(edx_gen_norm)
val_h2o <- as.h2o(val_gen_norm)

# Specify labels and predictors
y <- "rating"
x <- setdiff(names(edx_h2o), y)

# Turn the labels into categorical data.
edx_h2o[,y] <- as.factor(edx_h2o[,y])
val_h2o[,y] <- as.factor(val_h2o[,y])

# Train a deep learning model and validate on test set

DL_model <- h2o.deeplearning(
  x = x,
  y = y,
  training_frame = edx_h2o,
  validation_frame = val_h2o,
  distribution = "AUTO",
  activation = "RectifierWithDropout",
  hidden = c(256, 256, 256, 256),
  input_dropout_ratio = 0.2,
  sparse = TRUE,
  epochs = 15,
  stopping_rounds = 3,
  stopping_tolerance = 0.01, #stops if it doesn't improve at least 0.1%
  stopping_metric = "AUTO",
  nfolds = 10,
  variable_importances = TRUE,
  shuffle_training_data = TRUE,
  mini_batch_size = 2000
)



# Get RMSE
DL_RMSE_validation <- h2o.rmse(DL_model, valid = TRUE) # Validation RMSE = 0.8236556
DL_RMSE_training <- h2o.rmse(DL_model) # Train RMSE = 0.8241222

