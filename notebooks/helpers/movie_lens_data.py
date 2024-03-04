##-----------------------------------------------------------------------------##
#
# Uses ideas from:
#    Building Recommender Systems with Machine Learning and AI, Sundog Education
# Most of the code is written from scratch, though, and adapted to MovieLens
# dataset on Kaggle
#
##-----------------------------------------------------------------------------##


import os
import pandas as pd
import sys
import re

from surprise import Dataset
from surprise import Reader

from collections import defaultdict
import numpy as np


class MovieLensData:
    """
    Movie Lens Data
    """

    def __init__(self, users_path, ratings_path, movies_path, genre_path):
        self.users_path = users_path
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.genre_path = genre_path

    def read_user_data(self):
        """
        read user data, set user_data
        """
        user_columns = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        self.user_data = pd.read_csv(self.users_path, sep='|', names=user_columns)
        return self.user_data

    def read_ratings_data(self):
        """
        read ratings data, set ratings_data

        """
        ratings_columns = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        ratings_df = pd.read_csv(self.ratings_path, sep='\t', names=ratings_columns)
        ratings_df.drop("unix_timestamp", inplace=True, axis=1)
        self.ratings_data_df = ratings_df
        reader = Reader(rating_scale=(1, 5))
        self.ratings_data = Dataset.load_from_df(ratings_df, reader=reader)

        return self.ratings_data

    def clean_title(self, title):
        """
        auxiliary function for readings movie data
        """
        return re.sub("[\(\[].*?[\)\]]", "", title)

    def process_genre(self, series):
        """
        auxiliary function for readings movie data
        """
        genres = series.index[6:-2]
        text = []
        for i in genres:
            if series[i] == 1:
                text.append(i)
                break
        return ", ".join(text)

    def read_movies_data(self):
        """
        read movies data, set movie_data

        """
        self.movie_id_to_name = {}
        self.name_to_movie_id = {}

        genre_df = pd.read_csv(self.genre_path, sep='|', encoding='latin-1')
        genre_columns = ["unknown"] + list(genre_df[genre_df.columns[0]].values)

        movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
        self.movie_data = pd.read_csv(self.movies_path, sep='|', names=movie_columns + genre_columns,
                                      encoding='latin-1')
        self.movie_data['title'] = self.movie_data['title'].apply(self.clean_title)
        self.movie_data['title'] = self.movie_data['title'].str.strip()
        self.movie_data['genre'] = self.movie_data.apply(self.process_genre, axis=1)

        for index, row in self.movie_data.iterrows():
            movie_id = int(row['movie_id'])
            movie_name = row['title']
            self.movie_id_to_name[movie_id] = movie_name
            self.name_to_movie_id[movie_name] = movie_id

        return self.movie_data

    def get_user_ratings(self, user):
        """
        select ratings for a certain user
        Args
            user: user for which to return the ratings
        Returns
            the ratings for a certain user
        """
        user_ratings = []
        hit_user = False
        user_ratings = self.ratings_data_df.loc[self.ratings_data_df.user_id == user]
        user_ratings = user_ratings[['movie_id', 'rating']]

        return user_ratings

    def get_ratings(self):
        return self.ratings_data_df

    def get_popularity_ranks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        for index, row in self.ratings_data_df.iterrows():
            movie_id = int(row['movie_id'])
            ratings[movie_id] += 1
        rank = 1
        for movie_id, rating_count in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[movie_id] = rank
            rank += 1
        return rankings

    def get_movie_name(self, movie_id):
        if movie_id in self.movie_id_to_name:
            return self.movie_id_to_name[movie_id]
        else:
            return ""

    def get_movie_id(self, movie_name):
        if movie_name in self.name_to_movie_id:
            return self.name_to_movie_id[movie_name]
        else:
            return 0


import torch
class MovieLensDataset(torch.utils.data.Dataset):
    """
    MovieLens Dataset
    Data preparation
        treat samples with a rating less than 3 as negative samples
    """
    def __init__(self, ratings):
        data = ratings.copy().to_numpy()
        self.items = data[:, :2].astype(np.int32) - 1  # -1 because ID begins from 1
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0, ), dtype=np.int64)
        self.item_field_idx = np.array((1,), dtype=np.int64)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target = target / 5.
        return target