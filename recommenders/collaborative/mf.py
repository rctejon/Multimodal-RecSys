import pickle
from utils import create_ratings_df
from turicreate import SFrame

user_ratings = create_ratings_df(pickle.load(open('pickles/user_ratings5.pkl', 'rb')))
user_ratings = SFrame(data=user_ratings)

print(user_ratings)
