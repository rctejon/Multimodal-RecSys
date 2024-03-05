import pickle
from utils import create_ratings_df
from turicreate import SFrame
import turicreate as tc

user_ratings = create_ratings_df(pickle.load(open('pickles/user_ratings5.pkl', 'rb')))
user_ratings = SFrame(data=user_ratings)

train, test = tc.recommender.util.random_split_by_user(user_ratings, 'user', 'course')

f_model = tc.factorization_recommender.create(train, user_id='user', item_id='course', target='rating')
rf_model = tc.ranking_factorization_recommender.create(train, user_id='user', item_id='course', target='rating')
tc.recommender.util.compare_models(test, [f_model, rf_model])
