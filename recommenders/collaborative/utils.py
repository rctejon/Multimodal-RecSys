import pandas as pd


def create_ratings_df(user_ratings) -> pd.DataFrame:
    temp = []
    for user, courses in user_ratings.items():
        for course, rating in courses.items():
            temp.append([user, course, rating])
    ratings_df = pd.DataFrame(temp, columns=['user', 'course', 'rating'])
    return ratings_df
