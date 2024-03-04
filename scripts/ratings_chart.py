import matplotlib.pyplot as plt
import pickle

user_ratings = pickle.load(open('../pickles/user_ratings2.pkl', 'rb'))

temp = []
for key in user_ratings.values():
    for value_list in key.values():
        temp.append(value_list)

print(len(temp))

plt.hist(temp, bins=20)
plt.show()
