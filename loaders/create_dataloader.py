from loaders.rating_dataset import RatingDataset
from torch.utils.data import DataLoader
import random
import pandas as pd
from tqdm import tqdm

class CreateDataloader(object):
	"""
	Construct Dataloaders
	"""
	def __init__(self, args, train_ratings, test_ratings):
		self.train_ratings = train_ratings
		self.test_ratings = test_ratings
		self.ratings = pd.concat([train_ratings, test_ratings], ignore_index=True)
		self.num_ng = args.num_ng
		self.num_ng_test = args.num_ng_test
		self.batch_size = args.batch_size

		self.user_pool = set(self.ratings['user_id'].unique())
		self.item_pool = set(self.ratings['item_id'].unique())

		print('negative sampling')
		self.negatives = self._negative_sampling(self.ratings)
		print('done')
		random.seed(args.seed)


	def _negative_sampling(self, ratings):
		interact_status = (
			ratings.groupby('user_id')['item_id']
			.apply(set)
			.reset_index()
			.rename(columns={'item_id': 'interacted_items'}))
		interact_status['train_negative_samples'] = interact_status['interacted_items'].apply(lambda x: random.sample(self.item_pool - x, self.num_ng))
		interact_status['test_negative_samples'] = interact_status['interacted_items'].apply(lambda x: random.sample(self.item_pool - x, self.num_ng_test))
		return interact_status[['user_id', 'train_negative_samples', 'test_negative_samples']]

	def get_train_instance(self):
		users, items, ratings = [], [], []
		train_ratings = pd.merge(self.train_ratings, self.negatives[['user_id', 'train_negative_samples']], on='user_id')
		for row in tqdm(train_ratings.itertuples(), total=train_ratings.shape[0]):
			users.append(int(row.user_id))
			items.append(int(row.item_id))
			ratings.append(float(row.rating))
			for i in getattr(row, 'train_negative_samples'):
				users.append(int(row.user_id))
				items.append(int(i))
				ratings.append(float(0))
		dataset = RatingDataset(
			user_list=users,
			item_list=items,
			rating_list=ratings)
		return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

	def get_test_instance(self):
		users, items, ratings = [], [], []
		test_ratings = pd.merge(self.test_ratings, self.negatives[['user_id', 'test_negative_samples']], on='user_id')
		for row in tqdm(test_ratings.itertuples(), total= test_ratings.shape[0]):
			users.append(int(row.user_id))
			items.append(int(row.item_id))
			ratings.append(float(row.rating))
			for i in getattr(row, 'test_negative_samples'):
				users.append(int(row.user_id))
				items.append(int(i))
				ratings.append(float(0))
		dataset = RatingDataset(
			user_list=users,
			item_list=items,
			rating_list=ratings)
		return DataLoader(dataset, batch_size=self.num_ng_test+1, shuffle=False, num_workers=2)