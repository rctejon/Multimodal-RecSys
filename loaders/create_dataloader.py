from loaders.rating_dataset import RatingDataset
from torch.utils.data import DataLoader
import torch
import random
import pandas as pd
from tqdm import tqdm
import pickle
import os

class CreateDataloader(object):
	"""
	Construct Dataloaders
	"""
	def __init__(self, args, train_ratings, test_ratings, dataset_path, with_text=False, texts=None):
		self.train_ratings = train_ratings
		self.test_ratings = test_ratings
		self.ratings = pd.concat([train_ratings, test_ratings], ignore_index=True)
		print(train_ratings.shape, test_ratings.shape, self.ratings.shape)
		self.num_ng = args.num_ng
		self.num_ng_test = args.num_ng_test
		self.batch_size = args.batch_size
		self.dataset_path = dataset_path
		self.with_text = with_text

		if self.with_text and not os.path.exists(f'{self.dataset_path}/test_texts_{self.num_ng}.pkl'):
			self.texts = texts

		self.user_pool = set(self.ratings['user_id'].unique())
		self.item_pool = set(self.ratings['item_id'].unique())

		if not os.path.exists(f'{self.dataset_path}/test_users_{self.num_ng}.pkl'):
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
	
	
	def collate_fn(self, batch):
		return (
			torch.stack([x[0] for x in batch]),
			torch.stack([x[1] for x in batch]),
			torch.stack([x[2] for x in batch]),
			[x[3] for x in batch] if self.with_text else None,
		)

	def get_train_instance(self):
		if not os.path.exists (f'{self.dataset_path}/train_users_{self.num_ng}.pkl'):
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
			pickle.dump(users, open(f'{self.dataset_path}/train_users_{self.num_ng}.pkl', 'wb'))
			pickle.dump(items, open(f'{self.dataset_path}/train_items_{self.num_ng}.pkl', 'wb'))
			pickle.dump(ratings, open(f'{self.dataset_path}/train_ratings_{self.num_ng}.pkl', 'wb'))
		else:
			users = pickle.load(open(f'{self.dataset_path}/train_users_{self.num_ng}.pkl', 'rb'))
			items = pickle.load(open(f'{self.dataset_path}/train_items_{self.num_ng}.pkl', 'rb'))
			ratings = pickle.load(open(f'{self.dataset_path}/train_ratings_{self.num_ng}.pkl', 'rb'))

		if self.with_text:
			text_list = self._get_train_texts()
			dataset = RatingDataset(
				user_list=users,
				item_list=items,
				rating_list=ratings,
				text_list=text_list)
		else:
			dataset = RatingDataset(
				user_list=users,
				item_list=items,
				rating_list=ratings)
			
		return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=self.collate_fn)
	
	def _get_train_texts(self):
		texts = []
		if not os.path.exists(f'{self.dataset_path}/train_texts_{self.num_ng}.pkl'):
			items = pickle.load(open(f'{self.dataset_path}/train_items_{self.num_ng}.pkl', 'rb'))
			for item in tqdm(items, total=len(items)):
				texts.append(self.texts.iloc[item]['text'])
			pickle.dump(texts, open(f'{self.dataset_path}/train_texts_{self.num_ng}.pkl', 'wb'))
		else:
			texts = pickle.load(open(f'{self.dataset_path}/train_texts_{self.num_ng}.pkl', 'rb'))
		return texts


	def get_test_instance(self):
		if not os.path.exists(f'{self.dataset_path}/test_users_{self.num_ng_test}.pkl'):
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
			pickle.dump(users, open(f'{self.dataset_path}/test_users_{self.num_ng_test}.pkl', 'wb'))
			pickle.dump(items, open(f'{self.dataset_path}/test_items_{self.num_ng_test}.pkl', 'wb'))
			pickle.dump(ratings, open(f'{self.dataset_path}/test_ratings_{self.num_ng_test}.pkl', 'wb'))
		else:
			users = pickle.load(open(f'{self.dataset_path}/test_users_{self.num_ng_test}.pkl', 'rb'))
			items = pickle.load(open(f'{self.dataset_path}/test_items_{self.num_ng_test}.pkl', 'rb'))
			ratings = pickle.load(open(f'{self.dataset_path}/test_ratings_{self.num_ng_test}.pkl', 'rb'))

		if self.with_text:
			text_list = self._get_test_texts()
			dataset = RatingDataset(
				user_list=users,
				item_list=items,
				rating_list=ratings,
				text_list=text_list)
		else:
			dataset = RatingDataset(
				user_list=users,
				item_list=items,
				rating_list=ratings)
		
		return DataLoader(dataset, batch_size=self.num_ng_test+1, shuffle=False, num_workers=12, collate_fn=self.collate_fn)
	
	def _get_test_texts(self):
		texts = []
		if not os.path.exists(f'{self.dataset_path}/test_texts_{self.num_ng_test}.pkl'):
			items = pickle.load(open(f'{self.dataset_path}/test_items_{self.num_ng_test}.pkl', 'rb'))
			for item in tqdm(items, total=len(items)):
				texts.append(self.texts.iloc[item]['text'])
			pickle.dump(texts, open(f'{self.dataset_path}/test_texts_{self.num_ng_test}.pkl', 'wb'))
		else:
			texts = pickle.load(open(f'{self.dataset_path}/test_texts_{self.num_ng_test}.pkl', 'rb'))
		return texts