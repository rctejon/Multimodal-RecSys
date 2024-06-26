from loaders.rating_dataset import RatingDataset
from torch.utils.data import DataLoader
from transformers import BatchEncoding
import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os

class CreateDataloader(object):
	"""
	Construct Dataloaders
	"""
	def __init__(self, args, train_ratings, test_ratings, dataset_path, with_text=False, tokenizations=None, graph_embeddings=None, use_item_embedding=True):
		self.num_ng = args.num_ng
		self.num_ng_test = args.num_ng_test
		self.batch_size = args.batch_size
		self.dataset_path = dataset_path
		self.with_text = with_text
		self.train_bert = args.train_bert
		self.use_item_embedding = use_item_embedding

		self.NUM_USERS = 694529
		# self.NUM_USERS = 2942027

		self.graph_embeddings = graph_embeddings

		# Load the embeddings if the model requires it
		if self.with_text:
			self.token_size = args.token_size
			if os.path.exists(f'{self.dataset_path}/paper_embeddings.npy'):
				self.embeddings = np.load(f'{self.dataset_path}/paper_embeddings.npy')

		# We need to do the negative sampling just once, then is store in a pickle file
		if not os.path.exists(f'{self.dataset_path}/test_users_{self.num_ng_test}.pkl'):
			self.train_ratings = train_ratings
			self.test_ratings = test_ratings
			self.ratings = pd.concat([train_ratings, test_ratings], ignore_index=True)
			self.user_pool = set(self.ratings['user_id'].unique())
			self.item_pool = set(self.ratings['item_id'].unique())
			print('negative sampling')
			self.negatives = self._negative_sampling(self.ratings)
			print('done')
			
		random.seed(args.seed)

		# Load the tokenizations if the model trains the BERT model
		if not self.train_bert:
			if self.with_text and not os.path.exists(f'{self.dataset_path}/test_embeddings_{self.num_ng}_{self.token_size}.pkl'):
				self.tokenizations = tokenizations
		else:
			if self.with_text and not os.path.exists(f'{self.dataset_path}/test_tokenizations_{self.num_ng_test}_{self.token_size}.pkl'):
				self.tokenizations = tokenizations

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
		
		# MOOCCubeX
		# encoded_inputs = torch.cat([x[3] for x in batch]) if not self.train_bert and self.with_text else None			
			
		if self.with_text and self.train_bert:
			input_ids = torch.cat(tuple(map(lambda x: x[3]['input_ids'], batch)), dim=0)
			attention_mask = torch.cat(tuple(map(lambda x: x[3]['attention_mask'], batch)), dim=0)
			token_type_ids = torch.cat(tuple(map(lambda x: x[3]['token_type_ids'], batch)), dim=0)

			encoded_inputs = BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})
		if self.graph_embeddings is not None:
			if self.use_item_embedding:
				graph_embeddings = torch.zeros((len(batch), self.graph_embeddings.shape[1] * 2))
			else:
				graph_embeddings = torch.zeros((len(batch), self.graph_embeddings.shape[1]))
			if self.with_text and not self.train_bert:
				encoded_inputs = torch.zeros((len(batch), 768))
			for i, x in enumerate(batch):
				user = x[0].item()
				if not self.train_bert and self.with_text:
					if user < self.NUM_USERS:
						encoded_inputs[i] = torch.tensor(self.embeddings[user])
				if self.use_item_embedding:
					item = x[1].item() + self.NUM_USERS
					graph_embeddings[i] = torch.tensor(np.concatenate([self.graph_embeddings[user], self.graph_embeddings[item]]))
				else:
					graph_embeddings[i] = torch.tensor(self.graph_embeddings[user])
		elif self.with_text and not self.train_bert:
			encoded_inputs = torch.zeros((len(batch), 768))
			for i, x in enumerate(batch):
				user = x[0].item()
				if user < self.NUM_USERS:
					# print(user, self.embeddprings[user].shape)
					encoded_inputs[i] = torch.tensor(self.embeddings[user])


		return (
			torch.stack([x[0] for x in batch]),
			torch.stack([x[1] for x in batch]),
			torch.stack([x[2] for x in batch]),
			encoded_inputs if self.with_text else None,
			graph_embeddings if self.graph_embeddings is not None else None
		)

	def get_train_instance(self):
		if not os.path.exists (f'{self.dataset_path}/train_users_{self.num_ng}.npy'):
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
			users = np.load(f'{self.dataset_path}/train_users_{self.num_ng}.npy')
			items = np.load(f'{self.dataset_path}/train_items_{self.num_ng}.npy')
			ratings = np.load(f'{self.dataset_path}/train_ratings_{self.num_ng}.npy')

		if self.with_text:
			# tokenization_list = self._get_train_tokenizations()
			tokenization_list = None
			dataset = RatingDataset(
				user_list=users,
				item_list=items,
				rating_list=ratings,
				tokenization_list=tokenization_list)
		else:
			dataset = RatingDataset(
				user_list=users,
				item_list=items,
				rating_list=ratings)
			
		return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=self.collate_fn)
	
	def _get_train_tokenizations(self):
		tokenizations = []
		if not self.train_bert:
			if not os.path.exists(f'{self.dataset_path}/train_embeddings_{self.num_ng}_{self.token_size}.pkl'):
				items = pickle.load(open(f'{self.dataset_path}/train_items_{self.num_ng}.pkl', 'rb'))
				for item in tqdm(items, total=len(items)):
					tokenizations.append(self.tokenizations.iloc[item]['embedding'])
				pickle.dump(tokenizations, open(f'{self.dataset_path}/train_embeddings_{self.num_ng}_{self.token_size}.pkl', 'wb'))
			else:
				tokenizations = pickle.load(open(f'{self.dataset_path}/train_embeddings_{self.num_ng}_{self.token_size}.pkl', 'rb'))

		else:
			if not os.path.exists(f'{self.dataset_path}/train_tokenizations_{self.num_ng}_{self.token_size}.pkl'):
				items = pickle.load(open(f'{self.dataset_path}/train_items_{self.num_ng}.pkl', 'rb'))
				for item in tqdm(items, total=len(items)):
					tokenizations.append(self.tokenizations.iloc[item]['tokenization'])
				pickle.dump(tokenizations, open(f'{self.dataset_path}/train_tokenizations_{self.num_ng}_{self.token_size}.pkl', 'wb'))
			else:
				tokenizations = pickle.load(open(f'{self.dataset_path}/train_tokenizations_{self.num_ng}_{self.token_size}.pkl', 'rb'))
		return tokenizations


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
			print('loading test data')
			users = np.array(pickle.load(open(f'{self.dataset_path}/test_users_{self.num_ng_test}.pkl', 'rb')))
			print('done loading test users')
			items = np.array(pickle.load(open(f'{self.dataset_path}/test_items_{self.num_ng_test}.pkl', 'rb')))
			print('done loading test items')
			ratings = np.array(pickle.load(open(f'{self.dataset_path}/test_ratings_{self.num_ng_test}.pkl', 'rb')))
			print('done loading test ratings')

		if self.with_text:
			# tokenization_list = self._get_test_tokenizations()
			tokenization_list = None
			dataset = RatingDataset(
				user_list=users,
				item_list=items,
				rating_list=ratings,
				tokenization_list=tokenization_list)
		else:
			dataset = RatingDataset(
				user_list=users,
				item_list=items,
				rating_list=ratings)
		
		return DataLoader(dataset, batch_size=self.num_ng_test+1, shuffle=False, num_workers=0, collate_fn=self.collate_fn)
	
	def _get_test_tokenizations(self):
		tokenizations = []
		if not self.train_bert:
			if not os.path.exists(f'{self.dataset_path}/test_embeddings_{self.num_ng_test}_{self.token_size}.pkl'):
				items = pickle.load(open(f'{self.dataset_path}/test_items_{self.num_ng_test}.pkl', 'rb'))
				for item in tqdm(items, total=len(items)):
					tokenizations.append(self.tokenizations.iloc[item]['embedding'])
				pickle.dump(tokenizations, open(f'{self.dataset_path}/test_embeddings_{self.num_ng_test}_{self.token_size}.pkl', 'wb'))
			else:
				tokenizations = pickle.load(open(f'{self.dataset_path}/test_embeddings_{self.num_ng_test}_{self.token_size}.pkl', 'rb'))
		else:
			if not os.path.exists(f'{self.dataset_path}/test_tokenizations_{self.num_ng_test}_{self.token_size}.pkl'):
				items = pickle.load(open(f'{self.dataset_path}/test_items_{self.num_ng_test}.pkl', 'rb'))
				for item in tqdm(items, total=len(items)):
					tokenizations.append(self.tokenizations.iloc[item]['tokenization'])
				pickle.dump(tokenizations, open(f'{self.dataset_path}/test_tokenizations_{self.num_ng_test}_{self.token_size}.pkl', 'wb'))
			else:
				tokenizations = pickle.load(open(f'{self.dataset_path}/test_tokenizations_{self.num_ng_test}_{self.token_size}.pkl', 'rb'))
		return tokenizations