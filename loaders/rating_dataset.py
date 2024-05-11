from torch.utils.data import Dataset
from torch import tensor
import torch

class RatingDataset(Dataset):
	def __init__(self, user_list, item_list, rating_list, tokenization_list=None):
		super(RatingDataset, self).__init__()
		self.user_list = user_list
		self.item_list = item_list
		self.rating_list = rating_list
		self.tokenization_list = tokenization_list

	def __len__(self):
		return len(self.user_list)

	def __getitem__(self, idx):
		user = self.user_list[idx]
		item = self.item_list[idx]
		rating = self.rating_list[idx]
		if self.tokenization_list is not None:
			return (
				tensor(user, dtype=torch.long),
				tensor(item, dtype=torch.long),
				tensor(rating, dtype=torch.float),
				self.tokenization_list[idx]
				)

		return (
			tensor(user, dtype=torch.long),
			tensor(item, dtype=torch.long),
			tensor(rating, dtype=torch.float)
			)
			