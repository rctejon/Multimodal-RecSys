import numpy as np
import torch
from tqdm import tqdm

def hit(ng_items, pred_items):
	for ng_item in ng_items:
		if ng_item in pred_items:
			return 1
	return 0


def idcg(ng_items):
	idcg = 0
	for i in range(len(ng_items)):
		idcg += np.reciprocal(np.log2(i+2))
	return idcg


def ndcg(ng_items, pred_items):
	dcg = 0
	for ng_item in ng_items:
		if ng_item in pred_items:
			index = pred_items.index(ng_item)
			dcg += np.reciprocal(np.log2(index+2))
	return dcg / idcg(ng_items)


def mrr(ng_items, pred_items):
	min_index = 999
	for ng_item in ng_items:
		if ng_item in pred_items:
			index = pred_items.index(ng_item)
			if index < min_index:
				min_index = index
	if min_index != 999:
		return np.reciprocal(float(min_index+1))
	return 0


def metrics(model, test_loader, top_k, device, ng_num):
	HR, NDCG, MRR = [], [], []
	
	current_user = None
	current_item = None
	current_label = None

	current_user_id = None
	for user, item, label in tqdm(test_loader, total=len(test_loader)):
		user = user.cpu()
		item = item.cpu()
		label = label.cpu()
		
		if current_user == None:
			current_user = user
			current_item = item
			current_label = label

			current_user_id = user.numpy()[0]
		elif current_user_id == user.numpy()[0]:
			current_user = torch.cat((current_user, user), 0)
			current_item = torch.cat((current_item, item), 0)
			current_label = torch.cat((current_label, label), 0)
		else:
			ng_items, recommends = calculate_metrics_user(model, device, current_user, current_item, current_label, top_k, ng_num)
			HR.append(hit(ng_items, recommends))
			NDCG.append(ndcg(ng_items, recommends))
			MRR.append(mrr(ng_items, recommends))

			current_user = user
			current_item = item
			current_label = label

			current_user_id = user.numpy()[0]
	ng_items, recommends = calculate_metrics_user(model, device, current_user, current_item, current_label, top_k, ng_num)
	HR.append(hit(ng_items, recommends))
	NDCG.append(ndcg(ng_items, recommends))
	MRR.append(mrr(ng_items, recommends))
	return np.mean(HR), np.mean(NDCG), np.mean(MRR)


def calculate_metrics_user(model, device, user, item, label, top_k, ng_num=100):
	user = user.to(device)
	item = item.to(device)
	label = label.to(device)

	predictions = model(user, item)
	_, indices = torch.topk(predictions, top_k)
	recommends = torch.take(item, indices).cpu().numpy().tolist()

	ng_items = []

	for i in range(0, user.size(0), ng_num + 1):
		if label[i].item() == 1:
			ng_item = item[i].item()
			ng_items.append(ng_item)
	return ng_items, recommends