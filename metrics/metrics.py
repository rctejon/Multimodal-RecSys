import numpy as np
import torch
from tqdm import tqdm
from transformers import BatchEncoding

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
	min_index = 99999
	for ng_item in ng_items:
		if ng_item in pred_items:
			index = pred_items.index(ng_item)
			if index < min_index:
				min_index = index
	if min_index != 99999:
		return np.reciprocal(float(min_index+1))
	return 0

def recall(ng_items, pred_items):
	recall = 0
	for ng_item in ng_items:
		if ng_item in pred_items:
			recall += 1
	return recall / len(ng_items)

def precision(ng_items, pred_items):
	precision = 0
	for ng_item in ng_items:
		if ng_item in pred_items:
			precision += 1
	return precision / len(pred_items)

def combine_tokenization(t1, t2):
		input_ids = torch.cat((t1['input_ids'], t2['input_ids']), dim=0)
		attention_mask = torch.cat((t1['attention_mask'], t2['attention_mask']), dim=0)
		token_type_ids = torch.cat((t1['token_type_ids'], t2['token_type_ids']), dim=0)

		encoded_inputs = BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})
		return encoded_inputs

def metrics(model, test_loader, top_ks, device, ng_num):
	HR, NDCG, MRR = {}, {}, []
	RECALL, PRECISION = {}, {}

	for top_k in top_ks:
		HR[top_k] = []
		NDCG[top_k] = []
		RECALL[top_k] = []
		PRECISION[top_k] = []

	with torch.no_grad():
		for user, item, label, tokenization in tqdm(test_loader, total=len(test_loader)):
			user = user.to(device)
			item = item.to(device)
			label = label.to(device)
			if tokenization is not None:
				tokenization = tokenization.to(device)
			predictions = model(user, item, tokenization) if tokenization is not None else model(user, item)
			for top_k in top_ks:
				ng_items, recommends, mrr_recommends = calculate_metrics_user(predictions, device, user, item, label, tokenization, top_k, ng_num)
				HR[top_k].append(hit(ng_items, recommends))
				NDCG[top_k].append(ndcg(ng_items, recommends))
				RECALL[top_k].append(recall(ng_items, recommends))
				PRECISION[top_k].append(precision(ng_items, recommends))
			MRR.append(mrr(ng_items, mrr_recommends))

	for top_k in top_ks:
		HR[top_k] = np.mean(HR[top_k])
		NDCG[top_k] = np.mean(NDCG[top_k])
		RECALL[top_k] = np.mean(RECALL[top_k])
		PRECISION[top_k] = np.mean(PRECISION[top_k])
	MRR = np.mean(MRR)

	return HR, NDCG, MRR, RECALL, PRECISION


def calculate_metrics_user(predictions, device, user, item, label, tokenization, top_k, ng_num=100):
	user = user.to(device)
	item = item.to(device)
	label = label.to(device)
	if tokenization is not None:
		tokenization = tokenization.to(device)
	
	_, indices = torch.topk(predictions, top_k)
	recommends = torch.take(item, indices).cpu().numpy().tolist()
	_, mrr_indices = torch.topk(predictions, len(predictions))
	mrr_recommends = torch.take(item, mrr_indices).cpu().numpy().tolist()

	ng_items = []

	for i in range(0, user.size(0), ng_num + 1):
		if label[i].item() == 1:
			ng_item = item[i].item()
			ng_items.append(ng_item)
	return ng_items, recommends, mrr_recommends