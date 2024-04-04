import bigjson
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


citations_file_name = "data/DBLP_v12/dblp.v12.json"
references_dict = {'id':[],
                  'references':[]}

if not os.path.exists('data/DBLP_v12/papers_references.feather'):

    with open('data/DBLP_v12/dblp.v12.json', 'rb') as f:
        j = bigjson.load(f)
        for count in tqdm(range(4894081), total=4894081):
            element = j[count]
            if 'references' in element.keys():
                for i,val in enumerate(element['references']):
                    references_dict['references'].append(element['references'][i])
                    references_dict['id'].append(element['id'])

            else:
                references_dict['references'].append(-1)
                references_dict['id'].append(element['id'])
        papers_references = pd.DataFrame.from_dict(references_dict).astype('int32')
        papers_references.to_feather("data/DBLP_v12/papers_references.feather")

with open('data/DBLP_v12/papers_references.feather', 'rb') as f:
    if not os.path.exists('data/DBLP_v12/valid_papers.feather'):
        papers_references = pd.read_feather(f)
        paper_with_references = papers_references[papers_references['references'] != -1]
        paper_with_references['rating'] = 1

        counts = paper_with_references.groupby(['id'])['references'].count()

        index = counts[counts.between(5,100)].index.tolist()

        print(len(index))
        valid_papers = paper_with_references[paper_with_references['id'].isin(index)].reset_index(drop=True)

        valid_papers.to_feather("data/DBLP_v12/valid_papers.feather")
    else:
        with open('data/DBLP_v12/valid_papers.feather', 'rb') as f2:
            valid_papers = pd.read_feather(f2)
groups = valid_papers.groupby(['id'])
if not os.path.exists('data/DBLP_v12/train_index.pkl'):

    train_index = []
    test_index = []

    for name, group in tqdm(groups, total=2794154):
        train, test = train_test_split(group, test_size=0.3)

        for index, row in train.iterrows():
            train_index.append(index)
        
        for index, row in test.iterrows():
            test_index.append(index)
    with open('data/DBLP_v12/train_index.pkl', 'wb') as f:
        pickle.dump(train_index, f)
    with open('data/DBLP_v12/test_index.pkl', 'wb') as f:
        pickle.dump(test_index, f)
else:
    with open('data/DBLP_v12/train_index.pkl', 'rb') as f:
        train_index = pickle.load(f)
    with open('data/DBLP_v12/test_index.pkl', 'rb') as f:
        test_index = pickle.load(f)

train_papers = valid_papers[valid_papers.index.isin(train_index)].reset_index(drop=True)
test_papers = valid_papers[valid_papers.index.isin(test_index)].reset_index(drop=True)

print(len(train_index))
print(len(test_index))
print(train_papers.shape)
print(test_papers.shape)

if not os.path.exists('data/DBLP_v12/papers_train.feather'):
    train_papers.to_feather("data/DBLP_v12/papers_train.feather")
    test_papers.to_feather("data/DBLP_v12/papers_test.feather")



