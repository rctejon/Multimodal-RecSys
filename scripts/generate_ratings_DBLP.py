import bigjson
from tqdm import tqdm
import os
import pandas as pd

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
    papers_references = pd.read_feather(f)
    print(papers_references[papers_references['id'] < -1])
    paper_with_references = papers_references[papers_references['references'] != -1]
    paper_with_references['rating'] = 1
    print(paper_with_references)

    train = paper_with_references.sample(frac = 0.8)
    test = paper_with_references.drop(train.index)

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    # train.to_feather("data/DBLP_v12/papers_train.feather")
    # test.to_feather("data/DBLP_v12/papers_test.feather")



