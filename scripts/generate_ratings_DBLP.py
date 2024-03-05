import json
from tqdm import tqdm
import math

import pickle
import os


def calculate_rating(progress, threshold=0.8):
    if progress >= threshold:
        return 5
    else:
        return 5 * math.sqrt(progress / threshold)


citations_file_name = "data/DBLP_v12/dblp.v12.json"

citations_file = open(citations_file_name, encoding='utf-8')

citations = json.load(citations_file)

for citation in citations:
    print(1)

citations_file.close()
