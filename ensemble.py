# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 19:44:38 2018

@author: shirhe-lyh
"""

import json
import numpy as np


if __name__ == '__main__':
    test_result_json_1 = './test_result_10000_1029_ensemble.json'
    test_result_json_2 = './test_result_10000_normalize_1029_ensemble.json'
    
    with open(test_result_json_1, 'r') as reader:
        test_result_1 = json.load(reader)
    with open(test_result_json_2, 'r') as reader:
        test_result_2 = json.load(reader)
        
    logits_dict_1 = {}
    for logits_dict in test_result_1:
        image_name = logits_dict.get('image_id')
        logits = logits_dict.get('disease_class')
        logits_dict_1[image_name] = logits
    logits_dict_2 = {}
    for logits_dict in test_result_2:
        image_name = logits_dict.get('image_id')
        logits = logits_dict.get('disease_class')
        logits_dict_2[image_name] = logits
        
    result_list = []
    for image_name, logits in logits_dict_1.items():
        logits_ = logits_dict_2.get(image_name, None)
        if logits_ is None:
            print('error ', image_name)
            continue
        label = np.argmax(np.array(logits) + np.array(logits_))
        d = {}
        d['image_id'] = image_name
        d['disease_class'] = int(label)
        result_list.append(d)
        
    with open('result_ensemble_1029.json', 'w') as writer:
        json.dump(result_list, writer)