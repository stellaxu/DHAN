import pickle
import matplotlib.pyplot as plt
import numpy as np
import urllib


output_file = 'target_record.txt'


def format_result(result_line,asin_map,cat_map,pic_map):
    positive_sample_score = result_line[0]
    negative_sample_score = result_line[1]
    cat_attn_positive = result_line[2]
    cat_attn_negative = result_line[3]
    item_attn_positive = result_line[4]
    item_attn_negative = result_line[5]
    item_his = result_line[6]
    cat_his = result_line[7]
    pos_sample_id = result_line[8]
    neg_sample_id = result_line[9]
    item_his_ids = []
    cat_his_ids = []
    for item_his_id in item_his:
        item_his_ids.append(asin_map[item_his_id])
    for cat_his_id in cat_his:
        cat_his_ids.append(cat_map[cat_his_id])
    final_result = []
    final_result.append(asin_map[pos_sample_id])
    final_result.append(positive_sample_score)
    final_result.append(item_attn_positive)
    final_result.append(cat_attn_positive)
    final_result.append(asin_map[neg_sample_id])
    final_result.append(negative_sample_score)
    final_result.append(item_attn_negative)
    final_result.append(cat_attn_negative)
    final_result.append(item_his_ids)
    final_result.append(cat_his_ids)
    for item_his_id in item_his:
        if(asin_map[item_his_id] in pic_map.keys()):
            picurl = pic_map[asin_map[item_his_id]]
            urllib.urlretrieve(picurl,'picfolder/'+str(asin_map[item_his_id])+'.jpg')
    return final_result


pic_map = {}

def count_nonzero_item(input_line):
    item_his = input_line[6]
    return np.count_nonzero(item_his)

item_cat_mapping={}

with open('meta_Electronics.json','r') as f:
    for line in f:
        obj = eval(line)
        cat = obj["categories"][0][-1]
        item_cat_mapping[obj["asin"]]=cat
        if 'imUrl' in obj.keys():
            pic_map[obj['asin']] = obj['imUrl']
print(len(pic_map))

with open('test_output.pkl','r') as f:
    attn_file = pickle.load(f)

with open('dict_50electronic.pkl','r') as f:
    asin_map = pickle.load(f)
    cate_map = pickle.load(f)
#transfer item_cat_mapping into index
for key in item_cat_mapping.keys():
    if item_cat_mapping[key] in cate_map.keys():
        cat_index = cate_map[item_cat_mapping[key]]
        item_cat_mapping[key]=cat_index
    else:
        item_cat_mapping[key]=7

asin_map = {v: k for k, v in asin_map.iteritems()}
cate_map = {v: k for k, v in cate_map.iteritems()}

results=[]
for line in attn_file:
    #filter logic
    his_length = count_nonzero_item(line)
    cat_attn_positive = line[2]
    pos_sample_id = line[8]
    
    if np.argmax(cat_attn_positive)== item_cat_mapping[asin_map[pos_sample_id]] and np.count_nonzero(cat_attn_positive)>=4:
        print("got you!")
        if(his_length>=15 and his_length<=18):
            results.append(line)





file = open(output_file,'w')
for record in results:
    result = format_result(record,asin_map,cate_map,pic_map)
    print>>file,result

    

