/* 
 * Copyright (C) 2002-2019 Igor Sysoev
 * Copyright (C) 2011,2019 Nginx, Inc.
 * Copyright (C) 2010-2019 Alibaba Group Holding Limited
 * Copyright (C) 2011-2013 Xiaozhe "chaoslawful" Wang
 * Copyright (C) 2011-2013 Zhang "agentzh" Yichun
 * Copyright (C) 2011-2013 Weibin Yao
 * Copyright (C) 2012-2013 Sogou, Inc.
 * Copyright (C) 2012-2013 NetEase, Inc.
 * Copyright (C) 2014-2017 Intel, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

import sys
import random
import time
import pandas as pd

def to_df(file_path):
	with open(file_path, 'r') as fin:
		df = {}
		i = 0
		for line in fin:
			df[i] = eval(line)
			i += 1
		df = pd.DataFrame.from_dict(df, orient='index')
		return df


def process_meta(file):
	fi = open(file, "r")
	fo = open("item-info", "w")
	item_cat_mapping={}
	for line in fi:
		obj = eval(line)
		cat = obj["categories"][0][-1]
		item_cat_mapping[obj["asin"]]=cat
	cat_count={}
	for item in item_cat_mapping.keys():
		cat = item_cat_mapping[item]
		if cat not in cat_count.keys():
			cat_count[cat]=1
		else:
			cat_count[cat]+=1
	#generate valid cat set
	valid_cat = set()
	for cat in cat_count.keys():
		if cat_count[cat]>299:
			valid_cat.add(cat)
	print(len(valid_cat))
	print(valid_cat)
	fi = open(file, "r")
	for line in fi:
		obj = eval(line)
		cat = obj["categories"][0][-1]
		if cat in valid_cat:
			print>>fo, obj["asin"] + "\t" + cat
		else:
			print>>fo,obj["asin"] + "\t" + "default_cat"

def process_meta_new(file,review_file):
	fi = open(file, "r")
	fo = open("item-info", "w")
	fr = open(review_file,'r')
	item_with_review=set()
	for line in fr:
		obj = eval(line)
		itemID = obj["asin"]
		item_with_review.add(itemID)
	#only keep items with reviews
	item_cat_mapping={}
	for line in fi:
		obj = eval(line)
		cat = obj["categories"][0][-1]
		if obj["asin"] in item_with_review:
			item_cat_mapping[obj["asin"]]=cat
	cat_count={}
	for item in item_cat_mapping.keys():
		cat = item_cat_mapping[item]
		if cat not in cat_count.keys():
			cat_count[cat]=1
		else:
			cat_count[cat]+=1
	#generate valid cat set
	valid_cat = set()
	for cat in cat_count.keys():
		if cat_count[cat]>299:
			valid_cat.add(cat)
	print(len(valid_cat))
	print(valid_cat)
	fi = open(file, "r")
	for line in fi:
		obj = eval(line)
		cat = obj["categories"][0][-1]
		if cat in valid_cat:
			print>>fo, obj["asin"] + "\t" + cat
		# else:
		# 	print>>fo,obj["asin"] + "\t" + "default_cat"





def process_reviews(file):
	fi = open(file, "r")
	user_map = {}
	fo = open("reviews-info", "w")
	for line in fi:
		obj = eval(line)
		userID = obj["reviewerID"]
		itemID = obj["asin"]
		rating = obj["overall"]
		time = obj["unixReviewTime"]
		print>>fo, userID + "\t" + itemID + "\t" + str(rating) + "\t" + str(time)

def process_reviews_new(file,meta_file):
	fo = open("reviews-info", "w")
	reviews = to_df(file)
	meta = pd.read_csv(meta_file,sep='\t',header=None)
	meta.columns=['asin','categories']
	meta = meta[meta['categories']<>'default_cat']
	reviews = reviews[reviews['asin'].isin(meta['asin'].unique())]
	valid_reviews = pd.DataFrame(reviews.reviewerID.value_counts()).reset_index()
	valid_reviews.columns=['reviewerID','cnt']
	valid_reviews = valid_reviews[valid_reviews['cnt']>=5]
	reviews = reviews[reviews['reviewerID'].isin(valid_reviews['reviewerID'].unique())]

	reviews = reviews.reset_index(drop=True)
	for i in range(reviews.shape[0]):
		userID = reviews["reviewerID"][i]
		itemID = reviews["asin"][i]
		rating = reviews["overall"][i]
		time = reviews["unixReviewTime"][i]
		print>>fo, userID + "\t" + itemID + "\t" + str(rating) + "\t" + str(time)
	fo.close()
	new_meta = meta[meta['asin'].isin(reviews['asin'].unique())]
	new_meta = new_meta.reset_index()
	fo = open("item-info", "w")
	for i in range(new_meta.shape[0]):
		itemID = new_meta["asin"][i]
		cat = new_meta["categories"][i]
		print>>fo, itemID + "\t" + cat
	fo.close()




def manual_join():
	f_rev = open("reviews-info", "r")
	user_map = {}
	item_list = []
	for line in f_rev:
		line = line.strip()
		items = line.split("\t")
		#loctime = time.localtime(float(items[-1]))
		#items[-1] = time.strftime('%Y-%m-%d', loctime)
		if items[0] not in user_map:
			user_map[items[0]]= []
		user_map[items[0]].append(("\t".join(items), float(items[-1])))
		item_list.append(items[1])
	f_meta = open("item-info", "r")
	meta_map = {}
	for line in f_meta:
		arr = line.strip().split("\t")
		if arr[0] not in meta_map:
			meta_map[arr[0]] = arr[1]
			arr = line.strip().split("\t")
	fo = open("jointed-new", "w")
	for key in user_map:
		sorted_user_bh = sorted(user_map[key], key=lambda x:x[1])
		for line, t in sorted_user_bh:
			items = line.split("\t")
			asin = items[1]
			j = 0
			while True:
				asin_neg_index = random.randint(0, len(item_list) - 1)
				asin_neg = item_list[asin_neg_index]
				if asin_neg == asin:
					continue 
				items[1] = asin_neg
				if asin_neg in meta_map:
					print>>fo, "0" + "\t" + "\t".join(items) + "\t" + meta_map[asin_neg]
				else:
					print>>fo, "0" + "\t" + "\t".join(items) + "\t" + "default_cat"
				j += 1
				if j == 1:			 #negative sampling frequency
					break
			if asin in meta_map:
				print>>fo, "1" + "\t" + line + "\t" + meta_map[asin]
			else:
				print>>fo, "1" + "\t" + line + "\t" + "default_cat"


def split_test():
	fi = open("jointed-new", "r")
	fo = open("jointed-new-split-info", "w")
	user_count = {}
	for line in fi:
		line = line.strip()
		user = line.split("\t")[1]
		if user not in user_count:
			user_count[user] = 0
		user_count[user] += 1
	fi.seek(0)
	i = 0
	last_user = "A26ZDKC53OP6JD"
	for line in fi:
		line = line.strip()
		user = line.split("\t")[1]
		if user == last_user:
			if i < user_count[user] - 2:	# 1 + negative samples
				print>> fo, "20180118" + "\t" + line
			else:
				print>>fo, "20190119" + "\t" + line
		else:
			last_user = user
			i = 0
			if i < user_count[user] - 2:
				print>> fo, "20180118" + "\t" + line
			else:
				print>>fo, "20190119" + "\t" + line
		i += 1

#process_meta(sys.argv[1])
process_meta_new('../raw_data/meta_Electronics.json','../raw_data/reviews_Electronics_5.json')
process_reviews_new('../raw_data/reviews_Electronics_5.json','item-info')
manual_join()
split_test()
