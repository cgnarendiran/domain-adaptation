import pprint
import pandas as pd
import re
import os
import json
import glob
import numpy as np
import random
from statistics import mean

if __name__== "__main__" :
	pp = pprint.PrettyPrinter(indent=4)	
	print("Pre-processing CONLL docs about 5K of them: \n")
	with open('../data/conll/train.txt') as f:
		lines = f.readlines()
	with open('../data/conll/dev.txt') as f:
		lines = lines + f.readlines()
	raw_text = []
	doc = ''
	for line in lines:
		# print(line)
		# print('line is not new line')
		word = line.split(' ')[0]
		# print(word)
		if word == '-DOCSTART-' or word == '\n':
			if doc: 
				if len(doc) <4: pass
				else: raw_text.append(doc) 
			doc = ''
		else:
			doc = doc + ' ' + word
	raw_text.append(doc)
	random.shuffle(raw_text)
	raw_text_train = raw_text[:10000]
	raw_text_eval = raw_text[10000:10100]

	df_conll = pd.DataFrame(raw_text_train, columns=['Body'])
	# r = []
	# for i in range(5000):
	#  	r = r + [i,i]
	# s = pd.Series(r)
	# # print(s.head())
	# df_conll = df_conll.set_index([s])
	df_conll_eval = pd.DataFrame(raw_text_eval, columns=['Body'])

	pp.pprint(df_conll.head())
	news_bodies = df_conll.Body
	print('Max sentence length: ', max([len(news_body) for news_body in news_bodies]))
	print('Avg sentence length: ', mean([len(news_body) for news_body in news_bodies]))
	sorted_news_bodies = sorted(news_bodies, key=len)
	print("Total examples:", len(sorted_news_bodies))
	print("A sample news:\n")
	print(sorted_news_bodies[-1][:])
	# with open('conll_lm.txt', 'w') as f:
	# 	for i, doc in enumerate(raw_text):
	# 		for word in doc:
	# 			#f.seek(0)
	# 			if word is not '\n': f.write(word + ' ')
	# 			else: f.write(word) 

	# save the parsed dataframe to a file for training
	df_conll.to_csv('conll_lm.tsv',sep='\t', index=False, header=False)

	df_conll_eval.to_csv('conll_eval_lm.txt', sep='\t', index=False, header=False)
	