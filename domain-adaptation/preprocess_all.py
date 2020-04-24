import pprint
import pandas as pd
import re
import os
import json
import glob
import numpy as np
import random
from statistics import mean

def parse_doc(doc):
	# finding the keys in this particular email 
	keys = ['Message-ID']+re.findall('\n([\w\-]+):', doc[:doc.find('\n\n')])
	keys = pd.Series(keys).drop_duplicates().tolist()

	values = []
	for a, k in enumerate(keys):
		k = k+':'
		try:
			# append values for all the keys
			values.append(doc[doc.find(k)+len(k):doc.find(keys[a+1])].strip())
		except:
			# for the last key add values till the start of body which is '\n\n'
			values.append(doc[doc.find(k)+len(k):doc.find('\n\n')].strip())

	# Remove entries that have less than 10 length and truncate the maximum length of emails to 127
	og_message_pos =  doc.find('-----Original Message-----')
	fw_message_pos =  doc.find('---------------------- Forwarded by')
	if og_message_pos != -1:
		# print("\nog_message_pos:", og_message_pos)
		body_value = doc[doc.find('\n\n'):og_message_pos].strip() # stripping white spaces and new line chars
	elif fw_message_pos != -1:
		# print("\nfw_message_pos:", fw_message_pos)
		body_value = doc[doc.find('\n\n'):fw_message_pos].strip()
	else:
		# print("\nno og or fw pos")
		body_value = doc[doc.find('\n\n'):].strip()

	body_length = len(body_value)
	############ uncomment this block to save with a threshold of body length
	if  body_length < 4:
		return None
	elif body_length >= 512:
		body_value = body_value[0:512]

	# create dict of fields with values including the Body field
	d = dict(zip(keys+['Body'],values+[body_value]))

	# add or remove keys from the exhaustive list of allkeys
	k_to_remove = set(d.keys()) - set(allkeys)
	k_to_add = set(allkeys) - set(d.keys())
	
	for k in k_to_remove:
		d.pop(k)
	for k in k_to_add:
		d[k] = ''
	# keys = [k[:-1] for k in keys]
	return return_desired_fields(d, desired_fields)


def return_desired_fields(dict_doc, desired_fields):
	fields_to_remove = set(allkeys) - set(desired_fields)
	for f in fields_to_remove:
		dict_doc.pop(f)
	return dict_doc







#####################################################################
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
	raw_text_train = raw_text[:5000]
	raw_text_eval = raw_text[5000:5100]

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

	#####################################################
	# chunking and reading 10K samples for pre-training #
	#####################################################
	print("Pre-processing Enron emails and storing about 5K of them: \n")
	mylist = []
	mylist_eval = []
	# emails.csv contains 517490 emails
	for i, chunk in  enumerate(pd.read_csv('../data/emails.csv', chunksize=51740)):
		print("sampling chunk: \t", i)
		if i==10: break
		else: 
			mylist.append(chunk.sample(500))
			mylist_eval.append(chunk.sample(10))
	# creating a dataframe for easy access:
	df = pd.concat(mylist, axis= 0)
	df_eval = pd.concat(mylist_eval, axis=0)
	del mylist
	del mylist_eval

	# print(df.head())
	# Looks like this:
	                            # file                                            message
	# 8820   badeer-r/all_documents/264.  Message-ID: <11348414.1075863593218.JavaMail.e...
	# 841     allen-p/all_documents/318.  Message-ID: <12293216.1075855672454.JavaMail.e...
	# 32937           benson-r/inbox/31.  Message-ID: <19387609.1075840372845.JavaMail.e...
	# 47593     carson-m/sent_items/100.  Message-ID: <15354087.1075853147893.JavaMail.e...
	# 8625       badeer-r/_sent_mail/46.  Message-ID: <19695514.1075863607801.JavaMail.e...

	# print(df.shape)
	# (10000, 2)

	#########################
	# parse the text fields:#
	#########################
	# finding all the existing keys from 50 sampled emails
	docs = df.message.sample(50)
	allkeys = [re.findall('\n([\w\-]+):', doc[:doc.find('\n\n')]) for doc in docs]
	allkeys = set(sum(allkeys,[]))
	allkeys.add('Message-ID')
	allkeys.add('Body')
	print("Unique keys present in the email:\n")
	pp.pprint(allkeys)
	# fields to retain for training purposes:
	# desired_fields = ['From', 'To', 'Cc', 'Bcc', 'Body']
	desired_fields = ['Body']
	parsed_list = []
	for doc in df.message:
		parsed_doc = parse_doc(doc)
		if parsed_doc is not None: # checking whether it's None coz of minimal length threshold of 100
			parsed_list.append(parsed_doc)

	df_enron = pd.DataFrame(parsed_list)
	pp.pprint(df_enron.head())
	# adding new lines as delimiters between examples and storing them as text files

	concat_df = pd.concat([df_conll,df_enron]).sort_index().reset_index(drop=True)
	open('lm.txt', "w").write(''.join(concat_df.Body.values))

	# save the parsed dataframe to a file for training
	# parsed_df.to_csv('enron_lm.tsv',sep='\t')



	####################################################
	# Same for creating an eval txt file:
	parsed_list = []
	for doc in df_eval.message:
		parsed_doc = parse_doc(doc)
		if parsed_doc is not None: # checking whether it's None coz of minimal length threshold of 100
			parsed_list.append(parsed_doc)

	df_enron_eval = pd.DataFrame(parsed_list)

	concat_df_eval = pd.concat([df_conll_eval,df_enron_eval]).sort_index().reset_index(drop=True)
	open('lm_eval.txt', "w").write(''.join(concat_df_eval.Body.values))

	# save the parsed dataframe to a file for training
	# parsed_df.to_csv('enron_lm.tsv',sep='\t')
	# save a txt file when threholding of the body is NOT done:
	# parsed_df.to_csv('enron_lm.txt', sep='\t', index=False, header=False)
	##################################################
	# Checking the maximum body length of the emails:#
	##################################################
	email_bodies = df_enron.Body
	print('Max sentence length: ', max([len(email_body) for email_body in email_bodies]))
	print('Avg sentence length: ', mean([len(email_body) for email_body in email_bodies]))
	sorted_email_bodies = sorted(email_bodies, key=len)
	print("Total examples:", len(sorted_email_bodies))
	print("A sample email:\n")
	print(sorted_email_bodies[-1][:])

