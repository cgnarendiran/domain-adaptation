import pprint
import pandas as pd
import re
import os
import json
import glob
import numpy as np
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

if __name__== "__main__" :
	pp = pprint.PrettyPrinter(indent=4)
	#####################################################
	# chunking and reading 10K samples for pre-training #
	#####################################################
	print("Pre-processing Enron emails and storing about 10K of them: \n")
	mylist = []
	mylist_eval = []
	# emails.csv contains 517490 emails
	for i, chunk in  enumerate(pd.read_csv('../data/emails.csv', chunksize=51740)):
		print("sampling chunk: \t", i)
		if i==10: break
		else: 
			mylist.append(chunk.sample(1000))
			mylist_eval.append(chunk.sample(20))
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
	docs = df.message.sample(20)
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

	parsed_df = pd.DataFrame(parsed_list)
	pp.pprint(parsed_df.head())
	########################
	# check alert!!
	# open('debug_enron.txt', "w").write('\nNEW EMAIL\n'.join(parsed_df.loc[0:20,:].Body.values))
	parsed_df_subset = parsed_df.sample(n=200)
	parsed_df_subset.to_csv('debug_enron.csv', index=False, header=False)
	######################



	# adding new lines as delimiters between examples and storing them as text files
	# open('enron_lm.txt', "w").write('\n'.join(parsed_df.Body.values))

	# save the parsed dataframe to a file for training
	parsed_df.to_csv('enron_lm.tsv',sep='\t', index=False, header=False)



	####################################################
	# Same for creating an eval txt file:
	parsed_list = []
	for doc in df_eval.message:
		parsed_doc = parse_doc(doc)
		if parsed_doc is not None: # checking whether it's None coz of minimal length threshold of 100
			parsed_list.append(parsed_doc)

	parsed_df_eval = pd.DataFrame(parsed_list)
	# pp.pprint(parsed_df.head())

	# adding new lines as delimiters between examples and storing them as text files
	# open('enron_lm_eval.txt', "w").write('\n'.join(parsed_df_eval.Body.values))

	# save the parsed dataframe to a file for training
	parsed_df_eval.to_csv('enron_eval_lm.txt', sep='\t', index=False, header=False)
	


	##################################################
	# Checking the maximum body length of the emails:#
	##################################################
	email_bodies = parsed_df.Body
	print('Max sentence length: ', max([len(email_body) for email_body in email_bodies]))
	print('Avg sentence length: ', mean([len(email_body) for email_body in email_bodies]))
	sorted_email_bodies = sorted(email_bodies, key=len)
	print("Total examples:", len(sorted_email_bodies))
	print("A sample email:\n")
	print(sorted_email_bodies[-1][:])
