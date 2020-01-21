import pprint
import pandas as pd
import re
import os
import json
import glob
import numpy as np



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
    body_value = doc[doc.find('\n\n'):].strip().strip('\n').strip('\n\n')  # stripping white spaces and new line chars 
    body_length = len(body_value)
    if  body_length < 100:
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
	# emails.csv contains 517490 emails
	for i, chunk in  enumerate(pd.read_csv('../data/enron/emails.csv', chunksize=51740)):
		print("sampling chunk: \t", i)
		if i==10: mylist.append(chunk)
		else: mylist.append(chunk.sample(1000))
	# creating a dataframe for easy access:
	df = pd.concat(mylist, axis= 0)
	del mylist

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

	parsed_df = pd.DataFrame(parsed_list)
	pp.pprint(parsed_df.head())
	# adding new lines as delimiters between examples and storing them as text files
	# open('../data/enron/enron_lm.txt', "w").write(''.join(df))

	# save the parsed dataframe to a file for training
	parsed_df.to_csv('enron_lm.txt',sep='\t')
	##################################################
	# Checking the maximum body length of the emails:#
	##################################################
	email_bodies = parsed_df.Body
	print('Max sentence length: ', max([len(email_body) for email_body in email_bodies]))
	sorted_email_bodies = sorted(email_bodies, key=len)
	print(len(sorted_email_bodies))
	print("A sample email:\n")
	print(sorted_email_bodies[800][:])


	# for body in parsed_df.Body.sample(10):
	# 	print(type(body))
	# 	if len(body.split()) > 512:
	# 		body = ''.join(body.split()[:511])

