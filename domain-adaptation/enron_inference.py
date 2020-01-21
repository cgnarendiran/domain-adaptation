from bert import Ner
import pprint
import pandas as pd
import re
import os
import json
import glob
import numpy as np
import nltk
nltk.download('punkt')

pp = pprint.PrettyPrinter(indent=4)
model = Ner("output/")


# mylist = []

# for chunk in  pd.read_csv('emails.csv', chunksize=50000):
#     mylist.append(chunk)
#     break

# df = pd.concat(mylist, axis= 0)
# del mylist
# print(df.head())


# output = model.predict("In 1949, an Italian Jesuit priest named Roberto Busa presented a pitch to Thomas J. Watson, of I.B.M. Busa was trained in philosophy, and had just published his thesis on St. Thomas Aquinas, the Catholic theologian with a famously unmanageable œuvre. ")
# output = model.predict("Cecil,	Can you take a look at this presentation	and see if the numbers make sense?	thanks,	Monika") 
# pp.pprint(output)


#############################
## parse the text fields:
################################
# docs = df.message.sample(1000)
# allkeys = [re.findall('\n([\w\-]+):', doc[:doc.find('\n\n')]) for doc in docs]
# allkeys = sum(allkeys,[])
# allkeys = set(allkeys)
# allkeys.add('Message-ID')
# allkeys.add('Body')
# allkeys

# pp.pprint(allkeys)


# def parse_doc(doc):
#     keys = ['Message-ID']+re.findall('\n([\w\-]+):', doc[:doc.find('\n\n')])
#     keys = pd.Series(keys).drop_duplicates().tolist()

#     values = []
#     for a, k in enumerate(keys):
#         k = k+':'
#         try:
#             values.append(doc[doc.find(k)+len(k):doc.find(keys[a+1])].strip())
#         except:
#             values.append(doc[doc.find(k)+len(k):doc.find('\n\n')].strip())
    
#     d = dict(zip(keys+['Body'],values+[doc[doc.find('\n\n'):].strip()]))
#     k_to_remove = set(d.keys()) - set(allkeys)
#     k_to_add = set(allkeys) - set(d.keys())
    
#     for k in k_to_remove:
#         d.pop(k)
#     for k in k_to_add:
#         d[k] = ''

#     keys = [k[:-1] for k in keys]
#     return d

# parsed_list = []
# for doc in df.message.sample(20000):
# 	parsed_list.append(parse_doc(doc))

# parsed_df = pd.DataFrame(parsed_list)

# # print(parsed_df.head())




# for body in parsed_df.Body.sample(10):
# 	print(type(body))
# 	if len(body.split()) > 512:
# 		body = ''.join(body.split()[:511])
# 	output = model.predict(body) 
# 	pp.pprint(output)

#################################
# for text in df.message.sample(10):
# 	output = model.predict(text) 
# 	pp.pprint(output)

######################
# reading multiple json files
# pd.set_option('display.max_columns', None)

df_annotations = pd.DataFrame(columns=['completions', 'data', 'id', 'task_path'])
json_annotations = []

for index, file in enumerate(sorted(glob.glob('output-annotator-1/*.json'), key=lambda x: int(x.split('/')[1].split('.')[0])) ):
	with open(file) as f:
		json_data = json.load(f)
		json_annotations.append(json_data)
		# df_annotations.loc[index] = [json_data['completions'], json_data['data'], json_data['id'], json_data['task_path']]

# print(df_annotations.head())

for i, json_data in enumerate(json_annotations):
	print(json_data['id'])
	body = ''
	body_raw = json_data['data']['text']
	if len(body_raw) > 1000:
		body = ''.join(body_raw[:1000])
	else:
		body = body_raw
	print(body)
	output = model.predict(body) 
	pp.pprint(output)
	if i == 10: break


################################





######################
# reading multiple json files
# pd.set_option('display.max_columns', None)




# df_annotations = pd.DataFrame(columns=['completions', 'data', 'id', 'task_path'])
# json_annotations = []

# for index, file in enumerate(sorted(glob.glob('output-annotator-1/*.json'), key=lambda x: int(x.split('/')[1].split('.')[0])) ):
# 	with open(file) as f:
# 		json_data = json.load(f)
# 		json_annotations.append(json_data)
# 		# df_annotations.loc[index] = [json_data['completions'], json_data['data'], json_data['id'], json_data['task_path']]

# # print(df_annotations.head())

# for i, json_data in enumerate(json_annotations):
# 	print(json_data['id'])
# 	body = ''
# 	body_raw = json_data['data']['text']
# 	if len(body_raw) > 1000:
# 		body = ''.join(body_raw[:1000])
# 	else:
# 		body = body_raw
# 	print(body)
# 	output = model.predict(body) 
# 	pp.pprint(output)
# 	if i == 100: break