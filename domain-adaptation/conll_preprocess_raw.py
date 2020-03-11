import numpy as np 
import random

if __name__ == '__main__':
	with open('../data/conll/train.txt') as f:
		lines = f.readlines()
	with open('../data/conll/dev.txt') as f:
		lines = lines + f.readlines()
	raw_text = []
	doc = []
	for line in lines:
		# print(line)
		# print('line is not new line')
		word = line.split(' ')[0]
		# print(word)
		if word == '-DOCSTART-':
			if doc: raw_text.append(doc) 
			doc = []
		else:
			doc.append(word)
	raw_text.append(doc)
	random.shuffle(raw_text)

	print(len(raw_text))
	with open('conll_lm.txt', 'w') as f:
		for i, doc in enumerate(raw_text):
			if i == 5000: break
			for word in doc:
				#f.seek(0)
				if word is not '\n': f.write(word + ' ')
				else: f.write(word) 