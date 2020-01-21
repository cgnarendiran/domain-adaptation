import pandas as pd
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



if __name__ == "__main__":

	# Load the dataset into a pandas dataframe.
	df = pd.read_csv("enron_lm.csv", delimiter='\t')

	# Report the number of sentences.
	print('Number of training sentences: {:,}\n'.format(df.shape[0]))

	# Display 10 random rows from the data.
	sentences = df.Body.values

	# Load the BERT tokenizer.
	print('Loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

	# # Print the original sentence.
	# print(' Original: ', sentences[0])

	# # Print the sentence split into tokens.
	# print('Tokenized: ', tokenizer.tokenize(sentences[0]))

	# # Print the sentence mapped to token ids.
	# print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

	# Tokenize all of the sentences and map the tokens to thier word IDs.
	input_ids = []

	# For every sentence...
	for sent in sentences:
	    # `encode` will:
	    #   (1) Tokenize the sentence.
	    #   (2) Prepend the `[CLS]` token to the start.
	    #   (3) Append the `[SEP]` token to the end.
	    #   (4) Map tokens to their IDs.
	    encoded_sent = tokenizer.encode(
	                        sent,                      # Sentence to encode.
	                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'

	                        # This function also supports truncation and conversion
	                        # to pytorch tensors, but we need to do padding, so we
	                        # can't use these features :( .
	                        # max_length = 512,          # Truncate all sentences.
	                        return_tensors = 'pt',     # Return pytorch tensors.
	                   )
	    
	    # Add the encoded sentence to the list.
	    input_ids.append(encoded_sent)

	# Print sentence 0, now	 as a list of IDs.
	print('Original: ', sentences[0])
	print('Token IDs:', input_ids[0])


	# Set the maximum sequence length.
	# I've chosen 64 somewhat arbitrarily. It's slightly larger than the
	# maximum training sentence length of 47...
	MAX_LEN = 512

	print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)

	print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

	# Pad our input tokens with value 0.
	# "post" indicates that we want to pad and truncate at the end of the sequence,
	# as opposed to the beginning.
	input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
	                          value=0, truncating="post", padding="post")

	print('\nDone.')

	# Create attention masks	
	attention_masks = []

	# For each sentence...
	for sent in input_ids:
	    
	    # Create the attention mask.
	    #   - If a token ID is 0, then it's padding, set the mask to 0.
	    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
	    att_mask = [int(token_id > 0) for token_id in sent]
	    
	    # Store the attention mask for this sentence.
	    attention_masks.append(att_mask)

	# Convert all inputs and labels into torch tensors, the required datatype 
`	# for our model.
	train_masks = torch.tensor(attention_masks)


	# The DataLoader needs to know our batch size for training, so we specify it 
	# here.
	# For fine-tuning BERT on a specific task, the authors recommend a batch size of
	# 16 or 32.

	batch_size = 32

	# Create the DataLoader for our training set.
	train_data = TensorDataset(train_inputs, train_masks)
	train_sampler = RandomSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
