import torch

if __name__ == '__main__':
    # If there's a GPU available...
	if torch.cuda.is_available():    

	    # Tell PyTorch to use the GPU.    
	    device = torch.device("cuda")

	    print('There are %d GPU(s) available.' % torch.cuda.device_count())

	    print('We will use the GPU:', torch.cuda.get_device_name(0))

	# If not...
	else:
	    device = torch.device("cpu")

	    print('No GPU available, using the CPU instead.', torch.get_device_name(0))