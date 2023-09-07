import torch
import multiprocessing
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T
import torch.nn.functional as F
from batch_processing import process_batch

import itertools
import test
import util
import parser
import commons
import cosface_loss
import augmentations
from cosplace_model import cosplace_network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
from datasets.targetdomain_dataset import TargetDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments()
#### Datasets
groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]

dataloader = torch.utils.data.DataLoader(groups[0], num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=True,
                                        pin_memory=(args.device == "cuda"), drop_last=True)

dataloader_iterator = iter(dataloader)



# Set the number of worker processes for multiprocessing
num_workers = multiprocessing.cpu_count()

# Define a function to process each batch in parallel
def process_batch_wrapper(batch):
    return process_batch(batch)

# Create multiprocessing.Pool object
pool = multiprocessing.Pool(processes=num_workers)

input_data_list = []  # List to store input data tensors
class_labels_list = []  # List to store class labels
image_paths_list = []  # List to store image paths


total_batches = len(dataloader_iterator)
# Use tqdm for loading bar
with tqdm(total=total_batches, desc='Processing Batches') as pbar:
    print('Now I start with the for cycle')

    # Iterate over the dataloader iterator to get batches of data
    for batch_idx, batch in enumerate(dataloader_iterator):
        # Apply the process_batch function to the batch in parallel
        processed_batch = pool.apply_async(process_batch_wrapper, args=(batch,))
        input_data, class_labels, image_paths = processed_batch.get()

        # Append the tensors and labels to the respective lists
        input_data_list.append(input_data)
        class_labels_list.append(class_labels)
        image_paths_list.extend(image_paths)

        # Update the loading bar
        pbar.update(1)

    # Concatenate all the tensors and labels
    input_data_tensor = torch.cat(input_data_list)
    class_labels_tensor = torch.cat(class_labels_list)

    # Save the concatenated tensors to the final file
    torch.save(input_data_tensor, 'input_data_tensor.pt')
    torch.save(class_labels_tensor,'class_labels.pt' )
    torch.save(image_paths_list, 'image_paths.pt' )
    # Clear the lists
    input_data_list.clear()
    class_labels_list.clear()
    image_paths_list.clear()

# Close the multiprocessing.Pool
pool.close()
pool.join()




print('All tensors saved successfully.')