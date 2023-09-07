import os
import torch
import random
import logging
import numpy as np
from glob import glob
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as T
from collections import defaultdict

ImageFile.LOAD_TRUNCATED_IMAGES = True



def open_image(path):
    """Open and convert an image to RGB mode."""
    return Image.open(path).convert("RGB")


class TargetDataset(torch.utils.data.Dataset):
    def __init__(self, args, target_dataset_folder):
        """
        Initialize the Target Dataset.

        Parameters
        ----------
        args : args for data augmentation
        target_dataset_folder : str, the path of the target dataset folder with the images.
        """
        super().__init__()  # Call the superclass constructor
        self.target_dataset_folder = target_dataset_folder  # Set the target dataset folder
        self.augmentation_device = args.augmentation_device  # Set the augmentation device

        if self.augmentation_device == "cpu":
            self.transform = T.Compose([
                    T.ColorJitter(brightness=args.brightness,
                                  contrast=args.contrast,
                                  saturation=args.saturation,
                                  hue=args.hue),
                    T.RandomResizedCrop([512, 512], scale=[1-args.random_resized_crop, 1]),
                    #T. ...
                    #T. ...
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        self.target_dataset_paths = sorted(glob(os.path.join(self.target_dataset_folder, "*.jpg"), recursive=True))
        if len(self.target_dataset_paths) == 0:
            raise FileNotFoundError(f"Folder {self.target_dataset_folder} has no JPG images")
        self.images_paths = [p for p in self.target_dataset_paths]
    def __getitem__(self,index):
        image_path = self.images_paths[index]  # Get the path of the image at the specified index
        pil_image = open_image(image_path)  # Open the image using PIL
        tensor_image = T.functional.to_tensor(pil_image)  # Convert the PIL image to a tensor
        if self.augmentation_device == "cpu":
            tensor_image = self.transform(tensor_image)  # Apply the image transformations       
        return tensor_image, torch.tensor(0)  # Return the transformed image and its path
    def __len__(self):
        return len(self.images_paths)
