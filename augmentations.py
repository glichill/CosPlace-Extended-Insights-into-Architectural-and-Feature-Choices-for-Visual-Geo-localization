
import torch
import albumentations as A
from typing import Tuple, Union
import torchvision.transforms as T
from torchvision.transforms.functional import gaussian_blur

#already-modified
class DeviceAgnosticRandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, p: float):
        super().__init__(p=0.5)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        orizontal_flip = super(DeviceAgnosticRandomHorizontalFlip, self).forward
        augmented_images = [orizontal_flip(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images

class DeviceAgnosticGaussianBlur(T.GaussianBlur):
    def __init__(self, kernel_size: int, sigma: Tuple[float, float]):
        super().__init__(kernel_size = 51, sigma = (0.1, 2))
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        gaussian_blur = super(DeviceAgnosticGaussianBlur, self).forward
        augmented_images = [gaussian_blur(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images

#already_modified
class DeviceAgnosticRandomErasing(T.RandomErasing):
    def __init__(self, p: float, scale: Tuple[float, float], ratio: Tuple[float, float], value: int, inplace: bool):
        super().__init__(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        random_erasing = super(DeviceAgnosticRandomErasing, self).forward
        augmented_images = [random_erasing(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images



class DeviceAgnosticRandomErasingBorder(T.RandomErasing):
    def __init__(self, p: float, scale: Tuple[float, float], ratio: Tuple[float, float], value: int, inplace: bool):
        super().__init__(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        
        # Calculate the border size based on image dimensions
        border_size = int(0.1 * min(H, W))
        
        # Create a mask of the same shape as the images
        mask = torch.ones((B, 1, H, W), dtype=torch.float32, device=images.device)
        
        # Set the border region of the mask to zeros
        mask[:, :, :border_size, :] = 0  # Top border
        mask[:, :, -border_size:, :] = 0  # Bottom border
        mask[:, :, :, :border_size] = 0  # Left border
        mask[:, :, :, -border_size:] = 0  # Right border
        
        # Apply random erasing only to the border region
        random_erasing = super(DeviceAgnosticRandomErasingBorder, self).forward
        erased_images = images * mask + random_erasing(images) * (1 - mask)
        
        return erased_images
       


class DeviceAgnosticColorJitter(T.ColorJitter):
    def __init__(self, brightness: float = 0., contrast: float = 0., saturation: float = 0., hue: float = 0.):
        """This is the same as T.ColorJitter but it only accepts batches of images and works on GPU"""
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        color_jitter = super(DeviceAgnosticColorJitter, self).forward
        augmented_images = [color_jitter(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images


class DeviceAgnosticRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size: Union[int, Tuple[int, int]], scale: float):
        """This is the same as T.RandomResizedCrop but it only accepts batches of images and works on GPU"""
        super().__init__(size=size, scale=scale)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        random_resized_crop = super(DeviceAgnosticRandomResizedCrop, self).forward
        augmented_images = [random_resized_crop(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        return augmented_images


if __name__ == "__main__":
    """
    You can run this script to visualize the transformations, and verify that
    the augmentations are applied individually on each image of the batch.
    """
    from PIL import Image
    # Import skimage in here, so it is not necessary to install it unless you run this script
    from skimage import data
    
    # Initialize DeviceAgnosticRandomResizedCrop
    random_crop = DeviceAgnosticRandomResizedCrop(size=[256, 256], scale=[0.5, 1])
    # Create a batch with 2 astronaut images
    pil_image = Image.fromarray(data.astronaut())
    tensor_image = T.functional.to_tensor(pil_image).unsqueeze(0)
    images_batch = torch.cat([tensor_image, tensor_image])
    # Apply augmentation (individually on each of the 2 images)
    augmented_batch = random_crop(images_batch)
    # Convert to PIL images
    augmented_image_0 = T.functional.to_pil_image(augmented_batch[0])
    augmented_image_1 = T.functional.to_pil_image(augmented_batch[1])
    # Visualize the original image, as well as the two augmented ones
    pil_image.show()
    augmented_image_0.show()
    augmented_image_1.show()
