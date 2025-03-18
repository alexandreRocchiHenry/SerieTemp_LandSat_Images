import torch
import torchvision.transforms as T
import random

class SatelliteAugmentation:
    def __init__(self, prob=0.5):
        """
        Initialize the augmentation pipeline with a given probability.
        """
        self.prob = prob

        # Define transformations that preserve the original size (1024x1024)
        self.rgb_transforms = T.Compose([
            T.RandomHorizontalFlip(p=prob),
            T.RandomVerticalFlip(p=prob),
            T.RandomRotation(degrees=(0, 90)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    def __call__(self, image, label):
        """
        Apply augmentations to both the image and label while preserving the IR band.
        """
        if random.random() < self.prob:
            # Split channels: RGB (first 3) and IR (last channel)
            rgb, ir = image[:3], image[3:]

            # Apply transformations to RGB channels only
            rgb = self.rgb_transforms(rgb)

            # Concatenate IR back to the transformed RGB
            image = torch.cat([rgb, ir], dim=0)

        return image, label
