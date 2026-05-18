import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import Counter

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Path to the directory with images.
            transform (callable, optional): Optional transformations for images.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        
        self.labels = self._extract_all_labels()

    def _extract_all_labels(self):
        """Helper method for quickly extracting labels from file names."""
        labels = []
        for img_name in self.image_files:
            name_without_ext = os.path.splitext(img_name)[0]
            class_label = int(name_without_ext.split('_')[-1])
            labels.append(class_label)
        return labels

    def get_class_counts(self):
        """Returns a dictionary with the count of elements for each class."""
        counts = Counter(self.labels)
        return dict(counts)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        class_label = self.labels[idx] 
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(class_label, dtype=torch.long)