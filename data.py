from __future__ import print_function, division
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ClassificationDataset(Dataset):
    """Image classification dataset."""

    def __init__(self, annotations, root_dir, transform=None):
        """
        Args:
            annotations (string): dict with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = annotations # {class : numar}
        self.root_dir = root_dir # root_folder
        self.transform = transform
        self.indexes = []
        for k, v in annotations.items():
            path = os.path.join(root_dir, k)
            for _, _, files in os.walk(path):
                for filename in files:
                    self.indexes.append((os.path.join(path, filename), v))
            
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):

        blob, class_id = self.indexes[idx]
        sample = {'blob': blob, 'class_id': class_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

class PillowLoader():
    def __call__(self, sample):
        sample['blob'] = Image.open(sample['blob'])
        return sample


if __name__ == '__main__':

    annotations = {
        'birman' : 0,
        'persian' : 1,
        'himalayan' : 2,
        'siberian': 3
    }

    root_dir = '/home/michael/ML_problems/cat_classifier/cat_races/train'
    transforms = PillowLoader()

    dataset = ClassificationDataset(annotations, root_dir, transforms)
    print([x['blob'].size[0] * x['blob'].size[1] for x in dataset])
    print(min([x['blob'].size[0] * x['blob'].size[1] for x in dataset]))
    print(max([x['blob'].size[0] * x['blob'].size[1] for x in dataset]))
    print(len(dataset))
    print(dataset[40])