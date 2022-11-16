from torch.utils.data import Dataset, sampler
import pathlib
import os
from PIL import Image
import numpy as np
import random


class ImagesDataset(Dataset):

    def __init__(self, data_path, images_transform, target_transform):
        self.data_path = pathlib.Path(data_path)
        self.classes, self.class_to_idx = self.find_classes()
        self.images, self.intervals, self.image_to_class = self.extract_images()
        self.transform = images_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, domain = self.images[index]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            domain = self.target_transform(domain)
        return image, domain

    def __len__(self):
        return len(self.images)

    def extract_images(self):
        images = []
        intervals = []
        image_to_class = {}
        j, i = 0, 0
        for domain in os.listdir(self.data_path):
            if os.path.isdir(self.data_path / domain):
                for image in os.listdir(self.data_path / domain):
                    images.append((self.data_path / domain / image, self.class_to_idx[domain]))
                    image_to_class[i] = self.class_to_idx[domain]
                    i += 1
            intervals.append([j, i])
            j = i
        if i != j:
            intervals.append((j, i))
        return images, intervals, image_to_class


    def find_classes(self):
        classes = []
        for domain in os.listdir(self.data_path):
            if os.path.isdir(self.data_path / domain):
                classes.append(domain)
        classes.sort()
        class_to_idx = dict(
            zip(classes, range(len(classes)))
            )
        return classes, class_to_idx


class BalanceSamplerFilled(sampler.Sampler):
    
    def __init__(self, intervals, n_images=2, repeat=1):
        """
        intervals: List[List]
            Intervals from dataset instance, number of images per class before sampling
        n_images: int
            Number of images per domain
        """
        class_len = len(intervals)
        list_sp = []
        self.idx = []
        
        # find the max interval, ARANGE
        interval_list = [np.arange(interval[0], interval[1]) for interval in intervals]
        len_max = max([interval[1] - interval[0] for interval in intervals])

        # Balancing maximum len
        if len_max % n_images != 0:
            if len_max % n_images < int(0.3 * n_images):
                len_max = len_max - len_max % n_images
            else:
                len_max = len_max - len_max % n_images + n_images
            
        for _ in range(repeat):
            # filled images for each class
            for l in interval_list:
                if l.shape[0] < len_max:
                    # If number of images for a class < len_max then add to len_max random examples
                    l_ext = np.random.choice(l, len_max - l.shape[0])
                    l_ext = np.concatenate((l, l_ext), axis=0)
                    l_ext = np.random.permutation(l_ext)
                elif l.shape[0] > len_max:
                    l_ext = np.random.choice(l, len_max, replace=False)
                    l_ext = np.random.permutation(l_ext)
                elif l.shape[0] == len_max:
                    l_ext = np.random.permutation(l)
                # l_ext is images indexes for a specific class with number of len_max
                list_sp.append(l_ext)
            random.shuffle(list_sp)

            self.idx += np.vstack(list_sp).reshape((n_images * class_len, -1)).T.reshape((1, -1)).flatten().tolist()
        random.shuffle(self.idx)
        # print('total images size in sampler: {}'.format(len(self.idx)))
        
    def __iter__(self):
        return iter(self.idx)
    
    def __len__(self):
        return len(self.idx)
