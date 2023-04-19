import os
from typing import Tuple

from PIL import Image

from torch.utils.data import Dataset


def read_path_images(data_dir: str, name_file: str) -> Tuple:
    with open(name_file) as f:
        left = []
        right = []
        for line in f:
            left.append(os.path.join(data_dir, os.path.relpath(line.rstrip().split()[0])))
            right.append(os.path.join(data_dir, os.path.relpath(line.rstrip().split()[1])))

    return (left, right)


class KittiLoader(Dataset):
    def __init__(self, data_path, filenames_list, mode, transform=None):

        left_path_list, right_path_list = read_path_images(data_path, filenames_list)

        # devo caricare i dataset a partire dal file
        # left_dir = os.path.join(file_name_path, 'image_02/data/')
        self.left_paths = left_path_list  # sorted([os.path.join(left_dir, fname) for fname in os.listdir(left_dir)])
        if mode == 'train':
            # right_dir = os.path.join(file_name_path, 'image_03/data/')
            self.right_paths = right_path_list  # sorted([os.path.join(right_dir, fname) for fname in os.listdir(right_dir)])
            assert len(self.right_paths) == len(self.left_paths)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        if self.mode == 'train':
            right_image = Image.open(self.right_paths[idx])
            sample = {'left_image': left_image, 'right_image': right_image}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
                left_image = self.transform(left_image)
            return left_image
