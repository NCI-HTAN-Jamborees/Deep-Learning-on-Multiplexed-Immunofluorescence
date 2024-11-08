import tifffile
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import torch

class multichannel_patch_pretrain_dataset(Dataset):
    def __init__(self, data_dir, num_channels, sample_frame, patch_key='patch_fname', channel_list=None, sample_transform=None, random_select=False):
        
        self.data_dir = data_dir
        self.sample_frame = sample_frame
        self.sample_transform = sample_transform
        self.num_channels = num_channels
        self.patch_key = patch_key
        self.channel_list = channel_list
        self.random_select = random_select
        
    def __len__(self):
        return len(self.sample_frame)
    
    def __getitem__(self, idx):
        patch_fname = self.sample_frame.iloc[idx][self.patch_key]
        image_path = os.path.join(self.data_dir, patch_fname)
        sample_im = tifffile.imread(image_path)[0:self.num_channels]
        if self.channel_list is not None:
            sample_im = sample_im[self.channel_list]
        # add something for random channels - single channel setting
        if self.random_select:
            rand_channel = np.random.choice(self.num_channels, 1)
            sample_im = sample_im[rand_channel]

        # sample_im = tifffile.imread(image_path)
        sample_im = np.moveaxis(sample_im, 0, -1).astype(np.float32)
        sample_im = transforms.functional.to_tensor(sample_im)
        
        # perform transformations then correct datatype and return a tensor
        if self.sample_transform:
            sample_im = self.sample_transform(sample_im)
        # view_1 = transforms.functional.to_tensor(view_1)
        # view_2 = transforms.functional.to_tensor(view_2)
        
        return sample_im


class multichannel_bag_dataset(Dataset):
    def __init__(self, embedding_dir, embedding_file_list, sample_frame, sample_transforms=None, im_key='image_fnames', class_key='outcome_response_bin'):
        self.embedding_dir = embedding_dir
        self.embedding_list = embedding_file_list
        self.sample_frame = sample_frame
        self.sample_transforms = sample_transforms
        self.im_key = im_key
        self.class_key = class_key

    def __len__(self):
        return len(self.sample_frame)

    def __getitem__(self, idx):
        wsi_fname = self.sample_frame[self.im_key].iloc[idx]
        wsi_label = self.sample_frame[self.class_key].iloc[idx]
        embedding_list = []
        # grab all patch embeddings for an image
        for embedding_fname in self.embedding_list:
            if wsi_fname not in embedding_fname:
                continue        
            embedding_arr = np.load(os.path.join(self.embedding_dir, embedding_fname)).reshape(1,-1)
            embedding_list.append(embedding_arr)
            
            
        combined_embeddings = np.concatenate(embedding_list, axis=0)
        
        if self.sample_transforms:
            combined_embeddings = self.sample_transforms(combined_embeddings)
            
        combined_embeddings = torch.squeeze(combined_embeddings)
            
        return combined_embeddings, wsi_label


class multichannel_bag_interpretation_dataset(Dataset):
    def __init__(self, embedding_dir, embedding_file_list, sample_frame, sample_transforms=None, im_key='image_fnames', class_key='outcome_response_bin'):
        self.embedding_dir = embedding_dir
        self.embedding_fnames = embedding_file_list
        self.sample_frame = sample_frame
        self.sample_transforms = sample_transforms
        self.im_key = im_key
        self.class_key = class_key

    def __len__(self):
        return len(self.sample_frame)

    def __getitem__(self, idx):
        wsi_fname = self.sample_frame[self.im_key].iloc[idx]
        wsi_label = self.sample_frame[self.class_key].iloc[idx]
        embedding_list = []
        embedding_fnames = []
        # grab all patch embeddings for an image
        for embedding_fname in self.embedding_fnames:
            if wsi_fname not in embedding_fname:
                continue        
            embedding_arr = np.load(os.path.join(self.embedding_dir, embedding_fname)).reshape(1,-1)
            embedding_list.append(embedding_arr)
            embedding_fnames.append(embedding_fname)
            
            
        combined_embeddings = np.concatenate(embedding_list, axis=0)

        if self.sample_transforms:
            combined_embeddings = self.sample_transforms(combined_embeddings)
            
        combined_embeddings = torch.squeeze(combined_embeddings)
            
        return combined_embeddings, wsi_label, embedding_fnames

