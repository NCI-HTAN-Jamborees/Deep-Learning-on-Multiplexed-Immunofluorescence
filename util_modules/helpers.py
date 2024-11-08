import torch
import numpy as np
from torchvision import transforms
from lightly.models import utils
import re


def unpatchify(
    patches: torch.Tensor, patch_size: int, channels: int = 3
) -> torch.Tensor:
    """
    Reconstructs images from their patches.

     Args:
         patches:
             Patches tensor with shape (batch_size, num_patches, channels * patch_size ** 2).
         patch_size:
             The patch size in pixels used to create the patches.
         channels:
             The number of channels the image must have

     Returns:
         Reconstructed images tensor with shape (batch_size, channels, height, width).
    """
    N, C = patches.shape[0], channels
    patch_h = patch_w = int(patches.shape[1] ** 0.5)
    assert patch_h * patch_w == patches.shape[1]

    images = patches.reshape(shape=(N, patch_h, patch_w, patch_size, patch_size, C))
    images = torch.einsum("nhwpqc->nchpwq", images)
    images = images.reshape(shape=(N, C, patch_h * patch_size, patch_h * patch_size))
    return images


def visualize_patch_recon(sample_im, decoded_im, patches, target, patch_size, num_channels, idx_keep, idx_mask):    
    # make the mask image
    inverted_patch_list = []
    new_keep_index = idx_keep - 1
    empty_patch = torch.zeros(patch_size*patch_size*num_channels)
    for patch_index in range(patches.shape[1]):
        if patch_index in new_keep_index:
            inverted_patch_list.append(torch.squeeze(patches)[patch_index].reshape(1,-1))
        else:
            inverted_patch_list.append(empty_patch.reshape(1,-1))
    inverted_patches = torch.cat(inverted_patch_list, 0)
    inverted_patches = torch.unsqueeze(inverted_patches, 0)
    
    # remake the original image
    new_patch_list = []
    new_target_index = idx_mask - 1
    new_target_index = torch.squeeze(new_target_index)
    for patch_index in range(patches.shape[1]):
        if patch_index in new_keep_index:
            new_patch_list.append(torch.squeeze(patches)[patch_index].reshape(1,-1))
        elif patch_index in new_target_index:
            target_patch_index = torch.nonzero(new_target_index==patch_index).item()
            new_patch_list.append(torch.squeeze(target)[target_patch_index].reshape(1,-1))
    input_patches = torch.cat(new_patch_list, 0)
    input_patches = torch.unsqueeze(input_patches, 0)    
    
    # visualize the reconstructed images here
    recon_patch_list = []
    for patch_index in range(patches.shape[1]):
        if patch_index in new_keep_index:
            recon_patch_list.append(torch.squeeze(patches)[patch_index].reshape(1,-1))
        elif patch_index in new_target_index:
            target_patch_index = torch.nonzero(new_target_index==patch_index).item()
            recon_patch_list.append(torch.squeeze(decoded_im)[target_patch_index].reshape(1,-1))
    recon_patches = torch.cat(recon_patch_list, 0)
    recon_patches = torch.unsqueeze(recon_patches, 0)
    
    input_image = unpatchify(input_patches, patch_size=patch_size, channels=num_channels)
    input_image = torch.squeeze(input_image)
    
    inverted_im = unpatchify(inverted_patches, patch_size=patch_size, channels=num_channels)
    inverted_im = torch.squeeze(inverted_im)
    
    recon_im = unpatchify(recon_patches, patch_size=patch_size, channels=num_channels)
    recon_im = torch.squeeze(recon_im).detach()
    
    return input_image, inverted_im, recon_im


def visualize_channel_token_patch_recon(sample_im, decoded_im, patches, target, patch_size, num_channels, num_patches, idx_keep, idx_mask):
    # here create the empty mask image
    inverted_channel_ims = []
    new_keep_index = idx_keep-1
    empty_channel_patch = torch.zeros(patch_size*patch_size)
    # make the patches easily accessible by the channel index
    patches = patches.reshape(patches.shape[0], num_channels, num_patches, patches.shape[-1])
    for channel_index in range(num_channels):
        channel_patches = torch.squeeze(patches[:,channel_index,:,:], dim=1)
        channel_patch_list = []
        for patch_index in range(num_patches):
            if patch_index in new_keep_index:
                channel_patch_list.append(torch.squeeze(channel_patches)[patch_index].reshape(1,-1))
            else:
                channel_patch_list.append(empty_channel_patch.reshape(1,-1))
            
        inverted_channel_patches = torch.cat(channel_patch_list, 0)
        inverted_channel_patches = torch.unsqueeze(inverted_channel_patches, 0)
        
        inverted_channel_im = unpatchify(inverted_channel_patches, patch_size=patch_size, channels=1)
        inverted_channel_im = torch.squeeze(inverted_channel_im, 1)
        
        inverted_channel_ims.append(inverted_channel_im)
        
    inverted_channel_im = torch.cat(inverted_channel_ims, 0)
    
    # return the original image using this channelwise format
    new_patch_list = []
    new_target_index = idx_mask-1
    new_target_index = torch.squeeze(new_target_index)
    # here reshape the targets
    target = target.reshape(target.shape[0], num_channels, len(new_target_index), target.shape[-1])
    for channel_index in range(num_channels):
        channel_patches = torch.squeeze(patches[:,channel_index,:,:], dim=1)
        channel_target = torch.squeeze(target[:,channel_index,:,:], dim=1)
        channel_patch_list = []
        for patch_index in range(num_patches):
            if patch_index in new_keep_index:
                channel_patch_list.append(torch.squeeze(channel_patches)[patch_index].reshape(1,-1))
            elif patch_index in new_target_index:
                target_patch_index = torch.nonzero(new_target_index==patch_index).item()
                channel_patch_list.append(torch.squeeze(channel_target)[target_patch_index].reshape(1,-1))
        
        input_channel_patches = torch.cat(channel_patch_list, 0)
        input_channel_patches = torch.unsqueeze(input_channel_patches, 0)
        
        input_channel_im = unpatchify(input_channel_patches, patch_size, channels=1)
        input_channel_im = torch.squeeze(input_channel_im, 1)
        
        new_patch_list.append(input_channel_im)
        
    input_channel_im = torch.cat(new_patch_list, 0)
    
    # now we create the reconstructed image using the channelwise format
    recon_patch_list = []
    decoded_im = decoded_im.reshape(decoded_im.shape[0], num_channels, len(new_target_index), decoded_im.shape[-1])
    for channel_index in range(num_channels):
        channel_patches = torch.squeeze(patches[:,channel_index,:,:], dim=1)
        channel_decoded = torch.squeeze(decoded_im[:,channel_index,:,:], dim=1)
        channel_patch_list = []
        for patch_index in range(num_patches):
            if patch_index in new_keep_index:
                channel_patch_list.append(torch.squeeze(channel_patches)[patch_index].reshape(1,-1))
            elif patch_index in new_target_index:
                target_patch_index = torch.nonzero(new_target_index==patch_index).item()
                channel_patch_list.append(torch.squeeze(channel_decoded)[target_patch_index].reshape(1,-1))
        
        channel_recon_patches = torch.cat(channel_patch_list, 0)
        channel_recon_patches = torch.unsqueeze(channel_recon_patches, 0)
        
        channel_recon_im = unpatchify(channel_recon_patches, patch_size, channels=1)
        channel_recon_im = torch.squeeze(channel_recon_im, 1)
        
        recon_patch_list.append(channel_recon_im)
        
    recon_channel_im = torch.cat(recon_patch_list, 0)
    
    return input_channel_im, inverted_channel_im, recon_channel_im        
    
def make_unique(sample_list):
    '''
    Takes some list of string elements and returns a list with the elements being made unique
    '''
    unique_samples, sample_counts = np.unique(sample_list, return_counts=True)
    repeat_indexes = np.nonzero(sample_counts>1)[0]
    repeat_names = unique_samples[repeat_indexes]
    
    if len(repeat_names)>0:
        non_unique_names_dict = {repeat_name: 0 for repeat_name in repeat_names}
    processed_unique_list = []
    for sample_item in sample_list:
        if sample_item not in repeat_names:
            processed_unique_list.append(sample_item)
        else:
            name_count = non_unique_names_dict[sample_item]
            processed_unique_list.append(f'{sample_item}-{name_count}')
            non_unique_names_dict[sample_item] = name_count+1
            
    return processed_unique_list
    
def extract_epoch_from_filename(filename):
    # Use a regular expression to find the epoch number
    match = re.search(r'epoch=(\d+)', filename)
    if match:
        # Extract the epoch number as an integer
        epoch_number = int(match.group(1))
        return epoch_number
    else:
        raise ValueError("Epoch number not found in the filename.")

    