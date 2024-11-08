import numpy as np
import tifffile
import pandas as pd
import imageio
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir')
parser.add_argument('--image_file_name')
parser.add_argument('--image_frame_path')
parser.add_argument('--channel_path', default=None)
parser.add_argument('--patch_size', default=64)
parser.add_argument('--max_channels', default=None)
parser.add_argument('--norm_type')
parser.add_argument('--clip_thresh', default=99)
parser.add_argument('--normalization_dir', default=None)
parser.add_argument('--sample_key', default='image_fnames')
parser.add_argument('--results_dir')


args = parser.parse_args()
image_dir = args.image_dir
im_fname = args.image_file_name
image_frame_path = args.image_frame_path
channel_path = args.channel_path
patch_size = int(args.patch_size)
max_channels = int(args.max_channels)
norm_type = args.norm_type
clip_thresh = int(args.clip_thresh)
normalization_dir = args.normalization_dir
sample_key = args.sample_key
results_dir = args.results_dir

image_frame = pd.read_csv(image_frame_path)
# image_fnames = os.listdir(image_dir)


if 'z_score' in norm_type:
    if normalization_dir is None:
        normalization_dir = image_dir
        
    norm_fnames = os.listdir(normalization_dir)

# if we do not have a folder for the results then create it
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
        
# create folders for the images and the cell dataframes
if not os.path.exists(os.path.join(results_dir, 'patches')):
    os.mkdir(os.path.join(results_dir, 'patches'))
if not os.path.exists(os.path.join(results_dir, 'region_patch_csvs')):
    os.mkdir(os.path.join(results_dir, 'region_patch_csvs'))

# if there are channels to filter load then here
if channel_path is not None and channel_path!='None':
    channels_to_keep = np.load(channel_path)
else:
    channels_to_keep = None
    

# get parameters for z-score normalization here
# we use the training set for normalization so no information will leak
if 'z_score' in norm_type:
    print(f'Performing z-score normalization...')
    means_list = []
    std_list = []
    im_size = []
    for norm_fname in norm_fnames:
        sample_name = norm_fname
        subset_frame = image_frame[image_frame[sample_key]==sample_name]
    
        if len(subset_frame)<1:
            continue
        
        # initial_im = imageio.volread(os.path.join(normalization_dir, im_fname))
        initial_im = tifffile.imread(os.path.join(normalization_dir, norm_fname))
        # if there are more than three dimensions we collapse the channel dimension
        if len(initial_im.shape)>3:
            initial_im = initial_im.reshape(-1, initial_im.shape[2], initial_im.shape[3])
        
        if max_channels is not None and max_channels!='None':
            initial_im = initial_im[0:max_channels]
        
        # if you want to filter channels then do that here
        if channel_path is not None and channel_path!='None':
            initial_im = initial_im[channels_to_keep, :, :]
        
        # if we want to do mean div normalization we must perform it here first
        if 'mean' in norm_type:
            chan_size, row_size, col_size = initial_im.shape
            sample_proj = initial_im.reshape(chan_size, -1)
            channel_means = np.mean(sample_proj, axis=1).reshape(-1,1)
            sample_proj = sample_proj/channel_means
            initial_im = sample_proj.reshape(chan_size, row_size, col_size)

        if 'log10' in norm_type:
            initial_im = np.log10(initial_im+0.5)

        if 'pre_clip' in norm_type:
            perc_thresh = np.percentile(initial_im, clip_thresh)
            initial_im[initial_im>perc_thresh] = perc_thresh        
            
        channel_means = np.array([np.mean(initial_im[chan,:,:].flatten()) for chan in range(initial_im.shape[0])]).reshape(1,-1)
        channel_stds = np.array([np.sum(np.square(initial_im[chan,:,:].flatten()-np.mean(initial_im[chan,:,:].flatten()))) for chan in range(initial_im.shape[0])]).reshape(1,-1)
        chan_pixels = np.array([len(initial_im[chan,:,:].flatten()) for chan in range(initial_im.shape[0])]).reshape(1,-1)
        means_list.append(channel_means)
        std_list.append(channel_stds)
        im_size.append(chan_pixels)

    mean_arr = np.concatenate(means_list, axis=0)
    std_arr = np.concatenate(std_list, axis=0)
    size_arr = np.concatenate(im_size, axis=0)
    mean_arr = np.mean(mean_arr, axis=0)
    std_arr = np.sum(std_arr, axis=0)
    size_arr = np.sum(size_arr, axis=0)
    std_arr = np.sqrt(std_arr/size_arr)

    # verify that the shapes of the mean and std arrays are correct
    print(f'Mean Arr Shape: {mean_arr.shape}')
    print(f'Std Arr Shape: {std_arr.shape}')
    print(f'Mean Arr: {mean_arr[0:10]}')
    print(f'Std Arr: {std_arr[0:10]}')


# create a list to hold all sample dataframes
patch_list = []

# # go through each image and process
# for im_fname in tqdm(image_fnames):
# just process a single roi
sample_name = im_fname

# subset the dataframe by sample name
subset_frame = image_frame[image_frame[sample_key]==sample_name]

# if len(subset_frame)<1:
#     print(f'Skipping sample')
#     continue

    
initial_im = tifffile.imread(os.path.join(image_dir, im_fname))

# if there are more than three dimensions we collapse the channel dimension
if len(initial_im.shape)>3:
    initial_im = initial_im.reshape(-1, initial_im.shape[2], initial_im.shape[3])

if max_channels is not None and max_channels!='None':
    initial_im = initial_im[0:max_channels]
# if you want to filter channels then do that here
if channel_path is not None and channel_path!='None':
    initial_im = initial_im[channels_to_keep, :, :]


# for normalization we apply the z-score normalization here   
if 'mean' in norm_type:
    chan_size, row_size, col_size = initial_im.shape
    sample_proj = initial_im.reshape(chan_size, -1)
    channel_means = np.mean(sample_proj, axis=1).reshape(-1,1)
    sample_proj = sample_proj/channel_means
    initial_im = sample_proj.reshape(chan_size, row_size, col_size)

if 'pre_clip' in norm_type:
    perc_thresh = np.percentile(initial_im, clip_thresh)
    initial_im[initial_im>perc_thresh] = perc_thresh

if 'log10' in norm_type:
    initial_im = np.log10(initial_im+0.5)

if 'z_score' in norm_type:
    current_im_list = []
    for channel in range(len(mean_arr)):
        current_im_list.append((initial_im[channel, :, :] - mean_arr[channel]) / (std_arr[channel] + 1e-16))
    initial_im = np.stack(current_im_list, axis=0)

if 'post_clip' in norm_type:
    perc_thresh = np.percentile(initial_im, clip_thresh)
    initial_im[initial_im>perc_thresh] = perc_thresh

if 'min_max' in norm_type:
    new_im = []
    for channel in range(initial_im.shape[0]):
        im_slice = initial_im[channel, :, :]
        # don't normalize blank channels
        if np.max(im_slice) < 1:
            new_im.append(im_slice[np.newaxis, :, :])
        else:
            im_slice = (im_slice - np.min(im_slice)) / (np.max(im_slice) - np.min(im_slice))
            new_im.append(im_slice[np.newaxis, :, :])

    initial_im = np.concatenate(new_im, axis=0)

current_im = initial_im

max_row_iter = np.floor(current_im.shape[1]/patch_size)
max_col_iter = np.floor(current_im.shape[2]/patch_size)
patch_count = 0

for row_count in range(int(max_row_iter)):
    for col_count in range(int(max_col_iter)):
        min_row_ind, max_row_ind = row_count * patch_size, (row_count + 1) * patch_size
        min_col_ind, max_col_ind = col_count * patch_size, (col_count + 1) * patch_size

        # add the patch column to the subset frame - also add patch centroids
        row_centroid = (row_count + 1) * patch_size - (patch_size // 2)
        col_centroid = (col_count + 1) * patch_size - (patch_size // 2)
        subset_frame = subset_frame.assign(patch_id=f'Patch_{patch_count}')
        subset_frame = subset_frame.assign(patch_centroids_row=row_centroid)
        subset_frame = subset_frame.assign(patch_centroids_col=col_centroid)
        subset_frame = subset_frame.assign(patch_fname=f'{sample_name}_patch_{patch_count}.tiff')
        # add the data split info to the new dataframe as well
        if 'train' in image_dir:
            subset_frame = subset_frame.assign(data_split='train')
        elif 'test' in image_dir:
            subset_frame = subset_frame.assign(data_split='test')

        patch_list.append(subset_frame)

        img_patch = current_im[:, min_row_ind:max_row_ind, min_col_ind:max_col_ind]
        # save the image patches as tif files
        tifffile.imsave(os.path.join(results_dir, 'patches', f'{sample_name}_patch_{patch_count}.tiff'), img_patch)

        patch_count += 1

# combine the cell frames and save as we go
save_sample_name = sample_name.split('.')[0]
final_patch_frame = pd.concat(patch_list, axis=0)
final_patch_frame.to_csv(os.path.join(results_dir, 'region_patch_csvs', f'{save_sample_name}_patch_info.csv'), index=False)





