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
parser.add_argument('--cell_frame_dir')
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
cell_frame_dir = args.cell_frame_dir
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

sample_name = im_fname.split('.')[0]
patient_name = im_fname.split('_')[0]

cell_frame_path = f'{cell_frame_dir}/{patient_name}.csv'

# print(f'Patient Label: {sample_name}')

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
# sample_name = im_fname

# subset the dataframe by sample name
subset_frame = image_frame[image_frame[sample_key]==im_fname]

# print(f'Subset Frame: {subset_frame}')

# if len(subset_frame)<1:
#     print(f'Skipping sample')
#     continue

    
initial_im = tifffile.imread(os.path.join(image_dir, im_fname))

# print(f'IM Shape: {initial_im.shape}')

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

# print(f'Current Im Shape: {current_im.shape}')

row_roi_size = current_im.shape[1]
col_roi_size = current_im.shape[2]
row_index = subset_frame.row_index.item()
col_index = subset_frame.col_index.item()

cell_frame = pd.read_csv(cell_frame_path)

# print(f'Cell Frame Shape: {cell_frame.shape}')

coord_arr = cell_frame[['row_centroid', 'column_centroid']].values
valid_row_indices = np.nonzero((coord_arr[:,0]>=row_index) & (coord_arr[:,0]<=row_index+row_roi_size))[0]
valid_col_indices = np.nonzero((coord_arr[:,1]>=col_index) & (coord_arr[:,1]<=col_index+col_roi_size))[0]
valid_cell_indices = np.intersect1d(valid_row_indices, valid_col_indices)

# if there are no valid cell indices then continue
# print(f'Valid Cell Indices: {len(valid_cell_indices)}')

region_cell_frame = cell_frame.iloc[valid_cell_indices]
region_cell_frame['adjusted_row_coords'] = region_cell_frame['row_centroid']-row_index
region_cell_frame['adjusted_col_coords'] = region_cell_frame['column_centroid']-col_index

# print(f'Cell Frame Shape: {region_cell_frame.shape}')

patch_count = 0
patch_list = []

# go through every cell in the subset frame
for count, row in tqdm(region_cell_frame.iterrows()):
    border_im = 'non-border'
    
    top_left_row = int(row.adjusted_row_coords)-patch_size//2
    top_left_col = int(row.adjusted_col_coords)-patch_size//2
    
    if top_left_row<0:
        top_left_row=0
        border_im = 'border'
    elif top_left_row+patch_size>row_roi_size:
        top_left_row -= patch_size
        border_im = 'border'

    if top_left_col<0:
        top_left_col = 0
        border_im = 'border'
    elif top_left_col+patch_size>col_roi_size:
        top_left_col -= patch_size
        border_im = 'border'

    subset_frame = subset_frame.assign(cell_id=int(row['CellID']))
    subset_frame = subset_frame.assign(patch_centroids_row_orig=row.row_centroid)
    subset_frame = subset_frame.assign(patch_centroids_col_orig=row.column_centroid)
    subset_frame = subset_frame.assign(patch_centroids_row_adj=row.adjusted_row_coords)
    subset_frame = subset_frame.assign(patch_centroids_col_adj=row.adjusted_col_coords)
    subset_frame = subset_frame.assign(patch_fname=f'{sample_name}_patch_{patch_count}.tiff')
    subset_frame = subset_frame.assign(boundary_status=border_im)

    patch_list.append(subset_frame)
    
    min_row_ind, max_row_ind = top_left_row, top_left_row+patch_size
    min_col_ind, max_col_ind = top_left_col, top_left_col+patch_size
    
    img_patch = current_im[:, min_row_ind:max_row_ind, min_col_ind:max_col_ind]
    tifffile.imsave(os.path.join(results_dir, 'patches', f'{sample_name}_patch_{patch_count}.tiff'), img_patch)

    patch_count += 1

# print(f'Patch List Length: {len(patch_list)}')

# combine the cell frames and save as we go
final_patch_frame = pd.concat(patch_list, axis=0)
final_patch_frame = final_patch_frame.reset_index(drop=True)
final_patch_frame = final_patch_frame.drop(columns=['Unnamed: 0'])
final_patch_frame.to_csv(os.path.join(results_dir, 'region_patch_csvs', f'{sample_name}_patch_info.csv'), index=False)



