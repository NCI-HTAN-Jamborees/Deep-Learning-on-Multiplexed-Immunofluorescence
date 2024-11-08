import tifffile
import numpy as np
from skimage import measure, draw
import pandas as pd
from tqdm import tqdm
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cell_frame_path')
parser.add_argument('--image_path')
parser.add_argument('--sample_name')
parser.add_argument('--roi_row_size', default=1000)
parser.add_argument('--roi_col_size', default=1300)
parser.add_argument('--roi_cell_count_threshold', default=1000)
parser.add_argument('--boundary_image_path')
parser.add_argument('--x_centroid_key', default='column_centroid')
parser.add_argument('--y_centroid_key', default='row_centroid')
parser.add_argument('--overlap', default=0.20)
parser.add_argument('--single_channel', action='store_true')
parser.add_argument('--save_roi_dir')

args = parser.parse_args()

cell_frame_path = args.cell_frame_path
image_path = args.image_path
sample_name = args.sample_name
roi_row_size = int(args.roi_row_size)
roi_col_size = int(args.roi_col_size)
cell_count_threshold = int(args.roi_cell_count_threshold)
prc_overlap = float(args.overlap)
boundary_image_path = args.boundary_image_path
single_channel = args.single_channel
y_centroid_key = args.y_centroid_key
x_centroid_key = args.x_centroid_key
save_roi_dir = args.save_roi_dir

# src_im = tifffile.imread(image_path)
src_im = tifffile.imread(os.path.join(image_path, f'{sample_name}.ome.tif'))

# cell_frame = pd.read_csv(cell_frame_path)
cell_frame = pd.read_csv(os.path.join(cell_frame_path, f'{sample_name}.csv'))

coord_arr = cell_frame[[y_centroid_key, x_centroid_key]].values

print(f'Coordinate Shape: {coord_arr.shape}')

print(f'Overlap: {prc_overlap}')
print(f'Row ROI Size {roi_row_size} - Col ROI Size {roi_col_size}')
print(f'Overlap Type {type(prc_overlap)} - Row Size Type: {type(roi_row_size)} - Col Size Type: {type(roi_col_size)}')
print(f'Source Image Shape: {src_im.shape}')
print(f'Cell Count Threshold: {cell_count_threshold}')

overlap_row_size = roi_row_size-int((prc_overlap*roi_row_size))
overlap_col_size = roi_col_size-int((prc_overlap*roi_col_size))

if not single_channel:
    wsi_row_size = src_im.shape[1]
    wsi_col_size = src_im.shape[2]
else:
    wsi_row_size = src_im.shape[0]
    wsi_col_size = src_im.shape[1]

row_end = wsi_row_size//overlap_row_size+1
col_end = wsi_col_size//overlap_col_size+1


print(f'Row Extend: {row_end}')
print(f'Col Extend: {col_end}')

print(f'Total Number of rois: {row_end*col_end}')

# for now we do not need to actually grab the ROIs
# we can just draw the roi boundaries on an empty template
boundary_template_image = np.zeros((wsi_row_size, wsi_col_size), dtype=np.uint8)

num_selected_regions = 0
overall_roi_id = 0

print(f'Creating ROI template image')
for row_index in tqdm(range(int(row_end))):
    for col_index in range(int(col_end)):
        overall_roi_id += 1
        new_row_index = row_index*int(overlap_row_size)
        new_col_index = col_index*int(overlap_col_size)

        # check if we exceed our boundaries, if so push them in and resample
        if new_row_index+roi_row_size>=wsi_row_size:
            new_row_index = wsi_row_size-roi_row_size-1
        if new_col_index+roi_col_size>=wsi_col_size:
            new_col_index = wsi_col_size-roi_col_size-1

        line_coords = [
            (new_row_index, new_col_index, new_row_index+roi_row_size, new_col_index),
            (new_row_index, new_col_index, new_row_index, new_col_index+roi_col_size),
            (new_row_index+roi_row_size, new_col_index, new_row_index+roi_row_size, new_col_index+roi_col_size),
            (new_row_index, new_col_index+roi_col_size, new_row_index+roi_row_size, new_col_index+roi_col_size)
        ]

        # get number of cells falling within the ROI - skip ROIs with cell counts less than our threshold
        valid_row_indices = np.nonzero((coord_arr[:,0]>=new_row_index) & (coord_arr[:,0]<=new_row_index+roi_row_size))[0]
        valid_col_indices = np.nonzero((coord_arr[:,1]>=new_col_index) & (coord_arr[:,1]<=new_col_index+roi_col_size))[0]
        valid_cell_indices = np.intersect1d(valid_row_indices, valid_col_indices)

        if len(valid_cell_indices)<cell_count_threshold:
            continue

        num_selected_regions += 1

        for coord_set in line_coords:
            r0, c0, r1, c1 = coord_set
            if r1>=wsi_row_size:
                r1=wsi_row_size-1
            elif c1>=wsi_col_size:
                c1=wsi_col_size-1

            rr, cc = draw.line(r0, c0, r1, c1)            
            boundary_template_image[rr, cc] = 1

        # if the image is valid then save it
        if not single_channel:
            region_im = src_im[:, new_row_index:new_row_index+roi_row_size, new_col_index:new_col_index+roi_col_size]
        else:
            region_im = src_im[new_row_index:new_row_index+roi_row_size, new_col_index:new_col_index+roi_col_size]

        tifffile.imwrite(os.path.join(save_roi_dir, f'{sample_name}_roi_{overall_roi_id}_row_{new_row_index}_col_{new_col_index}.ome.tif'), region_im)


print(f'Number of Selected ROIs: {num_selected_regions}')

# save the template_image
# tifffile.imwrite(boundary_image_path, boundary_template_image)
tifffile.imwrite(os.path.join(boundary_image_path, f'{sample_name}_boundary_image.tif'), boundary_template_image)
