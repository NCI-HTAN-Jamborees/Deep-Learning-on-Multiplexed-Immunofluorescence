import os
from dataset_modules.dataset_classes import multichannel_patch_pretrain_dataset
from ssl_model_modules.ssl_models import MAE
from torch.utils.data import DataLoader
from ssl_model_modules.custom_mae_transform import MAETransform
import pandas as pd
import timm
from lightly.models import utils
import matplotlib.pyplot as plt
from util_modules.helpers import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--project_root_dir')
parser.add_argument('--dataset_name')
parser.add_argument('--num_viz_images')
parser.add_argument('--model_name')
parser.add_argument('--model_fname')
parser.add_argument('--input_im_size', default=64)
parser.add_argument('--patch_size')
parser.add_argument('--mask_ratio', default=0.50)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--embed_dim', default=768)
parser.add_argument('--num_channels')
parser.add_argument('--save_fmt', default='.jpg')
parser.add_argument('--decoder_dim', default=2048)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--channel_path', default=None, required=False)
args = parser.parse_args()

project_root_dir = args.project_root_dir
dataset_name = args.dataset_name
model_name = args.model_name
model_fname = args.model_fname
num_viz_images = int(args.num_viz_images)
input_im_size = int(args.input_im_size)
patch_size = int(args.patch_size)
mask_ratio = float(args.mask_ratio)
batch_size = int(args.batch_size)
num_channels = int(args.num_channels)
embed_dim = int(args.embed_dim)
decoder_dim = int(args.decoder_dim)
save_fmt = args.save_fmt
shuffle = args.shuffle
channel_path = args.channel_path

if channel_path is not None and channel_path!='None':
    channel_arr = np.load(channel_path)
else:
    channel_arr = None


if num_viz_images>batch_size:
    raise Exception('Number of images to show is greater than the batch size. Please either increase the patch size or show less images')

dataset_dir = os.path.join(project_root_dir, dataset_name)
patch_dir = os.path.join(dataset_dir, 'patches')
patch_frame = pd.read_csv(os.path.join(dataset_dir, 'patch_csvs', 'patch_info.csv'))

ssl_transform = MAETransform(input_size=input_im_size, normalize=False)

patch_dataset = multichannel_patch_pretrain_dataset(
    patch_dir,
    num_channels,
    patch_frame,
    channel_list=channel_arr,
    sample_transform=ssl_transform
)

patch_loader = DataLoader(
    patch_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=True,
    num_workers=2
)

if channel_arr is not None:
    num_channels=len(channel_arr)

vit = timm.create_model('vit_base_patch16_224', patch_size=patch_size, img_size=input_im_size, embed_dim=embed_dim, in_chans=num_channels, pretrained=False)
ssl_model = MAE.load_from_checkpoint(
    os.path.join(project_root_dir, 'ssl_models', model_name, f'saved_mae_models_patch_{patch_size}_token_{embed_dim}_mask_{mask_ratio}', model_fname), 
    vit=vit, 
    decoder_dim=decoder_dim, 
    mask_ratio=mask_ratio, 
    num_channels=num_channels
)

sample_ims = next(iter(patch_loader))

# create a folder to hold the reconstruction images
recon_viz_folder = os.path.join(project_root_dir, 'ssl_recon_visualizations', model_name)
if not os.path.exists(os.path.join(project_root_dir, 'ssl_recon_visualizations')):
    os.mkdir(os.path.join(project_root_dir, 'ssl_recon_visualizations'))

if not os.path.exists(recon_viz_folder):
    os.mkdir(recon_viz_folder)

# sample images from the number of images we want to sample
im_indices = np.random.choice(batch_size, num_viz_images, replace=False)

for image_index in im_indices:
    # just select a random image in the batch
    example_image = torch.unsqueeze(sample_ims[0][image_index], 0)

    sequence_length = ssl_model.backbone.sequence_length
    idx_keep, idx_mask = utils.random_token_mask(
        size=(1, sequence_length),
        mask_ratio=mask_ratio,
    )

    x_pred, patches, target, idx_keep, idx_mask = ssl_model.get_pred_im(example_image)

    orig_im, masked_im, recon_im = visualize_patch_recon(
        example_image, 
        x_pred, 
        patches, 
        target, 
        patch_size, 
        num_channels, 
        idx_keep, 
        idx_mask
    )

    # if the number of channels is greater than 10 then sample channels - otherwise go through all channels
    if num_channels>10:
        viz_channel_arr = np.random.choice(num_channels, 10, replace=False)
    else:
        viz_channel_arr = np.arange(num_channels)

    # create a folder for each image that we are showing and save all the channel visualizations within that folder
    recon_sample_folder = os.path.join(recon_viz_folder, f'sample_im_{image_index}')
    if not os.path.exists(recon_sample_folder):
        os.mkdir(recon_sample_folder)

    for channel_index in viz_channel_arr:
        orig_slice = orig_im[channel_index,:,:]
        masked_slice = masked_im[channel_index,:,:]
        recon_slice = recon_im[channel_index,:,:]
        fig, axes = plt.subplots(1,3,figsize=(10,2))
        im_1 = axes[0].imshow(orig_slice, cmap='bone')
        fig.colorbar(im_1, ax=axes[0])
        im_2 = axes[1].imshow(masked_slice, cmap='bone')
        fig.colorbar(im_2, ax=axes[1])
        im_3 = axes[2].imshow(recon_slice, cmap='bone')
        fig.colorbar(im_3, ax=axes[2])
        plt.suptitle(f'Channel Index: {channel_index} Original - Mask - Reconstructed Images')
        # save high quality reconstruction figures
        fig.savefig(os.path.join(recon_sample_folder, f'sample_im_{image_index}_c_{channel_index}_recon_viz.{save_fmt}'), dpi=450)
        plt.close()

