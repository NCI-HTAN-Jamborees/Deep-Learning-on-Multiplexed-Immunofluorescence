import argparse
import os
from pytorch_lightning.plugins.environments import SLURMEnvironment
from dataset_modules.dataset_classes import multichannel_patch_pretrain_dataset
from ssl_model_modules.ssl_models import MAE
from ssl_model_modules.custom_mae_transform import MAETransform
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import lightning as L
import timm
import numpy as np
import pandas as pd
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

parser = argparse.ArgumentParser()
parser.add_argument('--project_root_dir')
parser.add_argument('--val_set_size', default=0.10)
parser.add_argument('--input_im_size', default=64)
parser.add_argument('--num_epochs', default=1000)
parser.add_argument('--dataset_name')
parser.add_argument('--save_every', default=50)
parser.add_argument('--mask_ratio', default=0.50)
parser.add_argument('--batch_size', default=512)
parser.add_argument('--embed_dim', default=768)
parser.add_argument('--num_channels', default=7)
parser.add_argument('--decoder_dim', default=2048)
parser.add_argument('--patch_size', default=16)
parser.add_argument('--model_name')
parser.add_argument('--channel_path', default=None, required=False)
parser.add_argument('--random_channel_select', default=False)
args = parser.parse_args()

project_root_dir = args.project_root_dir
val_set_size = float(args.val_set_size)
input_im_size = int(args.input_im_size)
num_epochs = int(args.num_epochs)
save_every = int(args.save_every)
dataset_name = args.dataset_name
mask_ratio = float(args.mask_ratio)
batch_size = int(args.batch_size)
embed_dim = int(args.embed_dim)
num_channels = int(args.num_channels)
decoder_dim = int(args.decoder_dim)
patch_size = int(args.patch_size)
model_name = args.model_name
channel_path = args.channel_path
ramdom_channel_select = args.random_channel_select

print(f'Mask Ratio: {mask_ratio} \n')
if channel_path is not None and channel_path!='None':
    print(f'Path to filtered channels: {channel_path}')
    channel_arr = np.load(channel_path)
else:
    print('Not filtering channels')
    channel_arr = None

# elif nuclei_train:
#     print('WARNING YOU ARE TRAINING ONLY WITH THE DRAQ5 and Hoechst channels - for debugging only!!!')
#     channel_arr = np.array([0, 55])

SLURMEnvironment.detect = lambda: False
os.environ['SLURM_JOB_NAME'] = 'interactive'

def load_train_objs(train_patch_dir,
                    num_channels,
                    train_frame,
                    val_frame,
                    ssl_transform,
                    channel_arr=None):
    # set dataset here
    train_dataset = multichannel_patch_pretrain_dataset(train_patch_dir, num_channels, train_frame, channel_list=channel_arr, sample_transform=ssl_transform)

    val_dataset = multichannel_patch_pretrain_dataset(train_patch_dir, num_channels, val_frame, channel_list=channel_arr, sample_transform=ssl_transform)

    if channel_arr is not None:
        num_channels = len(channel_arr)
        print(f'New Number of channels: {num_channels}')

    # set model here
    vit = timm.create_model('vit_base_patch16_224', patch_size=patch_size, img_size=input_im_size, embed_dim=embed_dim, in_chans=num_channels, pretrained=False)
    model = MAE(vit=vit, decoder_dim=decoder_dim, mask_ratio=mask_ratio, num_channels=num_channels)
    # model.decoder.decoder_pred = nn.Linear(in_features=decoder_dim, out_features=patch_size*patch_size*num_channels, bias=True)
    return train_dataset, val_dataset, model

def prepare_dataloader(dataset: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=2
    )

np.random.seed(0)
world_size = torch.cuda.device_count()
# train_dir = f'/ix/yufeihuang/Hugh/pitt_melanoma_dl/{dataset_name}'
train_dir = f'{project_root_dir}/{dataset_name}'
train_patch_dir = os.path.join(train_dir, 'patches')
# train_patch_fnames = os.listdir(train_patch_dir)
# patch_frame = pd.DataFrame(np.array(train_patch_fnames), columns=['patch_fnames'])

# actually grab the dataframe that we are using for training
patch_frame = pd.read_csv(os.path.join(train_dir, 'patch_csvs', 'patch_info.csv'))
patch_frame['split_id'] = 'train'
# # randomly sample a 10% validation set
size_10_perc = int(np.ceil(patch_frame.shape[0]*val_set_size))
rand_indices = np.random.choice(patch_frame.shape[0], size=size_10_perc, replace=False)
patch_frame['split_id'].iloc[rand_indices] = 'val'

# split the training and validation datasets 
train_frame = patch_frame[patch_frame['split_id']=='train']
val_frame = patch_frame[patch_frame['split_id']=='val']

channels_to_use = np.arange(num_channels)
ssl_transform = MAETransform(input_size=input_im_size, normalize=False)
torch.set_float32_matmul_precision('medium')

print('Preparing Datasets...')
train_dataset, val_dataset, model = load_train_objs(train_patch_dir,
                                                    num_channels,
                                                    train_frame,
                                                    val_frame,
                                                    ssl_transform,
                                                    channel_arr=channel_arr)


train_loader = prepare_dataloader(train_dataset, batch_size, shuffle=True)
val_loader = prepare_dataloader(val_dataset, batch_size, shuffle=False)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

# stats_root = '/ix/yufeihuang/Hugh/pitt_melanoma_dl/ssl_model_stats'
# model_root = '/ix/yufeihuang/Hugh/pitt_melanoma_dl/ssl_models'
stats_root = os.path.join(project_root_dir, 'ssl_model_stats')
model_root = os.path.join(project_root_dir, 'ssl_models')
if not os.path.exists(os.path.join(project_root_dir, 'ssl_model_stats')):
    os.mkdir(os.path.join(project_root_dir, 'ssl_model_stats'))
if not os.path.exists(os.path.join(project_root_dir, 'ssl_models')):
    os.mkdir(os.path.join(project_root_dir, 'ssl_models'))
if not os.path.exists(os.path.join(stats_root, model_name)):
    os.mkdir(os.path.join(stats_root, model_name))
if not os.path.exists(os.path.join(model_root, model_name)):
    os.mkdir(os.path.join(model_root, model_name))

callbacks = [
    ModelCheckpoint(dirpath=f'{model_root}/{model_name}/saved_mae_models_patch_{patch_size}_token_{embed_dim}_mask_{mask_ratio}/',
                    filename='mae_{epoch}_{train_loss:.4f}',
                    save_top_k=-1,
                    save_last=True,
                    mode="min",
                    every_n_epochs=save_every,
                    monitor="train_loss")
]

logger = CSVLogger(save_dir=f"{stats_root}/{model_name}/mae_model_stats_patch_{patch_size}_token_{embed_dim}_mask_{mask_ratio}/",
                   name="mae_model")

print('Instantiating Trainer')
trainer = L.Trainer(
    max_epochs=num_epochs,
    callbacks=callbacks,
    accelerator=accelerator,
    devices=world_size,
    strategy=DDPStrategy(find_unused_parameters=True),
    precision='16-mixed',
    logger=logger,
    log_every_n_steps=1,
    deterministic=True,
    use_distributed_sampler=True,
)

print('Training Model')
trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)


