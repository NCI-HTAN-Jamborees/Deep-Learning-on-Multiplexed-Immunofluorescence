import argparse
from pytorch_lightning.plugins.environments import SLURMEnvironment
import os
import torch
from dataset_modules.dataset_classes import multichannel_patch_pretrain_dataset
from ssl_model_modules.ssl_models import MAE
from ssl_model_modules.custom_mae_transform import MAETransform
import anndata as ad
import timm
import numpy as np
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--project_root_dir')
parser.add_argument('--model_dir')
parser.add_argument('--model_fname')
parser.add_argument('--dataset_name')
parser.add_argument('--model_name')
parser.add_argument('--num_channels')
parser.add_argument('--channel_path', default=None, required=False)
parser.add_argument('--im_size', default=64)
parser.add_argument('--patch_size', default=16)
parser.add_argument('--embed_dim', default=768)
parser.add_argument('--mask_ratio', default=0.0)
parser.add_argument('--decoder_dim', default=2048)
parser.add_argument('--batch_size', default=128)

args = parser.parse_args()

SLURMEnvironment.detect = lambda: False
os.environ['SLURM_JOB_NAME'] = 'interactive'
torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()

project_root_dir = args.project_root_dir
num_channels = int(args.num_channels)
im_size = int(args.im_size)
model_name = args.model_name
patch_size = int(args.patch_size)
embed_dim = int(args.embed_dim)
dataset_name = args.dataset_name
mask_ratio = float(args.mask_ratio)
decoder_dim = int(args.decoder_dim)
batch_size = int(args.batch_size)
model_dir = args.model_dir
model_fname = args.model_fname
channel_path = args.channel_path

if channel_path is not None and channel_path!='None':
    channel_arr = np.load(channel_path)
else:
    channel_arr = None

patch_dir = os.path.join(project_root_dir, dataset_name, 'patches')
info_dir = os.path.join(project_root_dir, dataset_name, 'patch_csvs')

combined_frame = pd.read_csv(os.path.join(info_dir, 'patch_info.csv'))
combined_frame = combined_frame.reset_index(drop=True)
ssl_transform = MAETransform(input_size=im_size, normalize=False)

combined_dataset = multichannel_patch_pretrain_dataset(patch_dir, num_channels, combined_frame, channel_list=channel_arr, sample_transform=ssl_transform)

if channel_arr is not None:
    num_channels = len(channel_arr)

vit = timm.create_model('vit_base_patch16_224', patch_size=patch_size, img_size=im_size, embed_dim=embed_dim, in_chans=num_channels, pretrained=False)
model = MAE.load_from_checkpoint(os.path.join(model_dir, model_fname), vit=vit, decoder_dim=decoder_dim, mask_ratio=mask_ratio, num_channels=num_channels)

model.eval()

if not os.path.exists(f'{project_root_dir}/ssl_embeddings'):
    os.mkdir(f'{project_root_dir}/ssl_embeddings')

embedding_dir = f'{project_root_dir}/ssl_embeddings/{model_name}/mae_patch_{patch_size}_token_{embed_dim}_chan_{num_channels}'
combined_embed_names = 'combined_embeddings.h5ad'

if not os.path.exists(f'{project_root_dir}/ssl_embeddings/{model_name}'):
    os.mkdir(f'{project_root_dir}/ssl_embeddings/{model_name}')

if not os.path.exists(embedding_dir):
    os.mkdir(embedding_dir)
    

combined_loader = DataLoader(
    combined_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=4
)

trainer = L.Trainer()

combined_embeddings = trainer.predict(model, combined_loader)

combined_embeddings = torch.cat(combined_embeddings, dim=0)
combined_embeddings = combined_embeddings.detach().numpy()

print(f'Embedding Size: {combined_embeddings.shape}')

embedding_adata = ad.AnnData(combined_embeddings, obs=combined_frame)
embedding_adata.write_h5ad(os.path.join(embedding_dir, combined_embed_names))

