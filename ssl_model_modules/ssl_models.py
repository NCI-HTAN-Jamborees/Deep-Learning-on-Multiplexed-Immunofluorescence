import lightning as L
import timm
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
import torch.nn as nn
import torch


class MAE(L.LightningModule):
    def __init__(self, vit, mask_ratio=0.50, decoder_dim=512, num_channels=56):
        super().__init__()

        # initially 4096
        decoder_dim = decoder_dim
        self.mask_ratio = mask_ratio
        self.patch_size = vit.patch_embed.patch_size[0]
        self.criterion = nn.MSELoss()

        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=1,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )
        
        self.decoder.decoder_pred = nn.Linear(in_features=decoder_dim, out_features=self.patch_size*self.patch_size*num_channels, bias=True)

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred
    
    def get_pred_im(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device
        )
        x_encoded = self.forward_encoder(images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )
        
        patches = utils.patchify(images, self.patch_size)
        target = utils.get_at_index(patches, idx_mask-1)
        
        return x_pred, patches, target, idx_keep, idx_mask


    def training_step(self, batch, batch_idx):
        images = batch[0]
        # views = batch[0]
        # images = views[0]  # views contains only a single view
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        train_loss = self.criterion(x_pred, target)
        self.log('train_loss', train_loss, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        images = batch[0]
        # views = batch[0]
        # images = views[0]
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device
        )

        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        val_loss = self.criterion(x_pred, target)
        # log the val loss here
        self.log('val_loss', val_loss, sync_dist=True)
        
    def predict_step(self, batch, batch_idx):
        if type(batch)==list:
            images = batch[0]
        else:
            images = batch
            
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device
        )
        
        # get the embeddings and grab the classification token
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        
        return x_encoded[:,0]


    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1.5e-4)
        return optim


