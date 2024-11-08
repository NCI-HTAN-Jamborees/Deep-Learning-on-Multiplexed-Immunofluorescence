import lightning as L
import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.optim as optim


class GatedAttention(L.LightningModule):
    def __init__(self, orig_embed_dim=768, M_dim=768, L_dim=128, num_branches=1):
        super(GatedAttention, self).__init__()
        # new params for 4x4 model
        self.orig_embed_dim = orig_embed_dim
        self.M = M_dim
        self.L = L_dim
        self.ATTENTION_BRANCHES = num_branches

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.orig_embed_dim, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor(x)

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def forward_multi_prob(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor(x)

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        final_prob = torch.zeros(2)
        final_prob[0] = 1.0-Y_prob
        final_prob[1] = Y_prob

        return final_prob, Y_hat, A

    def forward_single_instance(self, x):

        H = self.feature_extractor(x)
        Y_prob = self.classifier(H)

        # instead of having a single neuron that represents probability of a one
        # we need to create two separate ones that represent probability of a zero and probability of a one
        # we don't necessarily need softmax so we can check and see what happens with and without it

        final_prob = torch.zeros(2)
        final_prob[0] = 1.0-Y_prob
        final_prob[1] = Y_prob

        return final_prob


    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
    
    def training_step(self, batch, batch_idx):
        data, bag_label = batch        
        # data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = self.calculate_objective(data, bag_label)
        error, predicted_label = self.calculate_classification_error(data, bag_label)
        
        self.log('train_loss', loss)
        self.log('train_error', error)
        return loss
    
    def test_step(self, batch, batch_idx):
        data, bag_label = batch
        # data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = self.calculate_objective(data, bag_label)
        error, predicted_label = self.calculate_classification_error(data, bag_label)
        
        self.log('test_loss', loss)
        self.log('test_error', error)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        data, _ = batch
        pred_prob, pred, attention_weights = self.forward(data)
        
        return pred_prob, pred, attention_weights


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=10e-5)
