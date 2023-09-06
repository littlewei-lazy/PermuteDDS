from einops.layers.torch import Rearrange, Reduce

import torch
from torch import nn, einsum


class MLP_block(nn.Module):
    def __init__(self, input_size, expansion_factor, dropout=0.5):
        super().__init__()
        # hidden_size = int(input_size * expansion_factor)
        hidden_size = expansion_factor
        self.net = nn.Sequential(
            # nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            Swish(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size)
            # nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class MLP_Communicator(nn.Module):
    def __init__(self, token, channel, expansion_factor, dropout=0.2):
        super(MLP_Communicator, self).__init__()

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(token),
            Rearrange('b n d -> b d n'),
            MLP_block(input_size=channel, expansion_factor=expansion_factor, dropout=dropout),
            Rearrange('b d n -> b n d'),
            # nn.Dropout(dropout)
        )

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(token),
            MLP_block(input_size=token, expansion_factor=expansion_factor, dropout=dropout),
            # nn.Dropout(dropout)  
        )

        # self.full_mixer = nn.Sequential(
        #     # Rearrange('b n d -> b (d n)'),
        #     nn.LayerNorm(token*channel),
        #     MLP_block(input_size=token*channel, expansion_factor=expansion_factor, dropout=dropout),
        #     # nn.Dropout(dropout)
        # )
        
    def forward(self, x): 
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        # x = x + self.token_mixer(x)
        # rx = rearrange(x, 'b n d -> b (d n)')
        # x = rx +  self.full_mixer(rx)
        return x


class Mixer(nn.Module):
    def __init__(self, token, channel, expansion_factor, depth=1, dropout=0.2):
        super(Mixer, self).__init__()
        self.depth = depth
        self.mixers = nn.ModuleList(
            [MLP_Communicator(token,
                   channel,
                   expansion_factor)
             for _ in range(self.depth)])

    def forward(self, x):
        for m in self.mixers:
            x = m(x)
        return x


class Swish(nn.Module):
    def __init__(
        self,
    ):
        """
        Init method.
        """
        super(Swish, self).__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * torch.sigmoid(input)


class CNNProjector(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout=0.5):
        
        super().__init__()

        self.proj_nn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(8),
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(16),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(32)
        )

        self.fn = nn.Sequential(
            nn.Linear(input_dim * 32, output_dim),
            nn.GELU(),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.proj_nn( x )
        x = Rearrange('b n d -> b (n d)')(x)
        x = self.fn(x)
        return x


class LinearProjector(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super().__init__()

        self.proj_nn = nn.Sequential(
            nn.BatchNorm1d(input_dim),

            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.proj_nn(x)


class Predictor(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        # input_size = input_dim * num_feature

        self.predictor =  nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 1024),
            # nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.BatchNorm1d(1024),
            nn.Linear(1024,1024),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.BatchNorm1d(1024),
            nn.Linear(1024,512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 1),
        )

        # self._weight_init()

    def forward(self, feature):
        return self.predictor(feature)

    # def _weight_init(self):
    #     for m in self.predictor.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight)


class PermuteDDS(nn.Module):
    def __init__(self, gene_dim, mutation_dim, d_model, dropout=0.5):
        
        super().__init__()

        fp_dim = 512
        self.proj_hashtt = CNNProjector(fp_dim, d_model)
        self.proj_map4 = CNNProjector(fp_dim, d_model)
        self.proj_maccs = CNNProjector(167, d_model)
        # self.proj_hashtt  = LinearProjector(512, d_model)
        # self.proj_map4 = LinearProjector(512, d_model)
        # self.proj_maccs = LinearProjector(167, d_model)

        self.proj_gene = nn.Sequential(
            nn.BatchNorm1d(gene_dim),
            CNNProjector(gene_dim, d_model)
        )
        # self.proj_cnv = CNNProjector(cnv_dim, d_model)  

        self.proj_mutation = nn.Sequential(
            nn.BatchNorm1d(mutation_dim),
            CNNProjector(mutation_dim, d_model)
        )

        # fp1, fp2, gene, mutation
        n_channel = 4
        self.fusion_early_hashtt = Mixer(d_model, n_channel, d_model, depth=2)
        self.fusion_early_map4 = Mixer(d_model, n_channel, d_model, depth=2)
        self.fusion_early_maccs = Mixer(d_model, n_channel, d_model, depth=2)

        self.pred_hashtt = Predictor(d_model * n_channel, dropout)
        self.pred_map4 = Predictor(d_model * n_channel, dropout)
        self.pred_maccs = Predictor(d_model * n_channel, dropout)

    def forward(self, hashtt1, hashtt2, map1, map2, maccs1, maccs2, cline_gene, cline_mutation):

        drug1_hashtt_f = self.proj_hashtt(hashtt1)
        drug2_hashtt_f = self.proj_hashtt(hashtt2)

        drug1_map4_f = self.proj_map4(map1)
        drug2_map4_f = self.proj_map4(map2)

        drug1_maccs_f = self.proj_maccs(maccs1)
        drug2_maccs_f = self.proj_maccs(maccs2)
        
        gene_feature = self.proj_gene(cline_gene)
        mutation_feature = self.proj_mutation(cline_mutation)


        hashtt_fusion_f = torch.stack((drug1_hashtt_f, drug2_hashtt_f, mutation_feature, gene_feature), axis=1)
        hashtt_fusion_f = self.fusion_early_hashtt(hashtt_fusion_f)
        hashtt_fusion_f = Rearrange('b n d -> b (n d)')(hashtt_fusion_f)
        # fp_fusion_f = Reduce("b n d ->b d", reduction="max")(fp_fusion_f)

        map4_fusion_f = torch.stack((drug1_map4_f, drug2_map4_f, mutation_feature, gene_feature), axis=1)
        map4_fusion_f = self.fusion_early_map4(map4_fusion_f)
        map4_fusion_f = Rearrange('b n d -> b (n d)')(map4_fusion_f)
        # m_fusion_f = Reduce("b n d ->b d", reduction="max")(m_fusion_f)

        maccs_fusion_f = torch.stack((drug1_maccs_f, drug2_maccs_f, mutation_feature, gene_feature), axis=1)
        maccs_fusion_f = self.fusion_early_maccs(maccs_fusion_f)
        maccs_fusion_f = Rearrange('b n d -> b (n d)')(maccs_fusion_f)
        # ma_fusion_f = Reduce("b n d ->b d", reduction="max")(ma_fusion_f)

        # return pred 
        pred_hashtt = self.pred_hashtt(hashtt_fusion_f)
        pred_map4 = self.pred_map4(map4_fusion_f)
        pred_maccs = self.pred_maccs(maccs_fusion_f) 

        return pred_hashtt, pred_map4, pred_maccs 
