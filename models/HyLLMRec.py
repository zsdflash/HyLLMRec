import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from BaseModel import BaseModel
from utils import build_interaction_matrix

class HGNNLayer(nn.Module):
    def __init__(self, n_hyper_layer=1):
        super(HGNNLayer, self).__init__()
        self.h_layer = n_hyper_layer

    def forward(self, i_hyper, u_hyper, embeds):
        i_ret = embeds
        for _ in range(self.h_layer):
            lat = torch.mm(i_hyper.T, i_ret)
            i_ret = torch.mm(i_hyper, lat)
            u_ret = torch.mm(u_hyper, lat)
        return u_ret, i_ret

class ExtraInfoHypergraphConstructor(nn.Module):
    def __init__(self, input_dims, num_hyperedges, tau=0.2, fusion='attention'):
        super().__init__()
        self.num_sources = len(input_dims)
        self.num_hyperedges = num_hyperedges
        self.tau = tau
        self.fusion = fusion
        self.proj_layers = nn.ModuleList([
            nn.Linear(dim, num_hyperedges) for dim in input_dims
        ])
        if fusion == 'attention':
            self.fusion_att = nn.Parameter(torch.Tensor(self.num_sources))
            nn.init.constant_(self.fusion_att, 1.0)
        elif fusion == 'concat':
            self.concat_proj = nn.Linear(num_hyperedges * self.num_sources, num_hyperedges)

    def forward(self, embeddings_list):
        H_list = []
        for i, emb in enumerate(embeddings_list):
            logits = self.proj_layers[i](emb)
            H_i = F.gumbel_softmax(logits, tau=self.tau, dim=1, hard=False)
            H_list.append(H_i)

        if self.fusion == 'attention':
            weights = F.softmax(self.fusion_att, dim=0)
            H = sum(w * h for w, h in zip(weights, H_list))
        elif self.fusion == 'concat':
            H = torch.cat(H_list, dim=1)
            H = self.concat_proj(H)
        return H

class CrossAttnBlockMHA(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1_q = nn.LayerNorm(dim)
        self.norm1_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x_q, x_kv, attn_mask=None, return_attn=False):
        q = self.norm1_q(x_q)
        k = self.norm1_kv(x_kv)
        v = self.norm1_kv(x_kv)
        y, attn = self.attn(q, k, v, attn_mask=None, need_weights=True, average_attn_weights=False)
        x = x_q + y
        x = x + self.mlp(self.norm2(x))
        return (x, attn) if return_attn else x


class VisTextCrossFusion(nn.Module):
    def __init__(self, d_vis=2048, d_txt=728, d=512, num_heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.proj_v = nn.Sequential(nn.Linear(d_vis, d), nn.GELU(), nn.LayerNorm(d))
        self.proj_t = nn.Sequential(nn.Linear(d_txt, d), nn.GELU(), nn.LayerNorm(d))

        self.xattn_v_from_t = CrossAttnBlockMHA(d, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop)
        self.xattn_t_from_v = CrossAttnBlockMHA(d, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop)

        self.gate = nn.Sequential(
            nn.Linear(4*d, d),
            nn.GELU(),
            nn.Linear(d, d)
        )
        self.bilinear = nn.Linear(d, d)
        self.out_ln = nn.LayerNorm(d)

    def forward(self, v, T, return_intermediate=False):

        B = v.size(0)
        v = self.proj_v(v).unsqueeze(1)      
        T = self.proj_t(T)                  

        v_ctx = self.xattn_v_from_t(v, T)    

        T_ctx = self.xattn_t_from_v(T, v)    

        v_out = v_ctx.squeeze(1)            
        t_out = T_ctx.mean(dim=1)            

        h = torch.cat([v_out, t_out, v_out * t_out, (v_out - t_out).abs()], dim=-1)
        g = torch.sigmoid(self.gate(h))
        z = g * v_out + (1 - g) * t_out + self.bilinear(v_out * t_out)
        z = self.out_ln(z)
        if return_intermediate:
            return z, v_out, t_out
        return z


class HyLLMRec(BaseModel):
    def __init__(self, config):
        super(HyLLMRec, self).__init__(config)

        self.cl_weight = 1e-4
        self.reg_weight = 1e-6
        self.tau = 0.2
        self.num_hyperedges_t = 45
        self.num_hyperedges_v = 30

        self.interaction_matrix = build_interaction_matrix(self.train_edge, self.num_user, self.num_item)
        inter_mat = self.interaction_matrix.tocoo() 
        indices_np = np.array([inter_mat.row, inter_mat.col])
        indices = torch.tensor(indices_np, dtype=torch.long)
        values = torch.tensor(inter_mat.data, dtype=torch.float32)
        shape = inter_mat.shape
        self.adj = torch.sparse_coo_tensor(indices, values, shape).to(self.device)

        self.n_nodes = self.num_user + self.num_item
        self.norm_adj = self.get_norm_adj_mat()

        self.user_embedding = nn.Embedding(self.num_user, self.dim_E)
        self.item_embedding = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.hgnn = HGNNLayer(n_hyper_layer=1)

        self.t_proj = nn.Linear(768, self.dim_E)
        self.v_proj = nn.Linear(2048, self.dim_E)

        l_feat1, l_feat2, l_feat3, l_feat4, l_feat5 = torch.unbind(self.local_feat, dim=2)
        self.local_feat_list = [l_feat1, l_feat2, l_feat3, l_feat4, l_feat5]
        self.dim_list = [768, 768, 768, 768, 768]

        self.t_constructor = ExtraInfoHypergraphConstructor(
            input_dims=self.dim_list,
            num_hyperedges = self.num_hyperedges_t,
            tau=self.tau,
            fusion='concat')

        self.vtf = VisTextCrossFusion(d_vis=2048, d_txt=768, d=2048, num_heads=8, drop=0.1)
        self.v_constructor = ExtraInfoHypergraphConstructor(
            input_dims=[2048],
            num_hyperedges=self.num_hyperedges_v,
            tau=self.tau,
            fusion='concat')

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.num_user), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.num_user, inter_M_t.col), [1] * inter_M_t.nnz)))
        for (i, j), v in data_dict.items():
            A[i, j] = v
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D @ A @ D
        L = sp.coo_matrix(L)
        row = torch.LongTensor(L.row)
        col = torch.LongTensor(L.col)
        i = torch.stack([row, col])
        data = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(i, data, torch.Size(L.shape)).to(self.device)

    def cge(self):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        out_emb = ego_embeddings
        for _ in range(2):
            out_emb = torch.sparse.mm(self.norm_adj, out_emb)
        return out_emb

    def mge(self, feat, proj_layer):
        item_emb = proj_layer(feat)
        user_emb = torch.sparse.mm(self.adj, item_emb)
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        for _ in range(2):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
        return all_emb

    def forward(self):

        u_embed = self.user_embedding.weight
        i_embed = self.item_embedding.weight


        H_item_t = self.t_constructor(self.local_feat_list)
        H_user_t = torch.sparse.mm(self.adj, H_item_t)
        u_hyper_t, i_hyper_t = self.hgnn(H_item_t, H_user_t, i_embed)

        key_T = torch.stack(self.key_list, dim=1)
        v_t_fused = self.vtf(self.v_feat, key_T)
        
        H_item_v = self.v_constructor([v_t_fused])
        H_user_v = torch.sparse.mm(self.adj, H_item_v)
        u_hyper_v, i_hyper_v = self.hgnn(H_item_v, H_user_v, i_embed)

        u_hyper = u_hyper_t + u_hyper_v
        i_hyper = i_hyper_t + i_hyper_v
        
        all_cge = self.cge()
        u_cge, i_cge = torch.split(all_cge, [self.num_user, self.num_item], dim=0)

        all_mge_text = self.mge(self.t_feat, self.t_proj)
        all_mge_vis = self.mge(self.v_feat, self.v_proj)
        u_mge, i_mge = torch.split(F.normalize(all_mge_text + all_mge_vis, dim=1), [self.num_user, self.num_item], dim=0)

        u_final = F.normalize(u_hyper + u_cge + u_mge, dim=1)
        i_final = F.normalize(i_hyper + i_cge + i_mge, dim=1)

        all_emb = torch.cat([u_final, i_final], dim=0)
        self.result = all_emb

        return all_emb, u_final, i_final, u_hyper, u_cge, u_mge, i_hyper, i_cge, i_mge

    def loss(self, user_tensor, item_tensor, out):
        all_emb, u_final, i_final, u_hyper, u_cge, u_mge, i_hyper, i_cge, i_mge = out

        users = user_tensor[:, 0]
        pos_items = item_tensor[:, 0]
        neg_items = item_tensor[:, 1]

        u_emb = all_emb[users]
        pos_emb = all_emb[pos_items]
        neg_emb = all_emb[neg_items]

        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        ssl_main = self.ssl_loss(
            u_final[users],
            i_final[pos_items - self.num_user],
            i_final
        )
        ssl_cross = self.ssl_loss(
            i_final[pos_items - self.num_user],
            i_final[neg_items - self.num_user],
            i_final
        )

        ssl_user = (
            self.InfoNCE(u_hyper[users], u_cge[users], self.tau)
            + self.InfoNCE(u_cge[users], u_mge[users], self.tau)
            + self.InfoNCE(u_hyper[users], u_mge[users], self.tau)
        ) / 3

        pos_idx = pos_items - self.num_user
        ssl_item = (
            self.InfoNCE(i_hyper[pos_idx], i_cge[pos_idx], self.tau)
            + self.InfoNCE(i_cge[pos_idx], i_mge[pos_idx], self.tau)
            + self.InfoNCE(i_hyper[pos_idx], i_mge[pos_idx], self.tau)
        ) / 3

        ssl_multi = (ssl_user + ssl_item) / 2

        reg_loss = (
            u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)
        ) / u_emb.shape[0]

        total_ssl = ssl_main + ssl_cross + 0.5 * ssl_multi

        return bpr_loss + self.cl_weight * total_ssl + self.reg_weight * reg_loss


    def ssl_loss(self, emb1, emb2, all_emb):
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        all_emb = F.normalize(all_emb, dim=1)
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.tau)
        ttl_score = torch.sum(torch.exp(torch.matmul(emb1, all_emb.T) / self.tau), dim=1)
        ssl = -torch.log(pos_score / ttl_score).sum()
        return ssl
    
    def InfoNCE(self, view1, view2, temperature):
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        pos_score = torch.exp(torch.sum(view1 * view2, dim=1) / temperature)
        ttl_score = torch.exp(torch.matmul(view1, view2.T) / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)