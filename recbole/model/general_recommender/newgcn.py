import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, HeteroGraphConv

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

class NewGCN(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NewGCN, self).__init__(config, dataset)

        # 获取设备类型（cuda 或 cpu）
        self.device = config["device"]

        # GCN 部分
        self.n_layers = config["n_layers"]
        self.latent_dim = config["embedding_size"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.n_heads = config["n_heads"]


        # DGI组件
        # 汇聚函数（全局图嵌入）
        self.readout = lambda x: torch.sigmoid(x.mean(dim=0))
        # 鉴别器，用于计算节点嵌入与图嵌入之间的互信息
        self.discriminator = nn.Bilinear(self.latent_dim, self.latent_dim, 1).to(self.device)
        # DGI损失权重
        self.dgi_loss_weight = 0.1

        # 嵌入层
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim).to(self.device)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim).to(self.device)

        # 定义 HeteroGraphConv 层，针对每种边类型使用独立的 GATConv
        self.gat_layers = nn.ModuleList()
        for layer in range(self.n_layers):
            # 第一层的 in_feats 是 latent_dim，后续层的 in_feats 是 latent_dim * n_heads
            in_feats = self.latent_dim if layer == 0 else self.latent_dim * self.n_heads
            self.gat_layers.append(HeteroGraphConv({
                'interacts': GATConv(
                    in_feats=in_feats,
                    out_feats=self.latent_dim,
                    num_heads=self.n_heads,
                    feat_drop=self.dropout,
                    attn_drop=self.dropout,
                    negative_slope=0.2,
                    residual=True,
                    activation=F.elu if layer < self.n_layers - 1 else None  # 最后一层不使用激活
                ),
                'co-rated': GATConv(
                    in_feats=in_feats,
                    out_feats=self.latent_dim,
                    num_heads=self.n_heads,
                    feat_drop=self.dropout,
                    attn_drop=self.dropout,
                    negative_slope=0.2,
                    residual=True,
                    activation=F.elu if layer < self.n_layers - 1 else None
                )
            }, aggregate='mean'))

        # 融合多头输出的 MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * self.n_heads, self.latent_dim),  # 修改这里
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # 构建邻接矩阵
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # Loss
        self.mf_loss = BPRLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # 参数初始化
        self.apply(xavier_uniform_initialization)

        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

        # 通过 DGL 构建图
        self.graph = self.build_graph()


    def get_norm_adj_mat(self):
        """创建归一化的邻接矩阵，使用自适应归一化和稀疏矩阵加速计算"""
        # 构建用户-物品交互矩阵（稀疏矩阵）
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()

        # 更新交互矩阵，保留不同的边类型
        data_dict = dict(zip(zip(inter_M.row, inter_M.col), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        # 计算度数（使用对角矩阵D）
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)  # 计算逆平方根
        D = sp.diags(diag)

        # 使用自适应归一化
        L = D * A * D

        # 将L矩阵转换为COO格式（压缩格式）
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col])).to(self.device)
        data = torch.FloatTensor(L.data).to(self.device)

        # 将稀疏矩阵转换为PyTorch稀疏矩阵
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape), dtype=torch.float32)

        # 返回归一化后的稀疏矩阵
        return SparseL

    def build_graph(self):
        """使用 DGL 构建异构图并初始化边数据"""
        # 确保item IDs不超过n_items-1
        max_item_id = self.interaction_matrix.col.max()
        print(f"Max item ID in interactions: {max_item_id}")
        assert max_item_id < self.n_items, f"Max item ID {max_item_id} exceeds n_items {self.n_items}"

        # 定义边，不进行偏移
        edges = (self.interaction_matrix.row, self.interaction_matrix.col)  # 用户-物品交互边
        reverse_edges = (self.interaction_matrix.col, self.interaction_matrix.row)  # 物品-用户交互边

        # 创建异构图，保留不同的边类型
        g = dgl.heterograph({
            ('user', 'interacts', 'item'): edges,
            ('item', 'co-rated', 'user'): reverse_edges  # 保留不同边类型
        }, num_nodes_dict={
            'user': self.n_users,
            'item': self.n_items
        }).to(self.device)

        return g

    def forward(self, graph=None):
        """ GCN (GAT) 特征提取并生成DGI相关嵌入"""
        if graph is None:
            graph = self.graph

        all_user_embedding = self.user_embedding.weight
        all_item_embedding = self.item_embedding.weight
        # 初始化节点嵌入
        embeddings_dict = {
            'user': all_user_embedding,  # [n_users, latent_dim]
            'item': all_item_embedding   # [n_items, latent_dim]
        }

        # 收集每层的嵌入
        all_layers_embeddings = {
            'user': [],
            'item': []
        }

        for layer_idx, gat_layer in enumerate(self.gat_layers):
            # HeteroGraphConv 处理异构图，返回一个字典，键为节点类型，值为对应节点类型的嵌入
            new_embeddings = gat_layer(graph, embeddings_dict)

            # 对每个节点类型的嵌入进行处理
            for ntype in ['user', 'item']:
                # Flatten 多头输出
                h = new_embeddings[ntype].flatten(1)  # [num_nodes, n_heads * out_feats]
                all_layers_embeddings[ntype].append(h)

            # 更新 embeddings_dict 为下一层的输入
            embeddings_dict = {}
            for ntype in new_embeddings:
                embeddings_dict[ntype] = new_embeddings[ntype].flatten(1)  # [num_nodes, n_heads * out_feats]

        # 叠加所有层的嵌入（类似 LightGCN）
        final_user_emb = torch.mean(torch.stack(all_layers_embeddings['user'], dim=1), dim=1)  # [n_users, n_heads * out_feats]
        final_item_emb = torch.mean(torch.stack(all_layers_embeddings['item'], dim=1), dim=1)  # [n_items, n_heads * out_feats]

        # 融合多头输出的 MLP
        final_user_emb = self.fusion_mlp(final_user_emb)  # [n_users, latent_dim]
        final_item_emb = self.fusion_mlp(final_item_emb)  # [n_items, latent_dim]

        # 生成全局图嵌入
        graph_emb = self.readout(torch.cat([all_user_embedding, all_item_embedding], dim=0))  # [latent_dim]

        # 生成负样本图嵌入（打乱用户和物品嵌入顺序）
        shuffled_user_emb = all_user_embedding[torch.randperm(all_user_embedding.size(0))]
        shuffled_item_emb = all_item_embedding[torch.randperm(all_item_embedding.size(0))]
        shuffled_graph_emb = self.readout(torch.cat([shuffled_user_emb, shuffled_item_emb], dim=0))  # [latent_dim]

        return final_user_emb, final_item_emb, graph_emb, shuffled_graph_emb

    def dgi_loss(self, user_emb, item_emb, graph_emb, shuffled_graph_emb):
        """
        计算DGI的对比损失，分别处理用户和物品嵌入。
        """
        # DGI for users
        batch_size_user = user_emb.size(0)
        graph_emb_user = graph_emb.unsqueeze(0).repeat(batch_size_user, 1)  # [batch_size_user, latent_dim]
        shuffled_graph_emb_user = shuffled_graph_emb.unsqueeze(0).repeat(batch_size_user, 1)  # [batch_size_user, latent_dim]

        pos_user_score = self.discriminator(user_emb, graph_emb_user).squeeze()  # [batch_size_user]
        neg_user_score = self.discriminator(user_emb, shuffled_graph_emb_user).squeeze()  # [batch_size_user]

        # 标签和分数
        labels_user = torch.cat([torch.ones_like(pos_user_score), torch.zeros_like(neg_user_score)]).to(self.device)  # [2 * batch_size_user]
        scores_user = torch.cat([pos_user_score, neg_user_score])  # [2 * batch_size_user]

        # 用户的二元交叉熵损失
        loss_user = F.binary_cross_entropy_with_logits(scores_user, labels_user)

        # DGI for items
        batch_size_item = item_emb.size(0)
        graph_emb_item = graph_emb.unsqueeze(0).repeat(batch_size_item, 1)  # [batch_size_item, latent_dim]
        shuffled_graph_emb_item = shuffled_graph_emb.unsqueeze(0).repeat(batch_size_item, 1)  # [batch_size_item, latent_dim]

        pos_item_score = self.discriminator(item_emb, graph_emb_item).squeeze()  # [batch_size_item]
        neg_item_score = self.discriminator(item_emb, shuffled_graph_emb_item).squeeze()  # [batch_size_item]

        # 标签和分数
        labels_item = torch.cat([torch.ones_like(pos_item_score), torch.zeros_like(neg_item_score)]).to(self.device)  # [2 * batch_size_item]
        scores_item = torch.cat([pos_item_score, neg_item_score])  # [2 * batch_size_item]

        # 物品的二元交叉熵损失
        loss_item = F.binary_cross_entropy_with_logits(scores_item, labels_item)

        # 合并用户和物品的损失
        loss = loss_user + loss_item

        return loss

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # 前向传播，获取DGI相关嵌入
        user_embeddings, item_embeddings, graph_emb, shuffled_graph_emb = self.forward()  # GAT 嵌入

        # 推荐损失（BPRLoss）
        u_emb = user_embeddings[user]
        pos_emb = item_embeddings[pos_item]
        neg_emb = item_embeddings[neg_item]

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        bpr_loss = self.mf_loss(pos_scores, neg_scores)

        # DGI损失
        dgi_loss = self.dgi_loss(user_embeddings, item_embeddings, graph_emb, shuffled_graph_emb)

        # 综合损失
        loss = bpr_loss + self.dgi_loss_weight * dgi_loss

        # 正则化（可选）
        # loss += self.reg_weight * (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2))

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, _, _ = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()[:2]  # 只需要用户和物品嵌入
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
