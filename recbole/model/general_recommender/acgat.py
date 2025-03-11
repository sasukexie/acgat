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


# Adaptive Contrastive Graph Attention Network for Recommendation (ACGAT)
class ACGAT(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(ACGAT, self).__init__(config, dataset)

        # 获取设备类型（cuda 或 cpu）
        self.device = config["device"]

        # GCN 部分
        self.n_layers = config["n_layers"]
        self.latent_dim = config["embedding_size"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.dropout1 = config["dropout1"]
        self.n_heads = config["n_heads"]
        self.negative_slope = config["negative_slope"]
        self.cl_weight = config["cl_weight"]
        self.temperature = config["temperature"]
        self.ed_rate = config["ed_rate"]

        # 汇聚函数（全局图嵌入）
        # self.readout = lambda x: x.mean(dim=0) # 无激活函数
        # self.readout = lambda x: torch.sigmoid(x.mean(dim=0))
        self.readout = lambda x: torch.tanh(x.mean(dim=0))
        # 分数mlp，用于计算节点嵌入与图嵌入之间的互信息
        self.score_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * 4, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1)
        ).to(self.device)
        self.item_score_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * 3, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1)
        ).to(self.device)

        # 嵌入层
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim).to(self.device)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim).to(self.device)

        # GAT
        # 定义 GraphConv 层，针对每种边类型使用独立的 GATConv
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
                    negative_slope=self.negative_slope,
                    residual=True,
                    activation=F.elu if layer < self.n_layers - 1 else None  # 最后一层不使用激活
                ),
                'co-rated': GATConv(
                    in_feats=in_feats,
                    out_feats=self.latent_dim,
                    num_heads=self.n_heads,
                    feat_drop=self.dropout,
                    attn_drop=self.dropout,
                    negative_slope=self.negative_slope,
                    residual=True,
                    activation=F.elu if layer < self.n_layers - 1 else None
                )
            }, aggregate='mean'))

        # 融合多头输出的 MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * self.n_heads, self.latent_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout1)
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

    def generate_views(self, graph, num_views=2):
        """生成多个扰动后的图视图，结合不同的扰动策略"""
        views = []
        for _ in range(num_views):
            corrupted_graph = graph
            # 扰动策略1：随机删除一定比例的交互边
            num_edges = graph.num_edges(etype='interacts')
            num_remove = int(num_edges * self.ed_rate)  # 删除10%的交互边
            if num_remove > 0:
                remove_ids = torch.randperm(num_edges, device=self.device)[:num_remove]  # 确保remove_ids在正确设备上
                corrupted_graph = dgl.remove_edges(corrupted_graph, remove_ids, etype='interacts')

            # 扰动策略2：随机删除一定比例的反向边
            num_edges_rev = corrupted_graph.num_edges(etype='co-rated')
            num_remove_rev = int(num_edges_rev * self.ed_rate)  # 删除10%的反向边
            if num_remove_rev > 0:
                remove_ids_rev = torch.randperm(num_edges_rev, device=self.device)[:num_remove_rev]  # 确保remove_ids_rev在正确设备上
                corrupted_graph = dgl.remove_edges(corrupted_graph, remove_ids_rev, etype='co-rated')

            views.append(corrupted_graph)
        return views

    def compute_node_importance(self, etype):
        """计算节点的重要性（基于度数）"""
        degrees = self.graph.in_degrees(etype=etype).float()
        importance = degrees / degrees.sum()
        return importance

    def forward_graph(self, graph, corrupted=False):
        """对原始图或扰动后的图进行编码，生成节点嵌入和中间层嵌入"""
        if corrupted:
            # 动态负样本生成：扰动节点嵌入，例如通过随机遮盖部分特征
            corrupted_user_emb = self.user_embedding.weight[torch.randperm(self.n_users, device=self.device)]
            corrupted_item_emb = self.item_embedding.weight[torch.randperm(self.n_items, device=self.device)]
            embeddings_dict = {'user': corrupted_user_emb, 'item': corrupted_item_emb}
        else:
            # 正常的节点嵌入
            all_user_embedding = self.user_embedding.weight
            all_item_embedding = self.item_embedding.weight
            embeddings_dict = {'user': all_user_embedding, 'item': all_item_embedding}

        # 收集每层的嵌入
        all_layers_embeddings = {'user': [], 'item': []}

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
        gat_user_emb = torch.mean(torch.stack(all_layers_embeddings['user'], dim=1), dim=1)  # [n_users, n_heads * out_feats]
        gat_item_emb = torch.mean(torch.stack(all_layers_embeddings['item'], dim=1), dim=1)  # [n_items, n_heads * out_feats]

        # 融合多头输出的 MLP
        gat_user_emb = self.fusion_mlp(gat_user_emb)  # [n_users, latent_dim]
        gat_item_emb = self.fusion_mlp(gat_item_emb)  # [n_items, latent_dim]

        return gat_user_emb, gat_item_emb, all_layers_embeddings  # 返回中间层嵌入

    def forward(self):
        """GCN (GAT) 特征提取并生成ACGI相关嵌入"""
        # 生成多个视图
        views = self.generate_views(self.graph.clone(), num_views=2)  # 生成两个扰动视图

        # 正样本嵌入
        user_emb, item_emb, all_layers_embeddings = self.forward_graph(self.graph, corrupted=False)

        # 负样本嵌入
        corrupted_embeddings = []
        for corrupted_graph in views:
            corrupted_user_emb, corrupted_item_emb = self.forward_graph(corrupted_graph, corrupted=True)[:2]
            corrupted_embeddings.append((corrupted_user_emb, corrupted_item_emb))

        # 生成全局图嵌入
        combined_emb = torch.cat([user_emb, item_emb], dim=0)  # [n_users + n_items, latent_dim * 2]
        graph_emb = self.readout(combined_emb)  # [latent_dim]

        # 生成所有负样本图嵌入
        shuffled_graph_embs = []
        for corrupted_user_emb, corrupted_item_emb in corrupted_embeddings:
            corrupted_combined_emb = torch.cat([corrupted_user_emb, corrupted_item_emb], dim=0)  # [n_users + n_items, latent_dim * 2]
            shuffled_graph_emb = self.readout(corrupted_combined_emb)  # [latent_dim]
            shuffled_graph_embs.append(shuffled_graph_emb)

        return user_emb, item_emb, graph_emb, shuffled_graph_embs, all_layers_embeddings


    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # 前向传播，获取ACGI相关嵌入
        user_embeddings, item_embeddings, graph_emb, shuffled_graph_embs, all_layers_embeddings = self.forward()  # GAT 嵌入

        # 推荐损失（BPRLoss）
        u_emb = user_embeddings[user]
        pos_emb = item_embeddings[pos_item]
        neg_emb = item_embeddings[neg_item]

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        loss = 0.0
        # bpr损失
        loss += self.mf_loss(pos_scores, neg_scores)

        # ACGI损失
        loss += self.cl_weight * self.acgi_loss(shuffled_graph_embs, all_layers_embeddings) if self.cl_weight > 0.01 else 0.0

        # 正则化（可选）
        # loss += self.reg_weight * (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2))

        return loss

    def acgi_loss(self, shuffled_graph_embs, all_layers_embeddings):
        """
        计算ACGI的对比损失，结合多视图和多层次对比。
        """
        loss = 0.0
        # 计算节点重要性
        importance_user = self.compute_node_importance('co-rated')
        importance_item = self.compute_node_importance('interacts')
        for layer_idx in range(self.n_layers):
            # 获取当前层的嵌入
            user_emb_layer = all_layers_embeddings['user'][layer_idx]
            item_emb_layer = all_layers_embeddings['item'][layer_idx]

            # 生成全局图嵌入
            combined_emb = torch.cat([user_emb_layer, item_emb_layer], dim=0)  # [n_users + n_items, latent_dim * 2]
            graph_emb_layer = self.readout(combined_emb)  # [latent_dim]

            # 正样本对
            pos_combined = torch.cat([combined_emb, graph_emb_layer.unsqueeze(0).repeat(combined_emb.size(0), 1)], dim=1)  # [n_users + n_items, latent_dim * 3]
            pos_labels = torch.ones(pos_combined.size(0), device=self.device)  # [n_users +n_items]

            # 计算正样本的分数
            pos_scores = self.score_mlp(pos_combined).squeeze()  # [n_users +n_items]

            # 计算正样本的损失
            # 计算自适应权重
            weights = torch.cat([importance_user, importance_item], dim=0)  # [n_users + n_items]
            # weights = torch.cat([weights, weights], dim=0)  # [2 * (n_users + n_items)]
            pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight=weights, reduction='mean')

            # 初始化负样本损失
            neg_loss = 0.0

            for shuffled_graph_emb in shuffled_graph_embs:
                # 负样本对
                neg_combined = torch.cat([combined_emb, shuffled_graph_emb.unsqueeze(0).repeat(combined_emb.size(0), 1)], dim=1)  # [n_users +n_items, latent_dim * 3]
                neg_labels = torch.zeros(neg_combined.size(0), device=self.device)  # [n_users +n_items]

                # 计算负样本的分数
                neg_scores = self.item_score_mlp(neg_combined).squeeze()  # [n_users +n_items]
                # 对正样本和负样本的 scores 进行温度缩放，通过除以 temperature，来调节正负样本之间的相似度差距。
                neg_scores = neg_scores / self.temperature
                # 计算负样本的损失
                neg_loss += F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight=weights, reduction='mean')

            # 平均负样本损失
            neg_loss /= len(shuffled_graph_embs)

            # 加权层次损失
            layer_weight = 1.0 / (layer_idx + 1)  # 示例权重，浅层权重更高
            loss += layer_weight * (pos_loss + neg_loss)

        # 综合所有层的损失
        loss /= self.n_layers
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()[:2]  # 只需要用户和物品嵌入

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
