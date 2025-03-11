# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import json

class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        # token_id对应新的序号,例如: ['[PAD]','196','186']
        self.uid_seq = dataset.field2id_token['user_id']
        self.iid_seq = dataset.field2id_token['item_id']

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

        self.user_topn_dict = {}

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        # print(self.training)
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]

        # 假设你想要的top-n值
        k = 200  # 可根据需求变化

        # 对scores排序，取出top-n
        # scores的形状为[B, n_items]
        # 每行代表一个用户，对n_items物品的打分

        # 使用torch.topk可以快速获取top-n的值和索引
        # topk_values: [B, n]
        # topk_indices: [B, n] 表示top-n物品在item_embedding.weight中的行索引
        topk_values, topk_indices = torch.topk(scores, k=k, dim=1)

        # topk_indices中存放的就是每个用户对应的top-n物品的索引（在整个物品集合中的索引）
        # 如果你需要将这些索引映射到物品ID上，需要看看你的数据预处理里物品ID和embedding的索引关系
        # 通常，item_embedding的weight矩阵行索引就是物品的内部ID
        # 如果物品ID与内部ID是一致的，那么topk_indices已经是物品的内部ID（即item_id）

        # 假设item与item_id是一致映射，那么topk_indices就是top-n的item_id列表
        # 你可以将结果存储到一个列表或字典中
        # 例如为用户i存储top-n结果
        # 注意: B是batch维度，这里需要知道你的interaction中用户信息的获取方式
        # 通常在训练或评估过程中，每个batch中会有多个用户
        # 你可以从interaction中提取用户id，假设为 self.USER_ID (需要根据你的数据和代码逻辑)

        if self.USER_ID in interaction:
            user_ids = interaction[self.USER_ID]  # [B]
        else:
            # 如果没有USER_ID，需要根据具体数据获取用户id
            # 比如如果是测试集评估时，会有user_id对齐的列
            # 在此示例中先假设有user_ids
            user_ids = torch.arange(scores.size(0))  # 假设用户是0~B-1


        # 将top-n物品和用户对应存起来
        for i, user_id in enumerate(user_ids):
            # user_id.item()将张量转换为python标量，如果user_id是张量
            user_id_val = user_id.item() if isinstance(user_id, torch.Tensor) else user_id
            # topk_indices[i]是该用户的top-n物品ID列表
            topn_item_ids = topk_indices[i].cpu().tolist()  # 转为python列表
            topn_item_token_ids = [self.iid_seq[iid] for iid in topn_item_ids]
            self.user_topn_dict[self.uid_seq[user_id_val]] = topn_item_token_ids

        # 现在user_topn_dict中存放了 {user_id: [top-n物品列表]} 的信息
        # 你可以根据需要将user_topn_dict返回，或者在class中存储
        # 注意，这里是full_sort_predict方法内部的示例，如果需要对外输出可以返回该字典
        topk_path = f'E:/data/dataset/rs/topk/{self.config["dataset"]}_top-{k}.json'
        with open(topk_path, "w", encoding="utf-8") as file:
            json.dump(self.user_topn_dict, file, ensure_ascii=False)

        return scores

