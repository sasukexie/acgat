import common.tool as tool
import run_main

RUNNING_FLAG = None

def main(model_name, dataset_name, parameter_dict):
    if ['GRU4Rec', 'SASRec', 'DuoRec'].__contains__(model_name):
        if ['yelp', 'ml-1m'].__contains__(dataset_name):
            parameter_dict['attn_dropout_prob'] = 0.1
            parameter_dict['hidden_dropout_prob'] = 0.1
    run_main.main(model_name, dataset_name, parameter_dict)

def process_base(model_name_arr, dataset_name_arr, parameter_dict):
    # param
    # config_file = None  # None/base
    parameter_dicts = {
        'BPR': {
            'train_batch_size': 4096,
        },
        'LightGCN': {
            'train_batch_size': 4096,
        },
        'SGL': {
            'train_batch_size': 4096,
        },
        'NewRec': {
            'train_batch_size': 4096,
        },
        'Caser': {
            'train_batch_size': 1024,
            'eval_batch_size': 256,
        },
        'GRU4Rec': {
            'train_batch_size': 1024,
            'eval_batch_size': 256,
        },
        'SRGNN': {
            'train_batch_size': 1024,
            'eval_batch_size': 256,
        },
        'GCSAN': {
            'train_batch_size': 1024,
            'eval_batch_size': 256,
        },
        'SASRec': {
            'train_batch_size': 1024,
            'eval_batch_size': 256,
            'train_neg_sample_args': None,
        },
        'BERT4Rec': {
            'train_batch_size': 512,
            'eval_batch_size': 2048,
        },
        'DuoRec': {
            'train_batch_size': 1024,
            'eval_batch_size': 256,
        },
        'CL4SRec': {
            'train_batch_size': 1024,
            'eval_batch_size': 256,
            'tau': 1.0,
            'sim': 'dot',
            'lmd': 0.01,
        },
        'DCRec': {
            "train_batch_size": 1024,
            "eval_batch_size": 256,

            "hidden_dropout_prob": 0.3,
            "attn_dropout_prob": 0.3,
            # Graph Args:
            "graph_dropout_prob": 0.3,
            "graphcl_enable": 1,
            "graphcl_coefficient": 1e-4,
            "cl_ablation": 'full',
            "graph_view_fusion": 1,
            "cl_temp": 1,
            # "save_dataloaders": False,
        },
        'KGAT': {
            'train_batch_size': 2048,
            'eval_batch_size': 4096,
            'load_col': {
                'inter': ['user_id', 'item_id', 'rating'],
                'kg': ['head_id', 'relation_id', 'tail_id'],
                'link': ['item_id', 'entity_id']
            }
        },
        'Other':{
            'train_batch_size': 4096,
        }
    }
    for model_name in model_name_arr:
        for dataset_name in dataset_name_arr:
            if parameter_dicts.__contains__(model_name):
                config = parameter_dicts[model_name]
            else:
                config = parameter_dicts['Other']
            tool.tranfer_dict(parameter_dict, config)
            custom_parameter(model_name, dataset_name, parameter_dict)
            main(model_name, dataset_name, parameter_dict)


def custom_parameter(model_name, dataset_name, parameter_dict):
    if ['DCRec'].__contains__(model_name):
        # BEST SETTINGS
        if dataset_name == "reddit":
            parameter_dict["train_batch_size"] = 128
            parameter_dict["graphcl_coefficient"] = 1
            parameter_dict["weight_mean"] = 0.5
            parameter_dict["sim_group"] = 4
            parameter_dict["kl_weight"] = 1
        else:
            parameter_dict["graphcl_coefficient"] = 1e-1
            parameter_dict["graph_dropout_prob"] = 0.5
            parameter_dict["hidden_dropout_prob"] = 0.5
            parameter_dict["attn_dropout_prob"] = 0.5
            parameter_dict["kl_weight"] = 1e-2

            if ["beauty", "ml-1m1", "steam", "Amazon_Books", "lfm1b-tracks", "Amazon_Electronics"].__contains__(dataset_name):
                parameter_dict["schedule_step"] = 30
                parameter_dict["attn_dropout_prob"] = 0.1
                parameter_dict["sim_group"] = 4
                parameter_dict["weight_mean"] = 0.5
                parameter_dict["cl_temp"] = 1
            elif ["sports"].__contains__(dataset_name):
                parameter_dict["attn_dropout_prob"] = 0.3
                parameter_dict["sim_group"] = 4
                parameter_dict["weight_mean"] = 0.5
                parameter_dict["cl_temp"] = 1
            elif ["ml-20m", 'yelp', 'yelp1'].__contains__(dataset_name):
                parameter_dict["sim_group"] = 4
                parameter_dict["weight_mean"] = 0.4
                parameter_dict["cl_temp"] = 0.8
    else:
        if ['yelp'].__contains__(dataset_name):
            parameter_dict["eval_batch_size"] = 128


if __name__ == '__main__':
    parameter_dict = {
        'epochs': 100,
        'train_batch_size': 4096,
        'eval_batch_size': 40960,
        'gpu_id': '0',  # (str) The id of GPU device(s).
        # 'MAX_ITEM_LIST_LENGTH': 50,
        # 'neg_sampling': None,
        # 'train_neg_sample_args': None,
    }

    ############## other #####################
    # model_name_arr = ['BPR']

    ############## base(源码运行) #####################
    ## CoSeRec,ICLRec

    ############## base #####################0
    # GNN:LightGCN, KG:KGAT, SR:SASRec, CL:SGL
    # model_name_arr = ['KGAT','LightGCN','Caser','GRU4Rec','SRGNN','GCSAN','SASRec','BERT4Rec','DuoRec','CL4SRec','DCRec','SGL']

    model_name_arr = ['LightGCN','SGL','NCL']

    # dataset_name_arr = ['movielens','RentTheRunway','lfm1b-tracks','netflix']
    dataset_name_arr = ['movielens','lfm1b-tracks','netflix','RentTheRunway']

    process_base(model_name_arr, dataset_name_arr, parameter_dict)
