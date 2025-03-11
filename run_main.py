import argparse

from recbole.quick_start import run
from datetime import datetime
import common.tool as tool
import platform

RUNNING_FLAG = None


def main(model_name, dataset_name, parameter_dict, config_file=None):
    # 1.set param
    parser = argparse.ArgumentParser()
    # set model
    parser.add_argument('--model', '-m', type=str, default=model_name, help='name of models')
    # set datasets # ml-1m,ml-20m,amazon-books,lfm1b-tracks
    parser.add_argument('--dataset', '-d', type=str, default=dataset_name, help='name of datasets')
    # set config
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    # get param
    args, _ = parser.parse_known_args()
    # config list
    config_file_list = ['zone/common.yaml']

    if config_file:
        config_file_list.append(f'zone/{config_file}.yaml')

    global RUNNING_FLAG
    RUNNING_FLAG = f'RF{datetime.now().strftime("%Y%m%d%H%M%S")}' if RUNNING_FLAG == None else RUNNING_FLAG
    parameter_dict['running_flag'] = RUNNING_FLAG
    system_name = platform.system()
    if system_name == 'Windows':
        parameter_dict['gpu_id'] = '0'
    elif system_name == 'Linux':
        pass

    # 设置多任务
    nproc = 1
    world_size = -1
    # nproc = torch.cuda.device_count()
    # gpu_id = ''
    # if nproc>1:
    #     world_size = nproc*2
    #     for i in range(nproc):
    #         gpu_id += f'{i},'
    # parameter_dict['gpu_id'] = gpu_id

    # 2.call recbole_trm: config,dataset,model,trainer,training,evaluation
    run(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=parameter_dict, nproc=nproc, world_size=world_size)


def process_0(parameter_dict):
    # param
    # set model # MODEL,SimDCL,SASRec,BERT4Rec,BPR,GRU4RecF
    # set datasets # ml-1m,ml-20m,Amazon_Books,Amazon_Sports_and_Outdoors,Amazon_All_Beauty,amazon-books,lfm1b-tracks
    model_name_arr = ['ACGAT']  # GRU4RecF,BPR
    dataset_name_arr = ['ml-100k']  # ] #
    for model_name in model_name_arr:
        for dataset_name in dataset_name_arr:
            main(model_name, dataset_name, parameter_dict)


def process_1(parameter_dict, dataset_name_arr):
    # param
    # set model
    model_name = parameter_dict['model_name']
    parameter_dict1 = {
        'embedding_size': 64,
        'n_heads': 2, # 2
        'n_layers': 2, # 2
        'dropout': 0.1, # *,[0.1-0.9], ml-100k 0.5, other 0.5
        'dropout1': 0.1, # *,[0.1-0.9], ml-100k 0.5, other 0.5
        'negative_slope': 0.2, # 0.2
        'temperature': 0.5,
        'cl_weight': 1.0,
        'ed_rate': 0.1,
    }

    tool.tranfer_dict(parameter_dict, parameter_dict1)
    system_name = platform.system()
    if system_name == 'Windows':
        print("This is a Windows System")
    elif system_name == 'Linux':
        print("This is a Linux System")

    dropouts = [0.1] #[0.05,0.1,0.2,0.3,0.4]
    ed_rates = [0.1] #[0.05,0.1,0.2,0.3,0.4]
    temperatures = [0.5] #[0.1,0.3,0.5,0.7,1.0]
    negative_slopes = [0.1]
    n_layerses = [2] #[1,2,4,6,8]
    for dataset_name in dataset_name_arr:
        for dropout in dropouts:
            for ed_rate in ed_rates:
                for temperature in temperatures:
                    for negative_slope in negative_slopes:
                        for n_layers in n_layerses:
                            parameter_dict['dropout'] = dropout
                            parameter_dict['dropout1'] = dropout
                            parameter_dict['temperature'] = temperature
                            parameter_dict['ed_rate'] = ed_rate
                            parameter_dict['n_layers'] = n_layers
                            parameter_dict['negative_slope'] = negative_slope
                            main(model_name, dataset_name, parameter_dict)


# Motivation:
if __name__ == '__main__':
    parameter_dict = {
        'model_name': 'ACGAT', # ACGAT,NewGCN
        'epochs': 100,
        'train_batch_size': 4096,
        'eval_batch_size': 10240,
        'gpu_id': '0',  # (str) The id of GPU device(s).
        # 'train_neg_sample_args': None,
        'stopping_step': 20,
        # 'user_inter_num_interval': [0,20],
    }
    # param
    # set model # MODEL,SimDCL,SASRec,BERT4Rec,BPR,GRU4RecF
    # set datasets # ['steam','lfm1b-tracks','ml-1m']
    # process_base()

    # model & dataset
    dataset_name_arr = ['movielens','lfm1b-tracks','RentTheRunway','netflix']  # ['mind','RentTheRunway','netflix','anime','yelp2022','lfm1b-tracks','movielens','ml-1m']
    process_1(parameter_dict, dataset_name_arr)
