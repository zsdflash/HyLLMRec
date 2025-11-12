
# Implementation Reference:
# This code builds upon existing implementations from:
# [1] https://github.com/weiyinwei/MMGCN
# [2] https://github.com/enoche/MMRec


import argparse
import numpy as np
import torch

from Dataset import TrainingDataset, data_load
from models.HyLLMRec import HyLLMRec

from torch.utils.data import DataLoader
from Train import train
from datetime import datetime
from torch.autograd import no_grad


def full_vt(epoch, model, data, prefix, topK):   
    model.eval()
    with no_grad():
        metrics, valid_user_count = model.full_accuracy(data, topK)
        print('{0}-epoch '.format(epoch+1) + prefix + ' users:{0}'.format(valid_user_count))
        for k in metrics:
            print('Top@{0} result {1}]'.format(k, metrics[k]))
        return metrics,valid_user_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--early_stop', type=int, default=40, help='early stop.')
    parser.add_argument('--dataset', '-d', default='baby', help='Dataset')
    parser.add_argument('--model', '-m', default='MMGCN', help='Model name.')

    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=16, help='Workers number.')

    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--aggr_mode', default='mean', help='Aggregation Mode.')
    parser.add_argument('--topK', type=int, default=[5,10,20], help='Workers number.')  

    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)
    device = torch.device(f"cuda:{args.gpu}")
    num_workers = args.num_workers

    dataset = args.dataset
    model_name = args.model
    
    learning_rate = args.l_r
    dim_E = args.dim_E
    aggr_mode = args.aggr_mode
    
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    topK = args.topK
    early_stop = args.early_stop
    
    num_user, num_item, train_edge, user_item_dict, v_feat, t_feat, val_data, test_data, local_feat, key_list = data_load(dataset)
    train_dataset = TrainingDataset(num_user, num_item, user_item_dict, train_edge)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    config = {
        'v_feat': v_feat,
        't_feat': t_feat,
        'local_feat': local_feat,
        'train_edge': train_edge,
        'num_user': num_user,
        'num_item': num_item,
        'aggr_mode': aggr_mode,
        'user_item_dict': user_item_dict,
        'dim_E': dim_E,
        'device': device,
        'batch_size': batch_size,
        'key_list':key_list
    }
    if model_name == 'HyLLMRec':
        model = HyLLMRec(config).to(device)
    else:
        raise ValueError("Model name not recognized.")
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])
    
    val_max = 0.0
    increase_count = 0

    final_metrics = None
    for epoch in range(num_epoch):
        
        print('Epoch: {0}'.format(epoch + 1))
        loss = train(epoch + 1, len(train_dataset), train_dataloader, model, optimizer, batch_size)
        
        if torch.isnan(loss):
            print('lr: {0} \t loss is Nan\r\n'.format(learning_rate))
            break
        torch.cuda.empty_cache()

        metrics, valid_user_count = full_vt(epoch, model, val_data, 'Val', topK)
        
        if metrics[20]['ndcg'] > val_max:
            val_max = metrics[20]['ndcg']
            metrics, valid_user_count = full_vt(epoch, model, test_data, 'Test', topK)
            final_metrics = metrics
            increase_count = 0
        else:
            if increase_count > early_stop:
                print("The model has been trained for {0} epochs, and the best test result is\n {1}".format(epoch, final_metrics))

                time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                with open('./Data/'+ dataset +'/{0}_{1}_{2}_result.txt'.format(model_name,dataset,time_str), 'a') as save_file:
                    save_file.write('lr: {0} \t =====>best result:\n{1}\n test users:{2}'.
                                    format(learning_rate, final_metrics, valid_user_count))
                break
            else:
                increase_count += 1