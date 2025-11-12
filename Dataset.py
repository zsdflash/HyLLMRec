import random
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset

def data_load(dataset):

    random.seed(42) 

    dir_str = './Data/' + dataset

    v_feat = torch.load(dir_str+'/v_feature.pt')
    t_feat = torch.load(dir_str+'/t_feature.pt')
    

    l_feat1 = torch.load(dir_str+'/t1_feature.pt')
    l_feat2 = torch.load(dir_str+'/t2_feature.pt')
    l_feat3 = torch.load(dir_str+'/t3_feature.pt')
    l_feat4 = torch.load(dir_str+'/t4_feature.pt')
    l_feat5 = torch.load(dir_str+'/t5_feature.pt')

    key1 = torch.load(dir_str+'/t_key1_feature.pt')
    key2 = torch.load(dir_str+'/t_key2_feature.pt')
    key3 = torch.load(dir_str+'/t_key3_feature.pt')
    key4 = torch.load(dir_str+'/t_key4_feature.pt')
    key5 = torch.load(dir_str+'/t_key5_feature.pt')
    key6 = torch.load(dir_str+'/t_key6_feature.pt')
    key7 = torch.load(dir_str+'/t_key7_feature.pt')
    key8 = torch.load(dir_str+'/t_key8_feature.pt')
    key9 = torch.load(dir_str+'/t_key9_feature.pt')
    key10 = torch.load(dir_str+'/t_key10_feature.pt')
    key_list = [key1, key2, key3, key4, key5, key6, key7, key8, key9, key10]

    local_feat_stack = torch.stack((l_feat1,l_feat2,l_feat3,l_feat4,l_feat5), dim=1) 

    local_feat = local_feat_stack.permute(0, 2, 1) 


    interaction = pd.read_csv(dir_str+'/interaction.csv')

    num_user = interaction['user_id'].nunique()
    num_item = interaction['item_id'].nunique()

    print('Dataset:', dataset)
    print("users: {0}, items: {1}".format(num_user, num_item))
    print("interactions: {0}".format(interaction.shape[0]))
    print("sparsity: {0}".format(1 - (interaction.shape[0] / (num_user * num_item))))
    print("v_feat: {0}, t_feat: {1}".format(v_feat.shape, t_feat.shape))

    # split train, val, test 0.8 0.1 0.1
    all_user_item = (interaction.groupby("user_id")["item_id"].apply(list).to_dict())
    user_item_dict, val_data, test_data = {}, {}, {}

    for user, items in all_user_item.items():
        items = items.copy()
        random.shuffle(items)

        n = len(items)

        train_size = int(n * 0.8)
        val_size = int(n * 0.1)

        train_items = items[:train_size]
        val_items = items[train_size:train_size + val_size]
        test_items = items[train_size + val_size:]

        user_item_dict[user] = train_items
        val_data[user] = val_items
        test_data[user] = test_items

    train_edge = [[user, item] for user, items in user_item_dict.items() for item in items]

    return num_user, num_item, train_edge, user_item_dict, v_feat, t_feat, val_data, test_data, local_feat, key_list

class TrainingDataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict, edge_index):
        self.edge_index = edge_index
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.all_set = set(range(num_user, num_user+num_item))

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, index):
        user, pos_item = self.edge_index[index]
        while True:
            neg_item = random.sample(self.all_set, 1)[0]
            if neg_item not in self.user_item_dict[user]:
                break
        return torch.LongTensor([user,user]), torch.LongTensor([pos_item, neg_item])
