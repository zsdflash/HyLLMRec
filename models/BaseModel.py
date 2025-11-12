import torch
import math

class BaseModel(torch.nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        
        self.num_user = config['num_user']
        self.num_item = config['num_item']
        self.aggr_mode = config['aggr_mode']
        self.user_item_dict = config['user_item_dict']
        self.device = config['device']

        self.weight = torch.tensor([[1.0],[-1.0]]).to(self.device)
        self.train_edge = config['train_edge']
        self.edge_index_ = torch.tensor(self.train_edge).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index_, self.edge_index_[[1,0]]), dim=1)
        self.v_feat = config['v_feat'].to(self.device)
        self.t_feat = config['t_feat'].to(self.device)
        self.local_feat = config['local_feat'].to(self.device)
        self.key_list = [k.to(self.device) for k in config['key_list']]
        self.dim_E = config['dim_E']
        self.batch_size = config['batch_size']
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method must be implemented by subclasses")
    
    def loss(self, user_tensor, item_tensor, out):
        user_tensor = user_tensor.view(-1)
        item_tensor = item_tensor.view(-1)

        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score*item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))

        return loss
    
    
    def full_accuracy(self, val_data, topk=[5, 10], batch_size=1024):
        user_tensor = self.result[:self.num_user] 
        item_tensor = self.result[self.num_user:] 
        item_tensor_t = item_tensor.t()

        metrics = {k: {'precision': 0.0, 'recall': 0.0, 'ndcg': 0.0} for k in topk}
        valid_user_count = 0

        user_ids = list(val_data.keys())
        num_users = len(user_ids)

        for start in range(0, num_users, batch_size):
            end = min(start + batch_size, num_users)
            batch_user_ids = user_ids[start:end]
            batch_user_emb = user_tensor[batch_user_ids] 

            score_matrix = torch.matmul(batch_user_emb, item_tensor_t)

        
            for i, user_id in enumerate(batch_user_ids):
                pos_items = self.user_item_dict.get(user_id, [])
                if pos_items:
                    item_indices = torch.LongTensor(pos_items) - self.num_user
                    score_matrix[i, item_indices] = -1e9  

            _, index_of_rank_list = torch.topk(score_matrix, max(topk), dim=1)
            index_of_rank_list += self.num_user 

            for i, user_id in enumerate(batch_user_ids):
                pos_items = val_data[user_id]
                num_pos = len(pos_items)
                if num_pos == 0:
                    continue
                valid_user_count += 1

                ranked_list = index_of_rank_list[i].tolist()
                for k in topk:
                    topk_items = set(ranked_list[:k])
                    hit_items = [x for x in topk_items if x in pos_items]
                    num_hit = len(hit_items)

                    metrics[k]['precision'] += num_hit / k
                    metrics[k]['recall'] += num_hit / num_pos

                    dcg = sum(1 / math.log2(i + 2) for i, item in enumerate(ranked_list[:k]) if item in pos_items)
                    idcg = sum(1 / math.log2(i + 2) for i in range(min(num_pos, k)))
                    if idcg > 0:
                        metrics[k]['ndcg'] += dcg / idcg

        for k in topk:
            if valid_user_count > 0:
                metrics[k]['precision'] /= valid_user_count
                metrics[k]['recall'] /= valid_user_count
                metrics[k]['ndcg'] /= valid_user_count

        return metrics, valid_user_count