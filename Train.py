from tqdm import tqdm

def train(epoch, length, dataloader, model, optimizer, batch_size, writer=None):    
    model.train()
    sum_loss = 0.0
    step = 0.0
    
    pbar = tqdm(total=length)
    pbar.set_description('Epoch: {0}'.format(epoch))

    for user_tensor, item_tensor in dataloader:
        optimizer.zero_grad()
        out = model()
        loss = model.loss(user_tensor, item_tensor, out)

        loss.backward()
        optimizer.step()

        pbar.update(batch_size)

        sum_loss += loss.cpu().item()
        step += 1.0

    pbar.close()
    print('--- loss value:{0}, mean loss{1}---'.format(sum_loss, sum_loss/step))
    return loss
