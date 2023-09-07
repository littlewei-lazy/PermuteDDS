import torch
import torch.nn as nn
from tqdm import tqdm 
from metrics import AverageMeter


def run_a_train_epoch(device, epoch, model, data_loader, loss_criterion, optimizer, scheduler=None):
    model.train()
    tbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for id, (*x, y) in tbar:

        for i in range(len(x)):
            x[i] = x[i].to(device)
        y = y.to(device)

        optimizer.zero_grad()
        # output1, output2, output3 = model(*x)
         
        # loss1 = loss_criterion(output1.view(-1), y.view(-1))
        # loss2 = loss_criterion(output2.view(-1), y.view(-1))
        # loss3 = loss_criterion(output3.view(-1), y.view(-1))

        # loss = (loss1 + loss2 + loss3) / 3
        loss = 0
        outputs = model(*x)

        for output in outputs:
            loss += loss_criterion(output.view(-1), y.view(-1))

        loss = loss / len(outputs)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item()  :.3f}')
        # tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item()  :.3f}  AUX_loss={aux_loss.item()  :.3f}')
    

def run_an_eval_epoch(device, model, data_loader, task_name, loss_criterion):
    model.eval()
    running_loss = AverageMeter()

    with torch.no_grad():
        preds = torch.Tensor()
        trues = torch.Tensor()
        for batch_id, (*x, y) in tqdm(enumerate(data_loader)):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)
            
            # logits1, logits2, logits3 =  model(*x)
            # logits = (logits1 + logits2 + logits3) / 3

            outputs =  model(*x)
            logits = 0
            for logit in outputs:
                logits += logit
            logits /= len(outputs)

            loss = loss_criterion(logits.view(-1), y.view(-1))

            if task_name == 'classification':
                logits = torch.sigmoid(logits)
            preds = torch.cat((preds, logits.cpu()), 0)
            trues = torch.cat((trues, y.view(-1, 1).cpu()), 0)
            running_loss.update(loss.item(), y.size(0))
        preds, trues = preds.numpy().flatten(), trues.numpy().flatten()
    val_loss = running_loss.get_average()
    return preds, trues, val_loss
