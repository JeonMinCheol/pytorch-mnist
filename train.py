import os
import torch
PATH = os.path.join(os.path.curdir, "checkpoint", "checkpoint.pth")

def train(train_datalodaer, model, optim, loss_fn, device):
    # x : image, y : truth
    min_loss = 10
    
    for batch, (x, y) in enumerate(train_datalodaer):
        x = x.to(device)
        y = y.to(device)
        
        if len(y) == 64:
            pred = model(x)
            loss = loss_fn(pred, y)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if batch % 200 == 0:
                print(f"BATCH : {batch} | LOSS : {loss}")
                
                if min_loss > loss:
                    min_loss = loss
                    torch.save(model.state_dict(), PATH)       
            
        