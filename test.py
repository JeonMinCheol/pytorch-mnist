import torch

def test(test_dataloader, model, loss_fn, device):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, test_correct = 0, 0
    
    with torch.no_grad():
        for (x,y) in test_dataloader:
            if len(y) == 64:
                x = x.to(device)
                y = y.to(device)
                
                pred = model(x)

                test_loss += loss_fn(pred, y).item()
                test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        test_loss /= num_batches
        test_correct /= size
        
        print(f"TEST CORRECT : {(100*test_correct):0.1f} | TEST_LOSS : {test_loss:>8f}")