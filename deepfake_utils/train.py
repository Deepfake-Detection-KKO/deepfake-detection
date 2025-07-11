import torch
from torcheval.metrics.functional import binary_auroc, binary_auprc, binary_accuracy
from torch.nn.functional import softmax

def train_epoch(dataloader, model, loss_fn, optimizer,  device, print_progress = True):
    """
    Iterate through all training samples once, update weights, and evaluate performance on training data
    
    Parameters
    ----------
        dataloader: torch.utils.data.dataloader.DataLoader
            Data to learn from
        model: torchvision.models
            Torch vision model to be updated (should be binary classifier)
        loss_fn: torch.nn.modules.loss
            Loss function to be evaluated and used to update weights (reduction should be 'sum', 2nd-order optimizers such as LBGFS or Conjugate Gradient are not supported)
        optimizer: torch.optim
            Optimization algorithm for learning weights
        device: {torch.device("cuda"), torch.device("cpu")}
            Whether data is to be pushed to GPU or CPU
    
    Returns
    ----------
        training loss: float
            Loss on training data after 1 iteration of training
        training ROC AUC: float
            Area under receiver operating characteristic on training data after 1 iteration of training
        training PR AUC: float
            Area under precision-recall curve (i.e. average precision) on training data after 1 iteration of training
        training accuracy: float
            Accuracy on training data after 1 iteration of training
    """
    # learn the weights
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # move data to GPU
        X = X.to(device)
        y = y.to(device)

        # forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        current = batch * dataloader.batch_size + len(X)
        if print_progress:
            print(f"Training Progress: \t[{current:>5d}/{len(dataloader.dataset):>5d}]")

    return validate_epoch(dataloader, model, loss_fn, device)

def validate_epoch(dataloader, model, loss_fn, device, print_progress = True):
    """
    Evaluate performance on passed data
    
    Parameters
    ----------
        dataloader: torch.utils.data.dataloader.DataLoader
            Data to learn from
        model: torchvision.models
            Torch vision model to be updated (should be binary classifier)
        loss_fn: torch.nn.modules.loss
            Loss function to be evaluated and used to update weights (2nd-order optimizers such as LBGFS or Conjugate Gradient are not supported)
        device: {torch.device("cuda"), torch.device("cpu")}
            Whether data is to be pushed to GPU or CPU
    
    Returns
    ----------
        validation loss: float
            Loss on passed data
        validation ROC AUC: float
            Area under receiver operating characteristic on passed data
        validation PR AUC: float
            Area under precision-recall curve (i.e. average precision) on passed data
        validation accuracy: float
            Accuracy on passed data
    """
    model.eval()

    # initialize variables to compute metrics over entire dataset
    val_loss = 0
    all_pred_prob = torch.zeros((0,2)).to(device)
    all_labels = torch.zeros(0).to(device)

    # to compute metrics properly, switch reduction method to 'sum' if not already
    loss_fn.reduction = 'sum'
    
    # collect predictions from all data
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # move data to GPU
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            
            # append batch predictions and labels
            pred_prob = softmax(pred, dim = 1)
            all_pred_prob = torch.cat((all_pred_prob, pred_prob), dim = 0)
            all_labels = torch.cat((all_labels, y), dim = 0)
        
            current = batch * dataloader.batch_size + len(X)
            if print_progress:
                print(f"Evaluation Progress: \t[{current:>5d}/{len(dataloader.dataset):>5d}]")

    # compute metrics over all samples
    val_auroc = binary_auroc(all_pred_prob[:,1], all_labels)
    val_auprc = binary_auprc(all_pred_prob[:,1], all_labels)
    val_acc = binary_accuracy(all_pred_prob[:,1], all_labels, threshold = 0.5)
    val_loss /= len(dataloader.dataset)

    return val_loss, val_auroc.item(), val_auprc.item(), val_acc.item()