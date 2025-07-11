import torch
from torcheval.metrics.functional import binary_auroc, binary_auprc, binary_accuracy
from torch.nn.functional import softmax

def print_(message, log = True):
    if log:
        print(message)

def train_epoch(dataloader, model, loss_fn, optimizer,  device, verbose = True):
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
        print_(f"\tTraining Progress: \t[{current:>5d}/{len(dataloader.dataset):>5d}]", verbose)

    return validate_epoch(dataloader, model, loss_fn, device, verbose)

def validate_epoch(dataloader, model, loss_fn, device, verbose = True):
    """
    Evaluate performance on passed data
    
    Parameters
    ----------
        dataloader: torch.utils.data.dataloader.DataLoader
            Data to evaluate performance on
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
            print_(f"\tEvaluation Progress: \t[{current:>5d}/{len(dataloader.dataset):>5d}]", verbose)

    # compute metrics over all samples
    val_auroc = binary_auroc(all_pred_prob[:,1], all_labels)
    val_auprc = binary_auprc(all_pred_prob[:,1], all_labels)
    val_acc = binary_accuracy(all_pred_prob[:,1], all_labels, threshold = 0.5)
    val_loss /= len(dataloader.dataset)

    return val_loss, val_auroc.item(), val_auprc.item(), val_acc.item()

def train(num_epochs, train_data_loader, val_data_loader, model, loss_fn, optimizer, device, verbose = True):
    """
    Train on data for a specified number of epochs
    
    Parameters
    ----------
        num_epochs: int
            Number of complete passes through training data
        train_data_loader: torch.utils.data.dataloader.DataLoader
            Data to learn from
        val_data_loader: torch.utils.data.dataloader.DataLoader
            Data to evaluate performance on
        model: torchvision.models
            Torch vision model to be updated (should be binary classifier)
        loss_fn: torch.nn.modules.loss
            Loss function to be evaluated and used to update weights (2nd-order optimizers such as LBGFS or Conjugate Gradient are not supported)
        optimizer: torch.optim
            Optimization algorithm for learning weights
        device: {torch.device("cuda"), torch.device("cpu")}
            Whether data is to be pushed to GPU or CPU
    
    Returns
    ----------
        train_loss: list(float)
        train_auroc: list(float)
        train_auroc: list(float)
        train_acc: list(float)
        val_loss: list(float)
        val_auroc: list(float)
        val_auroc: list(float)
        val_acc: list(float)
            Performance metrics logged at each epoch of training
    """
    # setup to training history
    train_loss_, train_auroc_, train_auprc_, train_acc_, val_loss_, val_auroc_, val_auprc_, val_acc_ = [None] * num_epochs, [None] * num_epochs, [None] * num_epochs, [None] * num_epochs, [None] * num_epochs, [None] * num_epochs, [None] * num_epochs, [None] * num_epochs
    for t in range(num_epochs):
        print_(f"Epoch {t+1}\n- - - - - - - - - - - - - - - - - - - - - - - - - ", verbose)
        
        print_("Training...", verbose)
        train_loss, train_auroc, train_auprc, train_acc = train_epoch(train_data_loader, model, loss_fn, optimizer, device)

        print_("Validating...", verbose)
        val_loss, val_auroc, val_auprc, val_acc = validate_epoch(val_data_loader, model, loss_fn, device)
        
        print_(f"Training Error: \n\tLoss: {train_loss:>8f}\tROC AUC: {train_auroc:>4f}\tPR AUC: {train_auprc:>4f}\tAccuracy: {train_acc:>4f}", verbose)
        print_(f"Validation Error: \n\tLoss: {val_loss:>8f}\tROC AUC: {val_auroc:>4f}\tPR AUC: {val_auprc:>4f}\tAccuracy: {val_acc:>4f}", verbose)
        train_loss_[t], train_auroc_[t], train_auprc_[t], train_acc_[t], val_loss_[t], val_auroc_[t], val_auprc_[t], val_acc_[t] = train_loss, train_auroc, train_auprc, train_acc, val_loss, val_auroc, val_auprc, val_acc
        print_("", verbose)
    return train_loss_, train_auroc_, train_auprc_, train_acc_, val_loss_, val_auroc_, val_auprc_, val_acc_