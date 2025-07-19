import torch
from torcheval.metrics import BinaryAUROC, BinaryAUPRC, BinaryAccuracy
from torch.nn.functional import softmax
from .earlystop import EarlyStopping

def print_(message, log = True):
    if log:
        print(message)

def train_epoch(dataloader, model, loss_fn, optimizer,  device, verbose = True, recalc_train_metrics=False, log_interval=10):
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

    # Initialize metrics
    train_loss = 0
    acc_metric = BinaryAccuracy(device=device)
    auroc_metric = BinaryAUROC(device=device)
    auprc_metric = BinaryAUPRC(device=device)

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

        # Accumulate metrics
        train_loss += loss.item()
        pred_prob = softmax(pred, dim=1)
        acc_metric.update(pred_prob[:, 1], y)
        auroc_metric.update(pred_prob[:, 1], y)
        auprc_metric.update(pred_prob[:, 1], y)

        # Reduce logging
        if (batch + 1) % log_interval == 0 or (batch + 1) == len(dataloader):
            current = (batch + 1) * dataloader.batch_size
            print_(f"\tTraining Progress: \t[{current:>5d}/{len(dataloader.dataset):>5d}]", verbose)

    if recalc_train_metrics:
        return validate_epoch(dataloader, model, loss_fn, device, verbose)
    else:
        # compute metrics over all samples
        train_loss /= len(dataloader.dataset)
        train_acc = acc_metric.compute()
        train_auroc = auroc_metric.compute()
        train_auprc = auprc_metric.compute()
        return train_loss, train_auroc.item(), train_auprc.item(), train_acc.item()

def validate_epoch(dataloader, model, loss_fn, device, verbose = True, log_interval=5):
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
    acc_metric = BinaryAccuracy(device=device)
    auroc_metric = BinaryAUROC(device=device)
    auprc_metric = BinaryAUPRC(device=device)

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
            acc_metric.update(pred_prob[:, 1], y)
            auroc_metric.update(pred_prob[:, 1], y)
            auprc_metric.update(pred_prob[:, 1], y)
        
            # Reduce logging
            if (batch + 1) % log_interval == 0 or (batch + 1) == len(dataloader):
                current = (batch + 1) * dataloader.batch_size
                print_(f"\tEvaluation Progress: \t[{current:>5d}/{len(dataloader.dataset):>5d}]", verbose)

    # compute metrics over all samples
    val_loss /= len(dataloader.dataset)
    val_acc = acc_metric.compute()
    if device.type == 'mps': # MacOS MPS doesn't support binary_auroc, binary_auprc because it converts to float64
        return val_loss, val_acc.item()
    else:
        val_auroc = auroc_metric.compute()
        val_auprc = auprc_metric.compute()

        return val_loss, val_auroc.item(), val_auprc.item(), val_acc.item()

def train(num_epochs, train_data_loader, val_data_loader, model, loss_fn, optimizer, device, verbose = True, lr_scheduler=None, recalc_train_metrics=False, log_interval=10):
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
    # Initialize early stopping criteria
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    # Start iterating over epochs
    for t in range(num_epochs):
        print_(f"Epoch {t+1}\n- - - - - - - - - - - - - - - - - - - - - - - - - ", verbose)
        
        if device.type == 'mps':
            print_("Training...", verbose)
            train_loss, train_acc = train_epoch(train_data_loader, model, loss_fn, optimizer, device, recalc_train_metrics=False, log_interval=log_interval)

            print_("Validating...", verbose)
            val_loss, val_acc = validate_epoch(val_data_loader, model, loss_fn, device, log_interval=log_interval/2)

            print_(f"Training Error: \n\tLoss: {train_loss:>8f}\tAccuracy: {train_acc:>4f}", verbose)
            print_(f"Validation Error: \n\tLoss: {val_loss:>8f}\tAccuracy: {val_acc:>4f}", verbose)
            train_loss_[t], train_acc_[t], val_loss_[t], val_acc_[t] = train_loss, train_acc, val_loss, val_acc
            print_("", verbose)
        else:
            print_("Training...", verbose)
            train_loss, train_auroc, train_auprc, train_acc = train_epoch(train_data_loader, model, loss_fn, optimizer, device, recalc_train_metrics=False, log_interval=log_interval)

            print_("Validating...", verbose)
            val_loss, val_auroc, val_auprc, val_acc = validate_epoch(val_data_loader, model, loss_fn, device, log_interval=log_interval/2)

            print_(f"Training Error: \n\tLoss: {train_loss:>8f}\tROC AUC: {train_auroc:>4f}\tPR AUC: {train_auprc:>4f}\tAccuracy: {train_acc:>4f}", verbose)
            print_(f"Validation Error: \n\tLoss: {val_loss:>8f}\tROC AUC: {val_auroc:>4f}\tPR AUC: {val_auprc:>4f}\tAccuracy: {val_acc:>4f}", verbose)
            train_loss_[t], train_auroc_[t], train_auprc_[t], train_acc_[t], val_loss_[t], val_auroc_[t], val_auprc_[t], val_acc_[t] = train_loss, train_auroc, train_auprc, train_acc, val_loss, val_auroc, val_auprc, val_acc
            print_("", verbose)

        # Check early stopping criteria
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Apply learning rate schedule if provided
        if lr_scheduler:
            lr_scheduler.step()
    
    # Load weights with lowest validation loss
    early_stopping.load_best_model(model)
        
    if device.type == 'mps':
        return train_loss_, train_acc_, val_loss_, val_acc_
    else:    
        return train_loss_, train_auroc_, train_auprc_, train_acc_, val_loss_, val_auroc_, val_auprc_, val_acc_