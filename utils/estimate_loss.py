import torch
from tqdm.notebook import tqdm

@torch.no_grad()
def estimate_loss(model, ctx, dataset, config, splits = ['train', 'val']):
    """
    Estimates the average loss for the provided model on the given data splits.

    This function puts the model into evaluation mode, then iteratively samples 
    batches from the dataset and computes the loss for each split. The function 
    returns a dictionary containing the average loss for each data split.

    Parameters
    ----------
    model : torch.nn.Module
        The model for which the loss is to be estimated.
    ctx : contextlib._GeneratorContextManager
        The context manager for gradient computation. This is typically the 
        result of a torch.no_grad() or torch.enable_grad() context.
    dataset : object
        The dataset object, which should have a get_batch() method for obtaining
        batches of data.
    config : object
        The configuration object. It should have the following attributes:
            eval_iters (int): The number of iterations to perform for each split.
    splits : list, optional
        The list of data splits for which the loss should be estimated. The 
        default is ['train', 'val'].

    Returns
    -------
    dict
        A dictionary where the keys are the names of the data splits and the 
        values are the estimated average losses for those splits.
    """
    # Create an empty dictionary to store the average loss for each split
    out = {}

    # Set the model to evaluation mode
    model.eval()

    # Loop over the data splits
    for split in splits:
        # Initialize a tensor to store the losses for each iteration in the current split
        losses = torch.zeros(config.eval_iters)

        # Get a batch iterator for the current split
        batch = dataset.get_batch(split)
        
        # Initialize a progress bar for the inner loop
        inner_loop = tqdm(range(config.eval_iters), desc=f"Evaluation - {split}", leave=False)

        # Start the inner loop
        for k in inner_loop:
            # Sample a new batch of data
            n_batch = next(batch)
            X = n_batch['inputs']
            X_mask = n_batch['inputs_mask']
            Y = n_batch['targets']
            Y_mask = n_batch['targets_mask']
            
            # Evaluate the loss for the current batch
            with ctx:
                logits, loss = model(X, Y, X_mask, Y_mask)

            # Store the current loss
            losses[k] = loss.item()
            
        # Calculate and store the mean loss for the current split
        out[split] = losses.mean()

    # Return the dictionary with the average losses
    return out