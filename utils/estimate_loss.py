import torch
from tqdm.notebook import tqdm

@torch.no_grad()
def estimate_loss(model, dataset, config, splits = ['train', 'val']):
    """
    This function estimates the loss of a model on specified data splits without performing backpropagation.
    It sets the model to evaluation mode, iterates over the data splits and calculates the average loss.

    Args:
        model (Transformer): The model for which loss needs to be estimated.
        dataset (CustomDataset): The dataset used for estimation. It should provide a 'get_batch' method.
        config (Config): The configuration object defining the number of evaluation iterations.
        splits (list[str]): List of the names of data splits to use for estimation.

    Returns:
        out (dict): A dictionary with split names as keys and corresponding average loss as values.
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
            logits, loss = model(X, Y, X_mask, Y_mask)

            # Store the current loss
            losses[k] = loss.item()
            
        # Calculate and store the mean loss for the current split
        out[split] = losses.mean()

    # Return the dictionary with the average losses
    return out