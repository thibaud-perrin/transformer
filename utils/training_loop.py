from tqdm.notebook import tqdm

from . import estimate_loss

def training_loop(model, optimizer, dataset, config, saved_path = "./out/transformer_state_dict.pth"):
    """
    This function performs the training loop for the given transformer model. It trains the model using the provided 
    optimizer and dataset according to the specified configuration. 

    Args:
        - model (Transformer): The transformer model to be trained.
        - optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
        - dataset (CustomDataset): The dataset used for training and validation. It should provide a 'get_batch' method.
        - config (Config): The configuration object that defines parameters like max_iters and eval_interval.

    Returns:
        - losses_list (dict): A dictionary that contains the training and validation losses per evaluation step.
    """

    # Initialize a batch of data from the 'train' part of the dataset
    batch = dataset.get_batch('train')
    
    # Initialize a dictionary to keep track of training and validation losses
    losses_list = {
        'train': [],
        'val': [],
    }
    
    # Initialize minimum loss with a high value and the iteration number where the minimum was observed
    min_loss = 100
    iter_saved = 0

    # Initialize a progress bar for the outer training loop
    outer_loop = tqdm(range(config.max_iters), desc="Train loss nan, Val loss nan, Saved nan", leave=True)

    # Start the training loop
    for iter in outer_loop:
        # Every eval_interval iterations, compute and log the losses on the training and validation sets
        if iter % config.eval_interval == 0:
            # Estimate the losses for both training and validation datasets
            losses = estimate_loss(model, dataset, config)
            # Return the model to training mode
            model.train()
            # Record the estimated losses
            losses_list['train'].append(losses['train'])
            losses_list['val'].append(losses['val'])
            # Refresh the training batch
            batch = dataset.get_batch('train')
            # Get the latest losses
            last_loss_train = losses_list['train'][-1]
            last_loss_val = losses_list['val'][-1]
            
            # If the current validation loss is the lowest, save the model and update min_loss and iter_saved
            if min_loss > last_loss_val:
                model.save_model(saved_path)
                min_loss = last_loss_val
                iter_saved = iter
                            
            # Update the description of the progress bar
            outer_loop.set_description(f"Train loss {last_loss_train:.4f}, Val loss {last_loss_val:.4f}, Saved {iter_saved}")
            
        # Sample a new batch of data
        n_batch = next(batch)
        xb = n_batch['inputs']
        yb = n_batch['targets']
    
        # Evaluate the loss
        logits, loss = model(xb, yb)
        # Reset gradients
        optimizer.zero_grad(set_to_none=True)
        # Perform backpropagation
        loss.backward()
        # Update the model parameters
        optimizer.step()

    # Estimate the losses before closing for both training and validation datasets
    losses = estimate_loss(model, dataset, config)
    
    # If the current validation loss is the lowest, save the model
    if min_loss > losses['val']:
        model.save_model("./out/transformer_state_dict.pth")
    
    # Return the list of losses
    return losses_list