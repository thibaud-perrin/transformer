import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm.notebook import tqdm

from . import estimate_loss

# learning rate warmup and then decay, which is a standard practice in Transformer training.
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """ "warm-up, then decay" strategy. """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda)

def training_loop(model, ctx, optimizer, scaler, dataset, config, saved_path = "./out/transformer_state_dict.pth"):
    """
    This function performs the training loop for the given transformer model. It trains the model using the provided 
    optimizer and dataset according to the specified configuration. 

    Args:
        - model (Transformer): The transformer model to be trained.
        - optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
        - dataset (CustomDataset): The dataset used for training and validation. It should provide a 'get_batch' method.
        - config (Config): The configuration object that defines parameters like max_iters.

    Returns:
        - losses_list (dict): A dictionary that contains the training and validation losses per evaluation step.
    """
    raw_model = model.module if config.ddp else model # unwrap DDP container if needed
    # This is the total number of training steps,
    # which is typically the number of training examples times the number of epochs.
    num_training_steps = config.train_data_size * config.max_epochs
    # Choose warmup_steps such that it's 1% of total steps
    num_warmup_steps = num_training_steps // 100
    # You can add this after defining your optimizer
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    # Initialize a dictionary to keep track of training and validation losses
    losses_list = {
        'train': [],
        'val': [],
    }
    
    # Initialize minimum loss with a high value and the iteration number where the minimum was observed
    iter_saved = 0

    # init early stop
    best_loss = float('inf')
    val_loss = float('inf')
    # This is the number of epochs with no improvement after which training will be stopped.
    patience = 25
    # This is used to keep track of the number of epochs without improvement.
    patience_counter = 0

    # Initialize a progress bar for the outer training loop
    outer_loop = tqdm(range(config.max_epochs), desc="Train loss nan, Val loss nan, Saved nan", leave=True)

    # Start the training loop
    for epochs in outer_loop:
        # Initialize a batch of data from the 'train' part of the dataset
        iter_loop = tqdm(range(config.max_iters), leave=False)
        batch = dataset.get_batch('train')
        for iter in iter_loop:
            # Sample a new batch of data
            # Sample a new batch of data
            n_batch = next(batch)
            xb = n_batch['inputs']
            xb_mask = n_batch['inputs_mask']
            yb = n_batch['targets']
            yb_mask = n_batch['targets_mask']

            
            if config.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = ((iter+1) % config.grad_accumulation_steps == 0)
            with ctx:
                # Evaluate the loss
                logits, loss = model(xb, yb, xb_mask, yb_mask)
                loss = loss / config.grad_accumulation_steps # scale the loss to account for gradient accumulation

            # Performs the backward pass with gradient scaling.
            scaler.scale(loss).backward()

            # Gradient Accumulation over multiple batches
            if (iter+1) % config.grad_accumulation_steps == 0:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # optimizer.step() is now replaced with scaler.step(optimizer)
                scaler.step(optimizer)
                
                # Updates the scale for next iteration.
                scaler.update()
                
                # Don't forget to zero the grad again after performing optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # update sheduler
                scheduler.step()

        ############
        # Evaluation
        ############
        # Estimate the losses for both training and validation datasets
        losses = estimate_loss(model, ctx, dataset, config)
        # Return the model to training mode
        model.train()
        
        # Record the estimated losses
        losses_list['train'].append(losses['train'])
        losses_list['val'].append(losses['val'])
                    
        # Get the latest losses
        last_loss_train = losses_list['train'][-1]
        last_loss_val = losses_list['val'][-1]
        
        ############
        # early stop
        ############
        val_loss = losses['val']
        if val_loss < best_loss:
            torch.save(raw_model.state_dict(), saved_path)
            best_loss = val_loss
            patience_counter = 0
            iter_saved = epochs
        else:
            patience_counter += 1

        # Update the description of the progress bar
        outer_loop.set_description(f"Train loss {last_loss_train:.4f}, Val loss {last_loss_val:.4f}, Saved {iter_saved}")
        
        if patience_counter >= patience:
            print("Early stopping")
            break
    
    # Return the list of losses
    return losses_list