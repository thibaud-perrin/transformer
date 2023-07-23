import matplotlib.pyplot as plt

def plot_losses(losses, xlim = None, ylim = None):
    """
    Plots the training and validation losses per epoch on a black background with vibrant colors.

    The function creates a line plot with two lines - one for the training loss and one for the 
    validation loss. The x-axis represents the epoch number and the y-axis represents the loss.
    The function allows to manually set the x and y limits of the plot.

    Parameters
    ----------
    losses : dict
        A dictionary containing 'train' and 'val' lists. These lists should contain the recorded 
        losses for each epoch during training and validation, respectively.
    xlim : tuple, optional
        A tuple of two integers specifying the minimum and maximum x-values to be plotted on the 
        graph. If None, the x-axis limits will be determined automatically.
    ylim : tuple, optional
        A tuple of two integers specifying the minimum and maximum y-values to be plotted on the 
        graph. If None, the y-axis limits will be determined automatically.

    Returns
    -------
    None
        The function doesn't return a value. It displays a matplotlib plot.
    """
    train_losses = losses['train']
    val_losses = losses['val']

    plt.style.use('dark_background')
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Losses per epoch", color='white')
    plt.plot(train_losses, label='Training loss', color='pink')
    plt.plot(val_losses, label='Validation loss', color='lime')
    plt.xlabel("Epochs", color='white')
    plt.ylabel("Loss", color='white')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.show()