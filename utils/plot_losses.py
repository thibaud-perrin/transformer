import matplotlib.pyplot as plt

def plot_losses(losses):
    """
    Plots the losses for training and validation with more vibrant colors on a black background.

    Args:
        - losses (dict): A dictionary containing 'train' and 'val' lists with losses recorded for each epoch.
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
    plt.legend()
    plt.show()