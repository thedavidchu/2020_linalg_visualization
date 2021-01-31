"""
# Gradient Descent Plotter

Plot the loss and accuracy of a gradient descent model.
"""


import numpy as np
import matplotlib.pyplot as plt


def plot(trn_loss, trn_acc, trn_batch_per_epoch=1,
         val_loss=None, val_acc=None, val_batch_per_epoch=1,
         tst_loss=None, tst_acc=None, tst_batch_per_epoch=1,
         title='Loss and Accuracy Plot', figure='Loss and Accuracy Plot'):
    """
    Plot loss and accuracy onto subplot with title and figure as listed.

    Requirements:
        import matplotlib.pyplot as plt
        import numpy as np

    Assumption:
        - sample loss and accuracy at the same rate

    # Training loss and accuracy
    :param trn_loss: list - training loss by epoch
    :param trn_acc: list - training accuracy by epoch
    :param trn_batch_per_epoch: int - training batches per epoch

    # Validation loss and accuracy
    :param val_loss: list - validation loss by epoch
    :param val_acc: list - validation accuracy by epoch
    :param val_batch_per_epoch: int - validation batches per epoch

    # Testing loss and accuracy
    :param tst_loss: list - test loss by epoch
    :param tst_acc: list - test accuracy by epoch
    :param tst_batch_per_epoch: int - test batches per epoch

    # Figure title
    :param title: str - title of plot
    :param figure: str - name of the figure

    :return: None
    """

    # Plot Expanded Nodes
    fig, axs = plt.subplots(1, 2)
    fig.canvas.set_window_title(figure)
    fig.suptitle(title)

    trn_x = np.linspace(0, (len(trn_loss) - 1)/trn_batch_per_epoch, len(trn_loss))
    val_x = np.linspace(0, (len(val_loss) - 1)/val_batch_per_epoch, len(val_loss))
    tst_x = np.linspace(0, (len(tst_loss) - 1)/tst_batch_per_epoch, len(tst_loss))

    # Plot loss
    axs[0].plot(trn_x, trn_loss, 'g', label='Training')
    if val_loss is not None:    # Plot validation loss if applicable
        axs[0].plot(val_x, val_loss, 'r', label='Validation')
    if tst_loss is not None:    # Plot test loss if applicable
        axs[0].plot(tst_x, tst_loss, 'k', label='Test')
    axs[0].set_title(f'Loss vs Epoch')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc='upper right')

    # Plot accuracy
    axs[1].plot(trn_x, trn_acc, 'g', label='Training')
    if val_acc is not None:     # Plot validation accuracy if applicable
        axs[1].plot(val_x, val_acc, 'r', label='Validation')
    if tst_acc is not None:     # Plot test accuracy if applicable
        axs[1].plot(tst_x, tst_acc, 'k', label='Test')
    axs[1].set_title(f'Accuracy vs Epoch')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(loc='lower right')

    plt.show()
    return


if __name__ == '__main__':
    pass
    
