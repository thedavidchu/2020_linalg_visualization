"""
# Gradient Descent Plotter

Plot the loss and accuracy of a gradient descent model.
"""


import numpy as np
import matplotlib.pyplot as plt


def plot(trn_loss=None, trn_acc=None, trn_batch_per_epoch=1,
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
        - we are plotting at least one loss and one accuracy

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

    # Declare constants
    TRN_CLR = 'g-'
    VAL_CLR = 'r-'
    TST_CLR = 'k-'

    # Space out training x-axis
    if trn_loss is not None:
        trn_x = np.linspace(0, (len(trn_loss) - 1) / trn_batch_per_epoch, len(trn_loss))
    elif trn_acc is not None:
        trn_x = np.linspace(0, (len(trn_acc) - 1) / trn_batch_per_epoch, len(trn_acc))
    # Space out validation x-axis
    if val_loss is not None:
        val_x = np.linspace(0, (len(val_loss) - 1) / val_batch_per_epoch, len(val_loss))
    elif val_acc is not None:
        val_x = np.linspace(0, (len(val_acc) - 1) / val_batch_per_epoch, len(val_acc))
    # Space out testing x-axis
    if tst_loss is not None:
        tst_x = np.linspace(0, (len(tst_loss) - 1) / tst_batch_per_epoch, len(tst_loss))
    elif tst_acc is not None:
        tst_x = np.linspace(0, (len(tst_acc) - 1) / tst_batch_per_epoch, len(tst_acc))

    # Plot Expanded Nodes
    fig, axs = plt.subplots(1, 2)
    fig.canvas.set_window_title(figure)
    fig.suptitle(title)

    # Plot loss
    if trn_loss is not None:
        axs[0].plot(trn_x, trn_loss, TRN_CLR, label='Training')
    if val_loss is not None:    # Plot validation loss if applicable
        axs[0].plot(val_x, val_loss, VAL_CLR, label='Validation')
    if tst_loss is not None:    # Plot test loss if applicable
        axs[0].plot(tst_x, tst_loss, TST_CLR, label='Test')
    axs[0].set_title(f'Loss vs Epoch')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc='upper right')

    # Plot accuracy
    if trn_acc is not None:
        axs[1].plot(trn_x, trn_acc, TRN_CLR, label='Training')
    if val_acc is not None:     # Plot validation accuracy if applicable
        axs[1].plot(val_x, val_acc, VAL_CLR, label='Validation')
    if tst_acc is not None:     # Plot test accuracy if applicable
        axs[1].plot(tst_x, tst_acc, TST_CLR, label='Test')
    axs[1].set_title(f'Accuracy vs Epoch')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(loc='lower right')

    plt.show()
    return


if __name__ == '__main__':
    pass
    
