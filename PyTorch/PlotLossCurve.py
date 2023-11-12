import matplotlib.pyplot as plt
from typing import Dict, List

def plot_loss_curves(results: Dict[str, List[float]], save_path: str):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "val_loss": [...],
             "val_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and val)
    loss = results['train_loss']
    val_loss = results['val_loss']

    # Get the accuracy values of the results dictionary (training and val)
    accuracy = results['train_acc']
    val_accuracy = results['val_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

     # Save the plot to the specified path
    plt.savefig(save_path)

    plt.show()



import pickle
results_path = r'E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\PyTorch\Training Results\AlexNet_1_results.pkl'
save_path = r'E:\PyTorch\Classification Tasks\Brain Tumor Classification (MRI)\PyTorch\Training Results\AlexNet_1_results.png'

# ================= Loading the Training Results =================
with open(results_path, 'rb') as file:
    loaded_results = pickle.load(file)
print(loaded_results)

# ================= Plotting the Training Results =================
plot_loss_curves(loaded_results, save_path)
