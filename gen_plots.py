import numpy as np
import matplotlib.pyplot as plt
import itertools
import fileinput
import os
import sys


#Adapted from Canvas
def write_heatmap(in_file, out_file):
    conf_matrix = np.load(in_file).astype('int')
    plt.imshow(conf_matrix, interpolation='nearest')
    plt.colorbar()

    thresh = conf_matrix.max() / 2. # threshold for printing the numbers in black or white
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="black" if conf_matrix[i, j] > thresh else "white")

    plt.tight_layout()
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_file)
    plt.show()
    plt.close() # Closes for the next plot

def write_plot(train_file, out_file):
    train_vals = np.load(train_file)
    num_epochs = len(train_vals)
    x = np.linspace(0, num_epochs, num_epochs)
    plt.plot(x, train_vals, 'b-', label='Training')
    plt.plot(x, validation_vals, 'r-', label='Testing')
    plt.legend(loc='lower right')
    plt.ylabel('PSNR')
    plt.xlabel('Epochs')
    plt.savefig(out_file, bboxinches = 'tight')
    plt.show()
    plt.close()

def main(argv=sys.argv):
    suffix = argv[1]
    path_prefix = argv[2]
    train_file = os.path.join(path_prefix, 'train-' + suffix + '.npy')
    test_file = os.path.join(path_prefix, 'test-' + suffix + '.npy')
    write_plot(train_file, test_file, os.path.join(path_prefix, suffix + '.png'))

if __name__ == '__main__':
    main()


