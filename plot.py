import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def visualize(dataset, n_rows):
    fig = plt.figure(1, (n_rows, n_rows))
    n = dataset.shape[0]
    n_cols = n // n_rows + (n % n_rows > 0)
    print(n_rows, n_cols)
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=(0, 0))
    for i in range(n):
        grid[i].set_xlim([0, dataset[0].shape[0]])
        grid[i].set_ylim([0, dataset[0].shape[0]])
        grid[i].get_xaxis().set_ticks([])
        grid[i].get_yaxis().set_ticks([])
    for row in range(n_rows):
        for col in range(n_cols):
            target = dataset[row * n_cols + col]
            grid[row * n_cols + col].imshow(target, cmap="gray")

    plt.show()
