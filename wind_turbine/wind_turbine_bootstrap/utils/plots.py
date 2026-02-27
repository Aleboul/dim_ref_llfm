import matplotlib.pyplot as plt

def plot_matrix(M, title="Matrix", cmap='viridis'):
    plt.figure(figsize=(6, 4))
    plt.imshow(M, cmap=cmap, interpolation='none')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()

def plot_hist(values, title="Histogram", bins=50):
    plt.figure(figsize=(6,4))
    plt.hist(values, bins=bins, edgecolor='black')
    plt.title(title)
    plt.grid(True)
    plt.show()

