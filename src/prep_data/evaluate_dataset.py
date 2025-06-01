import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')

# function to plot file/class distribution
def plot_dataset(csv_path):
    df = pd.read_csv(csv_path) # read csv data

    # Fold distribution
    files_per_fold = df['fold'].value_counts().sort_index()
    class_counts = df['class'].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5)) # create plots

    # Fold Count bar graph
    axes[0].bar( 
        files_per_fold.index.astype(str),
        files_per_fold.values,
        width=0.9,
        color='salmon',
        edgecolor='gray'
    )
    axes[0].set_title("Number of Files per Fold") # add title
    axes[0].set_xlabel("Fold Number") # add x label
    axes[0].set_ylabel("File Count") # add y label

    # Class distribution bar chart
    axes[1].bar(
        class_counts.index,
        class_counts.values,
        width=0.9,
        color='steelblue',
        edgecolor='gray'
    )
    axes[1].set_title("Class Distribution") # add title
    axes[1].set_ylabel("Number of Samples") # add y label
    axes[1].tick_params(axis='x', rotation=45) # rotate x axis labels to 45 deg

    plt.tight_layout() # adjust layout
    plt.subplots_adjust(wspace=0.2) # add whitespace
    plt.show() # display plot