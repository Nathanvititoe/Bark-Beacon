import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from IPython.display import Audio, display, Markdown
import matplotlib.lines as mlines
plt.style.use('dark_background')
# get jupyter to display plots in-line
try:
    get_ipython() # type: ignore
except NameError:
    import matplotlib
    matplotlib.use('TkAgg')

# visualize loss v acc during training and validation 
def visualize_stats(classifier_history):
    # get metrics
    train_loss = classifier_history.history.get('loss', [])
    val_loss = classifier_history.history.get('val_loss', [])
    train_acc = classifier_history.history.get('accuracy', [])
    val_acc = classifier_history.history.get('val_accuracy', [])

    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(15,5)) # set plot size

    # Plot the training vs. validation Accuracy
    plt.subplot(1,2,1) 
    plt.plot(epochs, train_acc, label='Training Accuracy') # Training acc plot
    plt.plot(epochs, val_acc, label='Validation Accuracy') # Validation acc plot
    plt.title('Training vs Validation Accuracy') # Title
    plt.xlabel('Epoch') # x label
    plt.ylabel('Accuracy') # y label
    plt.legend() # legend

    # plot training v. validation Loss
    plt.subplot(1,2,2) 
    plt.plot(epochs, train_loss, label='Training Loss') # training loss plot
    plt.plot(epochs, val_loss, label='Validation Loss') # validation loss plot
    plt.title('Training vs Validation Loss') # title
    plt.xlabel('Epoch') # x label
    plt.ylabel('Loss') # y label
    plt.legend() # legend

    plt.tight_layout() # adjust spacing between the plots
    plt.show() # display them

# function to plot/compare raw v processed audio waveforms
def plot_waveform_comparison(class_waveforms_raw, class_waveforms_proc):
    num_classes = len(class_waveforms_raw) # get number of classes
    fig, axes = plt.subplots(num_classes, 2, figsize=(15, 2*num_classes), constrained_layout=True) # create subplots
    fig.suptitle("Raw vs. Processed Audio Waveforms\n\n", fontsize=14) # add title (position and fontsize)
    
    # iterate through audio waveform dict (for each class) 
    for i, label in enumerate(class_waveforms_raw):
        raw_waveform, sr = class_waveforms_raw[label] # get raw waveform
        proc_waveform, sr = class_waveforms_proc[label] # get processed waveform

        # convert samples to time for human readability (x axis)
        raw_time = np.arange(len(raw_waveform)) / sr 
        proc_time = np.arange(len(proc_waveform)) / sr

        # plot raw audio waveform channels
        if raw_waveform.ndim == 2:  # if stereo audio
            axes[i, 0].plot(raw_time, raw_waveform[:, 0], label='Left Channel', color='yellow', alpha=0.5) # plot left channel 
            axes[i, 0].plot(raw_time, raw_waveform[:, 1], label='Right Channel', color='red', alpha=0.5) # plot right channel
            axes[i, 0].set_xlim(raw_time[0], raw_time[-1]) # trim stereo whitespace
        else:  # if mono audio
            axes[i, 0].plot(raw_time, raw_waveform, label='Mono', color='orange') # plot single channel
            axes[i, 0].set_xlim(raw_time[0], raw_time[-1]) # trim mono whitespace
        
        # Raw Audio Titles
        axes[i, 0].set_title(f"{label} - Raw", fontsize=9) # add title for each subplot
        axes[i, 0].set_ylabel("Amplitude", fontsize=7) # add y label

        # processed audio plot and title
        axes[i, 1].plot(proc_time, proc_waveform, color='orange') # create plot
        axes[i, 1].set_title(f"{label} - Processed", fontsize=9) # add title
        axes[i, 1].set_xlim(proc_time[0], proc_time[-1])  # trim processed whitespace

    # add x label
    for ax in axes.flat:
        ax.set_xlabel("Time (sec)", fontsize=7) # add x labels
        ax.tick_params(axis='both', labelsize=6, length=2, pad=2) # set sizing for axis markings

    # create legend to show mono v. stereo
    legend_lines = [
    mlines.Line2D([], [], color='yellow', label='Left Channel'),
    mlines.Line2D([], [], color='red', label='Right Channel'),
    mlines.Line2D([], [], color='orange', label='Processed Mono')
    ]
    fig.legend(handles=legend_lines, loc='upper left', ncol=3, fontsize='medium') # add legend
    
    plt.tight_layout() # adjust layout
    plt.show() # display plots

# display one spectrogram per class
def plot_spectrograms(class_spectrograms):
    labels = sorted(class_spectrograms.keys()) # get class labels
    num_classes = len(labels) # get total number of classes
    num_cols = 2 # number of columns
    num_rows = int(np.ceil(num_classes / num_cols)) # number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_classes), squeeze=True) # create subplots for each class
    axes = axes.flatten() # allow for easy iteration over axes

    # iterate through class_spectrograms
    for i, label in enumerate(labels):
        spec = class_spectrograms[label] # get the spectrogram
        ax = axes[i] # select subplot
        ax.imshow(spec, aspect='auto', origin='lower', cmap='magma') # show the spectrogram
        ax.set_title(f"{label}", fontsize=10) # add title (label)
        # ax.axis('off') # turn off axis labels

        # remove axis ticks manually 
        ax.set_xticks([])
        ax.set_yticks([])

        # add border to each spectrogram
        for spine in ax.spines.values():
            spine.set_visible(True) # ensure border is visible
            spine.set_edgecolor('white') # border color
            spine.set_linewidth(1) # border thickness 

    # disable unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle("UrbanSound Spectrograms\n", fontsize=14) # add title
    plt.tight_layout() # adjust layout
    plt.show() # display plot

# function to display an audio sample to the user
def audio_sampler(filepath, sample_rate, duration, label):
    from src.prep_data.preprocess import load_file # import here to avoid cyclic error
    try:
        num_samples = sample_rate * duration # get num_samples
        audio = load_file(filepath, sample_rate, num_samples).numpy() # load sample file
 
        display(Markdown(f"**{label}**")) # add title
        display(Audio(audio, rate=sample_rate)) # show audio sample

    except Exception as e:
        print(f"Could not load {filepath}:\n→ {e}")


# function to display confusion matrix and monitor what the model is struggling with
def plot_confusion_matrix(audio_classifier, val_features, val_labels, label_names):
    print("Loading Features for Confusion Matrix...")
    y_pred = audio_classifier.predict(val_features) # get prediction on validation features
    y_pred_labels = np.argmax(y_pred, axis=1) # get prediction labels

    fig, ax = plt.subplots(figsize=(6, 4), squeeze=True)  # adjust size to fit everything

    # create scikit confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(
            val_labels, # validation labels 
            y_pred_labels, # prediction labels
            display_labels=label_names, # names of labels
            cmap='plasma', # color scheme
            ax=ax, # use plt subplot for display
            colorbar=False # turn off built in colorbar scale
            )
    
    disp.ax_.set_title("Confusion Matrix", fontsize=14) # add title/subtitle

    img = disp.im_  # get matrix image
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04) # add colorbar scale manually to match matrix height

    # rotate x axis labels for readability
    disp.ax_.set_xticklabels(disp.ax_.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout() # adjust layout
    plt.show() # display plot