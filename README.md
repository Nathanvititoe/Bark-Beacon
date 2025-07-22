Dataset - 
    
    empty on github but the dataset can be found at:
    https://www.kaggle.com/datasets/nathanvititoe/dog-vocalization-dataset

images -

    a few screenshots throughout model progress as it was created

visualizations -

    generated matplotlib images of audio waveforms, spectrograms and dataset class distribution

Within vocalization_classifier -

    ConvertForArduino contains:
        get_lite_model.py - a script to convert a saved h5 of the full model to a TF lite model
        get_c_array.py - a script to convert a saved TF lite model to a c array, which is required to      upload the model to an Arduino 

    Models contains: 
        h5 files - full model
        tflite files - quantized models for microcontrollers
        cc files - c arrays of the tf lite models, needed for Arduino use

    src contains: 
        All the code required to create the original full model, train it, and visualize the data

    tf_lite_utils contains: 
        converter - utility functions for converting h5 (full model) files to tf lite models
        tflite_utils - utility functions to 
            load_lite_model - load a lite model from a file
            lite_inference - get an inference from the saved tf lite model
            compare_models - gets predictions from the full h5 model and the tf lite model to compare   accuracy scores
    main.py -
        runs all logic to preprocess data, split dataset, train full model, convert to tf lite, and compare the accuracy scores between the full model and the lite model
combine_datasets.py - 

    A script for combining various folders of datasets into one large combined dataset with classes for bark, growl, whine and howl only (taken from the end of the file names)

extract_audio.py - 

    A script for preprocessing individual audio clips, converting them all to monochannel with a 16 kHz sample rate and then denoising them by removing all audio background noise based on a 1s sample 
