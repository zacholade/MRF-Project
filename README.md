Repository for this codebase can be found at https://github.com/zacholade/MRF-Project.

# Running the Application
Before proceeding, note that the full codes used in this project can be found at: https://github.com/zacholade/MRF-Project

## Setting up the environment
Setting up the development environment is made easy due to the inclusion of a Dockerfile. This installation guide assumes your native OS is running Linux for use on the University of Bath Hex GPU cloud.

- Firstly, make sure docker is installed on your system by following the steps found at: 
```
https://docs.docker.com/engine/install/ubuntu/
```

- Once installed, proceed to build an image from the provided Dockerfile. This will setup your Python installation, along with installing all associated Python packages. This will take several minutes to complete. In a terminal, navigate to the base directory containing the Dockerfile file and run the following command: 
```console
docker build -t <image_tag_here> .
```
    
-Assuming you would like to use a Nvidia GPU with cuda capabilities, find a free GPU and note its identifier with the following terminal command:
```
nvidia-smi
```


## Training Models
The main entry point into the training part of the application is the `mrf_trainer.py` file found at the root directory of the code. This file accepts various command line arguments which modify how the trainer will function and what model to use. Furthermore, there exists a config.ini file which contains a configuration entry for each model used in this project. This is where you can modify model specific parameters and enable or disable sub-modules such as attention, channel reduction, and so on.


An example configuration entry for the R(2+1)D model found inside the config.ini file. An entry exists for each model implemented in this project. During run time, the application code parses this file to initialise the model with the correct parameters
```ini
[r2plus1dHyperparameters]
seq_len = 300  # How many timepoints (channels) the fingerprints should have
total_epochs = 1000  # How many epochs to train (early stop may prevent reaching this.)
batch_size = 100 
lr = 0.001  # Initial learning rate
lr_step_size = 2  # How often to update the lr
lr_gamma = 0.97  # How much to update the lr by.
patch_size = 3  # The input patch size into the model
factorise = True  # Whether or not to factorise the 3D convolutions.
non_local_level = 1  # 0 = None, 1 = Temporal for 3D, 2 = Spatio-temporal
dimensionality_reduction_level = 0  # 0 = None, 1 = CBAM, 2 = Feature Extraction
```

The code expects data and labels for training models to be contained inside the `Data/` directory. There is a file provided called `gen_data.py` which contains all the code requires to generate fingerprint maps from the parameter files provided by our supervisor. This script also generates all requires folders. The following steps will explain how to run the training part of the algorithm:

- After following the installation steps, the container is ready to run. Note that several other arguments will need to be passed to launch the training algorithm (See further below). Where `?` denotes the GPU you would like to run the application, the base command to do this is:
```
docker run --rm --runtime=nvidia -e NVIDIA_VISIBILE_DEVICES=? <image_tag_here>
```

- mrf_trainer.py is missing some required arguments. Most importantly, a model must be provided. The complete list of models which can be run are:
```
cohen, oksuz_rnn, hoppe, song, balsiger, rca_unet, soyak, path_size, r2plus1d
```
    
- Specify a model using the -network argument:
```
docker run --rm --runtime=nvidia -e NVIDIA_VISIBILE_DEVICES=? <image_tag_here> -network <network_name_here>
```
    
- A full list of arguments you can pass are as follows:
```
-network    : The neural network you would like to train. Required.
-debug      : Defaults to false. Whether to run in debug mode which limits the number of iterations per epoch and number of files to open.
-workers    : Defaults to 1. The number of sub processes to use to load training data.
-skip_valid : Defaults to false. Whether to skip validation
-plot_every : Defaults to 1. Plots an anatomical brain map every n epochs.
-no_plot    : Defaults to false. Won't do any plotting all together. Overrides plot_every.
-notes      : Type string. If provided, will save all plots and models in a directory with this name included.
-cpu        : Defaults to false. Force to use the cpu for running the application.
-chunks     : Defaults to 100. Running a full scan in one batch is memory intensive. Validation scans will be done in separate batches to save memory.
-file_limit : Limit the number of files to open at one time.
```

While running, the training script will output to stdout to let you know how it is progressing. Furthermore, this information is logged to a file called `logs.log` using the Python logging module. This can be found in the subdirectory named `Exports/`. The training script will terminate either after the number of epochs specified in the `config.ini` file is exceeded, or when the early stop algorithm previously mentioned terminates for you, whichever comes first. 

At the end of each epoch, a logs.csv file is updated with the training statistics to let you know how it is going along. Information include statistics for the training and validation stages such as MAPE, MSE, MAE, loss, and so on. Additionally, svg image files are saved in the `Exports/Plots/` directory of each validation scan at the end of each epoch unless specified otherwise. Lastly, two models are saved in this Exports directory at the end of each epoch. These can be loaded for inference at a later stage. Whereas the models exported to the `Exports/Models/` directory only contain the specific weight values using the PyTorch `state_dict()` function, the models exported to `Exports/CompleteModels/` contain pickled Python objects. The latter is useful if changes are made to the code initially used to train the model, as the first model output is not robust to such changes and requires the original code to exist relative to the model file.

## Testing the models
Inference of the models can be done using the `eval.py` file.


- Firstly, the model you would like to run for inference that was outputted from the training algorithm should be moved to the `FinalModels/` directory. This can be the model file simply containing the learned weights or the complete model containing a pickled Python class. If using the latter, the script should be passed the command line argument `-full_model`, so that the evaluation script knows how to load it.

- In addition to this parameter, the evaluation script should still be passed the `-network` argument so that it knows what format to pass data to the model and what configuration parameters should be set from the `config.ini` file.
    
- Output from this script are to the `Exports/Test/` directory. Information outputted to this directory contain overall reconstruction accuracy, as well as per tissue accuracy, and svg image plots of the estimated reconstructed parameter maps.
    
- Two additional arguments: `-snr`, and `-cs` can be passed to use noise at a specific signal-to-noise ratio or undersampled data, respectively. If snr is unspecified, standard Gaussian noise with 0 mean and 1% SD will be added. Likewise, no undersampling will take place if undersampling is unspecified.

