# Model Training Script Usage

This script trains a CNN model for image classification using various configurable parameters. Below is an explanation of each argument and how to use the script in Google Colab.

## Arguments

- `--dataset`: (Required) Name of the dataset folder.
- `--model_name`: Architecture of the model to use. Options: 'simple_cnn' or 'deeper_cnn'. Default: 'simple_cnn'.
- `--epochs`: Number of training epochs. Default: 20.
- `--batch_size`: Batch size for training. Default: 32.
- `--learning_rate`: Learning rate for the optimizer. Default: 0.001.
- `--optimizer`: Optimizer to use for training. Options: 'Adam', 'RMSprop', 'SGD'. Default: 'Adam'.
- `--dropout_rate`: Dropout rate to use in the model. Default: 0.5.
- `--augmentation`: Flag to enable data augmentation. Default: False.

## Usage in Google Colab

To run this script in Google Colab, follow these steps:

1. Clone the repository:
   ```
   !git clone https://github.com/seungboAn/National_OceanoGraphic.git
   ```

2. Change to the repository directory:
   ```
   %cd National_OceanoGraphic/model_train
   ```

3. Install required packages:
   ```
   !pip install wandb tensorflow
   ```

4. Run the training script:
   ```
   !python train_model.py --dataset 1_kaggle --model_name base_model
   ```