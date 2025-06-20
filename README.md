# Super Cool

Super Cool is a fast single-image super-resolution (SISR) model capable of upscaling images with high visual fidelity.

## Comparison

![Super Cool 4X Small Comparison](https://raw.githubusercontent.com/andrewdalpino/SuperCool/master/docs/images/comparison-4x-small.png)

![Super Cool 4X Medium Comparison](https://raw.githubusercontent.com/andrewdalpino/SuperCool/master/docs/images/comparison-4x-medium.png)

![Super Cool 4X Large Comparison](https://raw.githubusercontent.com/andrewdalpino/SuperCool/master/docs/images/comparison-4x-large.png)

## Install Project Dependencies

Project dependencies are specified in the `requirements.txt` file. You can install them with [pip](https://pip.pypa.io/en/stable/) using the following command from the project root. We recommend using a virtual environment such as `venv` to keep package dependencies on your system tidy.

```
python -m venv ./.venv

source ./.venv/bin/activate

pip install -r requirements.txt
```

## Training

To start training with the default settings, add your training and testing images to the `./dataset/train` and `./dataset/test` folders respectively and call the pretraining script like in the example below. If you are looking for good training sets to start with we recommend the `DIV2K`, `Flicker2K`, and/or Outdoor Scene Train/Test (`OST`) datasets.

```
python train.py
```

You can customize the upscaler model by adjusting the `num_channels`, `hidden_ratio`, and `num_layers` hyper-parameters like in the example below.

```
python train.py --num_channels=128 --hidden_ratio=2 --num_layers=32
```

You can also adjust the `batch_size`, `learning_rate`, and `gradient_accumulation_steps` to suite your training setup.

```
python train.py --batch_size=16 --learning_rate=0.01 --gradient_accumulation_steps=8
```

In addition, you can control various training data augmentation arguments such as the brightness, contrast, hue, and saturation jitter.

```
python train.py --brightness_jitter=0.5 --contrast_jitter=0.4 --hue_jitter=0.3 --saturation_jitter=0.2
```

### Training Dashboard

We use [TensorBoard](https://www.tensorflow.org/tensorboard) to capture and display pretraining events such as loss and gradient norm updates. To launch the dashboard server run the following command from the terminal.

```
tensorboard --logdir=./runs
```

Then navigate to the dashboard using your favorite web browser.

### Training Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --train_images_path | "./dataset/train" | str | The path to the folder containing your training images. |
| --test_images_path | "./dataset/test" | str | The path to the folder containing your testing images. |
| --num_dataset_processes | 4 | int | The number of CPU processes to use to preprocess the dataset. |
| --target_resolution | 256 | int | The number of pixels in the height and width dimensions of the training images. |
| --upscale_ratio | 4 | (2, 4, 8) | The upscaling factor. |
| --brightness_jitter | 0.1 | float | The amount of jitter applied to the brightness of the training images. |
| --contrast_jitter | 0.1 | float | The amount of jitter applied to the contrast of the training images. |
| --saturation_jitter | 0.1 | float | The amount of jitter applied to the saturation of the training images. |
| --hue_jitter | 0.1 | float | The amount of jitter applied to the hue of the training images. |
| --batch_size | 32 | int | The number of training images to pass through the network at a time. |
| --gradient_accumulation_steps | 4 | int | The number of batches to pass through the network before updating the model weights. |
| --num_epochs | 400 | int | The number of epochs to train for. |
| --learning_rate | 1e-2 | float | The learning rate of the Adafactor optimizer. |
| --rms_decay | -0.8 | float | The decay rate of the RMS coefficient of the Adafactor optimizer. |
| --tv_penalty | 0.5 | float | The strength of the total variation penalty added to the reconstruction loss. |
| --low_memory_optimizer | False | bool | Should the optimizer reduce its memory consumption in exchange for a slightly slower runtime? |
| --max_gradient_norm | 1.0 | float | Clip gradients above this threshold norm before stepping. |
| --base_upscaler | "bicubic" | ("bilinear", "bicubic") | The base upscaler that feeds into the residual pathway. |
| --num_channels | 128 | int | The number of channels within each encoder block. |
| --hidden_ratio | 2 | (1, 2, 4) | The ratio of hidden channels to `num_channels` within the activation portion of each encoder block. |
| --num_layers | 16 | int | The number of blocks within the body of the encoder. |
| --eval_interval | 5 | int | Evaluate the model after this many epochs on the testing set. |
| --checkpoint_interval | 10 | int | Save the model checkpoint to disk every this many epochs. |
| --checkpoint_path | "./checkpoints/checkpoint.pt" | str | The path to the base checkpoint file on disk. |
| --resume | False | bool | Should we resume training from the last checkpoint? |
| --run_dir_path | "./runs/pretrain" | str | The path to the TensorBoard run directory for this training session. |
| --device | "cuda" | str | The device to run the computation on. |
| --seed | None | int | The seed for the random number generator. |

## Upscaling

You can use the provided `upscale.py` script to generate upscaled images from the trained model at the default checkpoint like in the example below. In addition, you can create your own inferencing pipeline using the same model under the hood that leverages batch processing for large scale production systems.

```
python upscale.py --image_path="./example.jpg"
```

To generate images using a different checkpoint you can use the `checkpoint_path` argument like in the example below.

```
python upscale.py --checkpoint_path="./checkpoints/fine-tuned.pt" --image_path="./example.jpg"
```

### Upscaling Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --image_path | None | str | The path to the image file to be upscaled by the model. |
| --checkpoint_path | "./checkpoints/fine-tuned.pt" | str | The path to the base checkpoint file on disk. |
| --device | "cuda" | str | The device to run the computation on. |

## References

>- J. Yu, et al. Wide Activation for Efficient and Accurate Image Super-Resolution, 2018.
>- C. Ledig, et al. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, 2017.
>- W. Shi, et al. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, 2016.
>- T. Salimans, et al. Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks, OpenAI, 2016.
