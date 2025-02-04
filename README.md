# SuperCool

Super Cool is a fast single-image super-resolution (SISR) model capable of upscaling images 2X, 4X, and 8X while maintaining high visual fidelity. The model is trained in two stages which optimize for different objectives. The first stage focuses on upscaling using a regularized reconstruction loss and the second stage focuses on increasing visual fidelity through generative adversarial network (GAN) training where a critic model is used to fine-tune the output of the upscaler. When combined, these objectives produces a model that outputs high-resolution images that are true to the original.

## Features

- **High visual information fidelity**: SuperCool employs a secondary fine-tuning stage that increases the visual fidelity (VIF) of the output while minimizing visual artifacts by dynamically balancing a critic loss with regularized reconstruction loss. The result are upscaled images that are more faithful to the original image to the human eye.

- **Very fast**: Instead of directly predicting the individual pixels of the upscaled image, SuperCool uses a fast deterministic bilinear or bicubic upscaling algorithm and then fills in the missing details through a residual pathway that operates purely within the low-resolution space. As such, the model is capable of being used for real-time image processing even at 8X the original size.

- **Adjustable model size**: Depending on your computing budget, you can train larger or smaller models by adjusting a few hyper-parameters such as the number of hidden channels, number of hidden layers, and size/strength of the adversarial model.

- **Train on your own images**: SuperCool's dataloader works with any training images. All you have to do is point the training script to the location of the folder containing your images. This allows you to train specialized upscalers that operate on specific types of images such as satellite photos or portraits.

## Install Project Dependencies

Project dependencies are specified in the `requirements.txt` file. You can install them with [pip](https://pip.pypa.io/en/stable/) using the following command from the project root. We recommend using a virtual environment such as `venv` to keep package dependencies on your system tidy.

```
python -m venv ./.venv

source ./.venv/bin/activate

pip install -r requirements.txt
```

## Pretraining

The first stage of training involves optimizing the regularized reconstruction loss. To start training with the default settings, add your training and testing images to the `./dataset/train` and `./dataset/test` folders respectively and call the pretraining script like in the example below.

```
python pretrain.py
```

You can customize the upscaler model by adjusting the `num_channels`, `hidden_ratio`, and `num_layers` hyper-parameters like in the example below.

```
python pretrain.py --num_channels=128 --hidden_ratio=2 --num_layers=32
```

You can also adjust the `learning_rate`, `batch_size`, and `gradient_accumulation_steps` to suite your training setup.

```
python pretrain.py --batch_size=16 --learning_rate=0.01 --gradient_accumulation_steps=8
```

### Training Dashboard

We use [TensorBoard](https://www.tensorflow.org/tensorboard) to capture and display pretraining events such as loss and gradient norm updates. To launch the dashboard server run the following command from the terminal.

```
tensorboard --logdir=./runs
```

Then navigate to the dashboard using your favorite web browser.

### Pretraining Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --train_images_path | "./dataset/train" | str | The path to the folder containing your training images. |
| --test_images_path | "./dataset/test" | str | The path to the folder containing your testing images. |
| --num_dataset_processes | 4 | int | The number of CPU processes to use to process the dataset. |
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

### Fine-tuning

The fine-tuning stage of the model is optional but can greatly improve the visual fidelity (VIF) of the upscaled images without impacting the peak signal-to-noise ratio (PSNR) or structural similarity (SSIM) achieved during pretraining. To fine-tune the model from the default checkpoint at `./checkpoints/checkpoint.pt` with the default arguments you can run the following command.

```
python fine-tune.py
```

### Fine-tuning Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --train_images_path | "./dataset/train" | str | The path to the folder containing your training images. |
| --test_images_path | "./dataset/test" | str | The path to the folder containing your testing images. |
| --num_dataset_processes | 4 | int | The number of CPU processes to use to process the dataset. |
| --target_resolution | 256 | int | The number of pixels in the height and width dimensions of the training images. |
| --brightness_jitter | 0.1 | float | The amount of jitter applied to the brightness of the training images. |
| --contrast_jitter | 0.1 | float | The amount of jitter applied to the contrast of the training images. |
| --saturation_jitter | 0.1 | float | The amount of jitter applied to the saturation of the training images. |
| --hue_jitter | 0.1 | float | The amount of jitter applied to the hue of the training images. |
| --batch_size | 32 | int | The number of training images to pass through the network at a time. |
| --gradient_accumulation_steps | 4 | int | The number of batches to pass through the network before updating the model weights. |
| --critic_warmup_epochs | 3 | int | The number of epochs to train the critic model before updating the upscaler. |
| --num_epochs | 100 | int | The number of epochs to train for. |
| --learning_rate | 1e-2 | float | The learning rate of the Adafactor optimizer. |
| --rms_decay | -0.8 | float | The decay rate of the RMS coefficient of the Adafactor optimizer. |
| --tv_penalty | 0.5 | float | The strength of the total variation penalty added to the reconstruction loss. |
| --low_memory_optimizer | False | bool | Should the optimizer reduce its memory consumption in exchange for a slightly slower runtime? |
| --max_gradient_norm | 1.0 | float | Clip gradients above this threshold norm before stepping. |
| --critic_model_size | "small" | ("small", "medium", "large") | The size/strength of the critic model used in adversarial training. |
| --eval_interval | 5 | int | Evaluate the model after this many epochs on the testing set. |
| --checkpoint_interval | 10 | int | Save the model checkpoint to disk every this many epochs. |
| --checkpoint_path | "./checkpoints/fine-tuned.pt" | str | The path to the base checkpoint file on disk. |
| --resume | False | bool | Should we resume training from the last checkpoint? |
| --run_dir_path | "./runs/fine-tune" | str | The path to the TensorBoard run directory for this training session. |
| --device | "cuda" | str | The device to run the computation on. |
| --seed | None | int | The seed for the random number generator. |

## Upscaling

Coming soon ...

## References

>- J. Yu, et al. Wide Activation for Efficient and Accurate Image Super-Resolution, 2018.
>- Y. Sugawara, et al. Super-resolution Using Convolutional Neural Networks Without Any Checkerboard Artifacts, 2018.
>- C. Ledig, et al. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, 2017.
>- W. Shi, et al. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, 2016.
>- T. Salimans, et al. Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks, OpenAI, 2016.
>- C. Dong, et al. Image Super-Resolution Using Deep Convolutional Networks, 2015.
>- J. Kim, et a. Accurate Image Super-Resolution Using Very Deep Convolutional Networks.
