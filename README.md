# SuperCool

Super Cool is a fast single-image super-resolution (SISR) model capable of upscaling images 2X, 4X, and 8X while maintaining high visual fidelity. The model is trained in two stages which optimize for different objectives. The first stage focuses on upscaling using a regularized reconstruction loss and the second stage focuses on increasing visual fidelity through generative adversarial network (GAN) training where a critic model is used to fine-tune the output of the upscaler. When combined, these objectives yield a model that produces upscaled images that are true to the original.

## Features

- **High visual information fidelity**: SuperCool employs a secondary fine-tuning stage that increases the visual fidelity (VIF) of the output while minimizing visual artifacts by dynamically balancing a critic loss with regularized reconstruction loss. The result are upscaled images that are more faithful to the original image to the human eye.

- **Very fast**: Instead of directly predicting the individual pixels of the upscaled image, SuperCool uses a fast deterministic bilinear or bicubic upscaling algorithm and then fills in the missing details through a residual pathway that operates purely within the low-resolution space. As such, the model is capable of being used for real-time image processing even at 8X the original size.

- **Adjustable model size**: Depending on your computing budget, you can train larger or smaller models by adjusting a few hyper-parameters such as the number of hidden channels, number of hidden layers, and size of the adversarial model.

- **Train on your own images**: SuperCool's dataloader works with any training images. All you have to do is point the training script to the location of the folder containing your images. This allows you to train specialized upscalers that operate on specific types of images such as satellite photos or portraits.

## Install Project Dependencies

Project dependencies are specified in the `requirements.txt` file. You can install them with [pip](https://pip.pypa.io/en/stable/) using the following command from the project root. We recommend using a virtual environment such as `venv` to keep package dependencies on your system tidy.

```
python -m venv ./.venv

source ./.venv/bin/activate

pip install -r requirements.txt
```

## Pretraining

Coming soon ...

### Fine-tuning

Coming soon ...

## References

>- J. Yu, et al. Wide Activation for Efficient and Accurate Image Super-Resolution, 2018.
>- Y. Sugawara, et al. Super-resolution Using Convolutional Neural Networks Without Any Checkerboard Artifacts, 2018.
>- C. Ledig, et al. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, 2017.
>- W. Shi, et al. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, 2016.
>- T. Salimans, et al. Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks, OpenAI, 2016.
>- C. Dong, et al. Image Super-Resolution Using Deep Convolutional Networks, 2015.
>- J. Kim, et a. Accurate Image Super-Resolution Using Very Deep Convolutional Networks.
