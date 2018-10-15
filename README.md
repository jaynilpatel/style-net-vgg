# neural style transfer using vgg16

Styling the images using a deep cnn for object recognition developed and trained by Oxford's renowned Visual Geometry Group (VGG), which achieved very good performance on the ImageNet dataset.

Style transfer is the technique of recomposing images in the style of other images. It all started when Gatys et al. published a [paper](https://arxiv.org/abs/1508.06576) on how it was actually possible to transfer artistic style from one painting to another picture using convolutional neural networks. Here is an online demonstration given at https://deepdreamgenerator.com/:

<p align="center">
<img align="center" src="https://b2h3x3f6.stackpathcdn.com/assets/landing/img/blend/horizontal/ts.jpg" width="400">
</p>

#### Working:

My implementation uses TensorFlow for training the network. The model used here Oxford University's VGG16 network. The network is less complex than that of Google's Inception network but uses a lot of parameters. Therefore, the training may take longer on a less powerful machine. For defining the content, I have used 4th layer's convolution for the determining the content since deeper layers of the network have wider receptive field.  For defining the style, I have stored output of every layers convolution in a list which will help further in determining the loss. Instead of using the raw activations of these layers, what the authors of the StyleNet paper suggest is to use the Gram activation of the layers instead, which mathematically is expressed as the matrix transpose multiplied by itself. The intuition behind this process is that it measures the similarity between every feature of a matrix. Or put another way, it is saying how often certain features appear together.

As far as optimization of the network is concerned, I have defined 3 loss functions to optimize. First loss function is for defining the Content Loss which tries to optimize the distance between the net's output at our content layer, and the content features. Second loss function is for defining Style Loss which compute the gram matrix of the current network output, and then measure the l2 loss with the precomputed style image's gram matrix. Lastly, a third loss is Total Variational Loss which will simply measure the difference between neighboring pixels. By including this as a loss, we're saying that we want neighboring pixels to be similar and no drastic change happens.
With all 3 loss functions, I have combined them, optimized the loss function, and created a stylized image.

```
loss = 0.1 * content_loss + 5.0 * style_loss + 0.01 * tv_loss
```

#### Output:
<p align="center">
<img align="center" src="https://github.com/jaynilpatel/style-transfer-vgg/blob/master/imgs/result1.PNG">
</p>

<p align="center">
<img align="center" src="https://github.com/jaynilpatel/style-transfer-vgg/blob/master/imgs/stylenet-gif1.gif" width="300">
</p>

<p align="center">
<img align="center" src="https://github.com/jaynilpatel/style-transfer-vgg/blob/master/imgs/result3.PNG">
</p>

<p align="center">
<img align="center" src="https://github.com/jaynilpatel/style-transfer-vgg/blob/master/imgs/stylenet-gif3.gif" width="300">
</p>

<p align="center">
<img align="center" src="https://github.com/jaynilpatel/style-transfer-vgg/blob/master/imgs/result2.PNG">
</p>

<p align="center">
<img align="center" src="https://github.com/jaynilpatel/style-transfer-vgg/blob/master/imgs/stylenet-gif2.gif" width="300">
</p>

#### Applications:

This technique to style is not just limited to images, but can also be extended to music, audio and videos. Click below to see the demo. 

[![style transfer video](https://img.youtube.com/vi/OW0V99nr0Ys/0.jpg)](https://youtu.be/OW0V99nr0Ys)
