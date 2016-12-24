
# Weights for Yolo

Tools :
available
* darknet to tensorflow
* tensorflow to MPS

## Darknet

Extraction code from darknet to Tensorflow written sequentially

```
	def conv_layer(self,idx,inputs,filters,size,stride):
		channels = inputs.get_shape()[3]
		f_w = open(self.weights_dir + str(idx) + '_conv_weights.txt','r')
		l_w = np.array(f_w.readlines()).astype('float32')
		f*w.close()
		w = np.zeros((size,size,channels,filters),dtype='float32')
		ci = int(channels)
		filter_step = ci*size*size
		channel_step = size*size
		for i in range(filters):
			for j in range(ci):
				for k in range(size):
					for l in range(size):
						w[k,l,j,i] = l_w[i*filter_step + j*channel_step + k*size + l]
```
k is filter_height
l is filter_width
j is input_depth
i is output_depth


## Tensorflow

"The ordering of convolution weight values is often tricky to deal
with when converting between different frameworks. In TensorFlow, the
filter weights for the Conv2D operation are stored on the second
input, and are expected to be in the order
[filter_height, filter_width, input_depth, output_depth], where
filter_count increasing by one means moving to an adjacent value in
memory." [2]


## Apple MPS-CNNConvolution

"Each entry is a float value. The number of entries is equal to inputFeatureChannels * outputFeatureChannels * kernelHeight * kernelWidth.

The layout of the filter weight is arranged so that it can be reinterpreted as a 4D tensor (array) weight[outputChannels][kernelHeight][kernelWidth][inputChannels/groups]
" [3]


[2] https://www.tensorflow.org/versions/r0.10/how_tos/tool_developers/
[3] https://developer.apple.com/reference/metalperformanceshaders/mpscnnconvolution/1648861-init



# Metal Image Recognition: Performing Image Recognition with Inception_v3 Network using Metal Performance Shaders Convolutional Neural Network routines

This sample demonstrates how to perform runtime inference for image recognition using a Convolutional Neural Network (CNN) built with Metal Performance Shaders. This sample is a port of the TensorFlow-trained Inception_v3 network, which was trained offline using the ImageNet dataset. The CNN creates, encodes, and submits different layers to the GPU. It then performs image recognition using trained parameters (weights and biases) that have been acquired and saved from the pre-trained network.

The Network can be found here:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/imagenet/classify_image.py

Instructions to run this network on TensorFlow can be found here:
https://www.tensorflow.org/versions/r0.8/tutorials/image_recognition/index.html#image-recognition

The Original Network Paper can be found here:
http://arxiv.org/pdf/1512.00567v3.pdf

The network parameters are included in binary .dat files that are memory-mapped when needed.

The weights for this particular network were batch normalized but for inference the following may be used for every feature channel separately to get the corresponding weights and bias:

A = 𝛄 / √(s + 0.001), b = ß - ( A * m )

W = w*A

s: variance
m: mean
𝛄: gamma
ß: beta

w: weights of a feature channel
b: bias of a feature channel
W: batch nomalized weights

This is derived from:
https://arxiv.org/pdf/1502.03167v3.pdf

## Requirements

### Build

Xcode 8.0 or later; iOS 10.0 SDK or later

### Runtime

iOS 10.0 or later

### Device Feature Set

iOS GPU Family 2 v1
iOS GPU Family 2 v2
iOS GPU Family 3 v1

Copyright (C) 2016 Apple Inc. All rights reserved.
