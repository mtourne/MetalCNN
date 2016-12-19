//
//  YoloTinyNet.swift
//  MetalImageRecognition
//
//  Created by Matthieu Tourne on 12/11/16.
//  Copyright © 2016 Dhruv Saksena. All rights reserved.
//

import Foundation
import UIKit
import MetalKit
import MetalPerformanceShaders
import Accelerate


class YoloTinyNet{
    // we keep the MTLDevice and MTLCommandQueue objects around for ease of use
    var device : MTLDevice!
    var commandQueue : MTLCommandQueue


    // pre-processing layers and an MPSTemporaryImage for it
    var lanczos : MPSImageLanczosScale!
    var scale : MPSCNNNeuronLinear!
    var preImage : MPSTemporaryImage!

    // MPSImages are declared we need srcImage and final softMax output as MPSImages so we can read/write to underlying textures
    var srcImage : MPSImage


    // standard neuron and softmax layers are declared
    let alpha : Float = 0.1                 // leaky relu constant
    var relu : MPSCNNNeuronReLU

    var softmax : MPSCNNSoftMax

    // convolution and fully connected layers
    let conv0, conv2, conv4, conv6, conv8, conv10, conv12, conv13, conv14 : SlimMPSCNNConvolution
    var pool1, pool3, pool5, pool7, pool9, pool11 : MPSCNNPoolingMax


    /* These MPSImage descriptors tell the network about the sizes of the data
       volumes that flow between the layers. */
    let input_id = MPSImageDescriptor(channelFormat: textureFormat,
                                      width: 416, height: 416, featureChannels: 3)
    let conv0_id  = MPSImageDescriptor(channelFormat: .float16,
                                       width: 416, height: 416, featureChannels: 16)
    let pool1_id  = MPSImageDescriptor(channelFormat: .float16,
                                       width: 208, height: 208, featureChannels: 16)
    let conv2_id  = MPSImageDescriptor(channelFormat: .float16,
                                       width: 208, height: 208, featureChannels: 32)
    let pool3_id  = MPSImageDescriptor(channelFormat: .float16,
                                       width:  104, height:  104, featureChannels: 32)
    let conv4_id  = MPSImageDescriptor(channelFormat: .float16,
                                       width:  104, height:  104, featureChannels: 64)
    let pool5_id  = MPSImageDescriptor(channelFormat: .float16,
                                       width:  52, height:  52, featureChannels: 64)
    let conv6_id  = MPSImageDescriptor(channelFormat: .float16,
                                       width:  52, height:  52, featureChannels: 128)
    let pool7_id  = MPSImageDescriptor(channelFormat: .float16,
                                       width:  26, height:  26, featureChannels: 128)
    let conv8_id  = MPSImageDescriptor(channelFormat: .float16,
                                       width:  26, height:  26, featureChannels: 256)
    let pool9_id = MPSImageDescriptor(channelFormat: .float16,
                                      width:  13, height:  13, featureChannels: 256)
    let conv10_id  = MPSImageDescriptor(channelFormat: .float16,
                                        width:  13, height:  13, featureChannels: 512)
    let pool11_id = MPSImageDescriptor(channelFormat: .float16,
                                       width:  13, height:  13, featureChannels: 512)
    let conv12_id  = MPSImageDescriptor(channelFormat: .float16,
                                        width:  13, height:  13, featureChannels: 1024)
    let conv13_id  = MPSImageDescriptor(channelFormat: .float16,
                                        width:  13, height:  13, featureChannels: 1024)
    let conv14_id  = MPSImageDescriptor(channelFormat: .float16,
                                        width:  13, height:  13, featureChannels: 425)


    init(withCommandQueue inputCommandQueue : MTLCommandQueue){

        // keep an instance of device and commandQueue around for use
        device = inputCommandQueue.device
        commandQueue = inputCommandQueue

        // we will resize the input image the input size of yolo
        lanczos = MPSImageLanczosScale(device: device)
        // we will scale pixel values to [-1,1]
        scale = MPSCNNNeuronLinear(device: device!, a: Float(2), b: Float(-1))

        // initialize activation layers
        // "This filter is called leaky
        //  ReLU in CNN literature. Some CNN literature defines
        //  classical ReLU as max(0, x). If you want this behavior,
        //  simply set the a property to 0."
        relu = MPSCNNNeuronReLU(device: device!, a: alpha)
        softmax = MPSCNNSoftMax(device: device!)

        // Initialize MPSImage from descriptors
        srcImage    = MPSImage(device: device!, imageDescriptor: sid)

        // define convolution, pooling and fullyConnected layers and
        // initialize them with proper weights this will occur as a 1
        // time cost during app launch, which is beneficial to us
        conv0 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 3,
                                      outputFeatureChannels: 16,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv_0" ,
                                      padding: false,
                                      strideXY: (1, 1))
        // by default pool should be 'SAME'
        pool1 = MPSCNNPoolingMax(device: device!,
                                 kernelWidth: 2, kernelHeight: 2,
                                 strideInPixelsX: 2, strideInPixelsY: 2)

        conv2 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 16,
                                      outputFeatureChannels: 32,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv_2" ,
                                      padding: false,
                                      strideXY: (1, 1))

        pool3 = MPSCNNPoolingMax(device: device!,
                                 kernelWidth: 2, kernelHeight: 2,
                                 strideInPixelsX: 2, strideInPixelsY: 2)


        conv4 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 32,
                                      outputFeatureChannels: 64,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv_4" ,
                                      padding: false,
                                      strideXY: (1, 1))

        pool5 = MPSCNNPoolingMax(device: device!,
                                 kernelWidth: 2, kernelHeight: 2,
                                 strideInPixelsX: 2, strideInPixelsY: 2)

        conv6 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 64,
                                      outputFeatureChannels: 128,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv_6" ,
                                      padding: false,
                                      strideXY: (1, 1))

        pool7 = MPSCNNPoolingMax(device: device!,
                                 kernelWidth: 2, kernelHeight: 2,
                                 strideInPixelsX: 2, strideInPixelsY: 2)

        conv8 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 128,
                                      outputFeatureChannels: 256,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv_8" ,
                                      padding: false,
                                      strideXY: (1, 1))

        pool9 = MPSCNNPoolingMax(device: device!,
                                 kernelWidth: 2, kernelHeight: 2,
                                 strideInPixelsX: 2, strideInPixelsY: 2)

        conv10 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 256,
                                      outputFeatureChannels: 512,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv_10" ,
                                      padding: false,
                                      strideXY: (1, 1))

        // pool11 has a stride of 1 (doesn't half the resolution)
        pool11 = MPSCNNPoolingMax(device: device!,
                                 kernelWidth: 2, kernelHeight: 2,
                                 strideInPixelsX: 1, strideInPixelsY: 1)

        conv12 = SlimMPSCNNConvolution(kernelWidth: 3,
                                       kernelHeight: 3,
                                       inputFeatureChannels: 512,
                                       outputFeatureChannels: 1024,
                                       neuronFilter: relu,
                                       device: device,
                                       kernelParamsBinaryName: "conv_12" ,
                                       padding: false,
                                       strideXY: (1, 1))

        conv13 = SlimMPSCNNConvolution(kernelWidth: 3,
                                       kernelHeight: 3,
                                       inputFeatureChannels: 1024,
                                       outputFeatureChannels: 1024,
                                       neuronFilter: relu,
                                       device: device,
                                       kernelParamsBinaryName: "conv_13" ,
                                       padding: false,
                                       strideXY: (1, 1))

        // last conv as a fully connected layer (1,1) kernel
        conv14 = SlimMPSCNNConvolution(kernelWidth: 1,
                                       kernelHeight: 1,
                                       inputFeatureChannels: 1024,
                                       outputFeatureChannels: 425,
                                       neuronFilter: nil,
                                       device: device,
                                       kernelParamsBinaryName: "conv_14" ,
                                       padding: false,
                                       strideXY: (1, 1))

    }

  // make an inference
  func forward( commandBuffer: MTLCommandBuffer, sourceTexture : MTLTexture?) {
    // This lets us squeeze some extra speed out of Metal.
    MPSTemporaryImage.prefetchStorage(with: commandBuffer, imageDescriptorList: [
      input_id, conv0_id, pool1_id, conv2_id, pool3_id, conv4_id, pool5_id,
      conv6_id, pool7_id, conv8_id, pool9_id, conv10_id, pool11_id,
      conv12_id, conv13_id, conv14_id
    ])

    // we use preImage to hold preprocesing intermediate results
    preImage = MPSTemporaryImage(commandBuffer: commandBuffer,
                                 imageDescriptor: input_id)

    // encode pre-processing layers to change input image to a size of 416*416*3
    lanczos.encode (commandBuffer: commandBuffer, sourceTexture: sourceTexture!, destinationTexture: preImage.texture)

    // with values in range [-1,1]

    // XX (mtourne) Things to think about
    // * what is the encoding for YOLO Net ?

    // Adjust the RGB values of each pixel to be in the range -128...127
    // by subtracting the "mean pixel". If the input texture is RGB, this
    // also swaps the R and B values because the model expects BGR pixels.
    // As far as I can tell there is no MPS shader that can do these things,
    // so we use a custom compute kernel.

    // Now we take the output from our custom shader and pass it through the
    // layers of the neural network. For each layer we use a new "temporary"
    // MPSImage to hold the results.
    // https://github.com/hollance/VGGNet-Metal/blob/master/VGGNet-iOS/VGGNet/VGGNet.swift#L279
    scale.encode(commandBuffer: commandBuffer, sourceImage: preImage, destinationImage: srcImage)

  }
}
