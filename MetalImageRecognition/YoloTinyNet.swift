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

        // initialize activation layers
        // "This filter is called leaky
        //  ReLU in CNN literature. Some CNN literature defines
        //  classical ReLU as max(0, x). If you want this behavior,
        //  simply set the a property to 0."
        relu = MPSCNNNeuronReLU(device: device!, a: alpha)
        softmax = MPSCNNSoftMax(device: device!)

        // Initialize MPSImage from descriptors
        srcImage = MPSImage(device: device!, imageDescriptor: input_id)

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

    // pixel value range is [0,1], and rgb
    // just like darknet model expects (hopefully)

    // Now we take the output from our custom shader and pass it through the
    // layers of the neural network. For each layer we use a new "temporary"
    // MPSImage to hold the results.
    // https://github.com/hollance/VGGNet-Metal/blob/master/VGGNet-iOS/VGGNet/VGGNet.swift#L279


    let conv0_img = MPSTemporaryImage(commandBuffer: commandBuffer,
                                      imageDescriptor: conv0_id)
    conv0.encode(commandBuffer: commandBuffer,
                 sourceImage: preImage,
                 destinationImage: conv0_img)


    let pool1_img = MPSTemporaryImage(commandBuffer: commandBuffer,
                                      imageDescriptor: pool1_id)
    pool1.encode(commandBuffer: commandBuffer,
                 sourceImage: conv0_img,
                 destinationImage: pool1_img)


    let conv2_img = MPSTemporaryImage(commandBuffer: commandBuffer,
                                      imageDescriptor: conv2_id)

    conv2.encode(commandBuffer: commandBuffer,
                 sourceImage: pool1_img,
                 destinationImage: conv2_img)


    let pool3_img = MPSTemporaryImage(commandBuffer: commandBuffer,
                                      imageDescriptor: pool3_id)
    pool1.encode(commandBuffer: commandBuffer,
                 sourceImage: conv2_img,
                 destinationImage: pool3_img)


    let conv4_img = MPSTemporaryImage(commandBuffer: commandBuffer,
                                      imageDescriptor: conv4_id)

    conv4.encode(commandBuffer: commandBuffer,
                 sourceImage: pool3_img,
                 destinationImage: conv4_img)


    let pool5_img = MPSTemporaryImage(commandBuffer: commandBuffer,
                                      imageDescriptor: pool5_id)
    pool5.encode(commandBuffer: commandBuffer,
                 sourceImage: conv4_img,
                 destinationImage: pool5_img)


    let conv6_img = MPSTemporaryImage(commandBuffer: commandBuffer,
                                      imageDescriptor: conv6_id)

    conv6.encode(commandBuffer: commandBuffer,
                 sourceImage: pool5_img,
                 destinationImage: conv6_img)


    let pool7_img = MPSTemporaryImage(commandBuffer: commandBuffer,
                                      imageDescriptor: pool7_id)
    pool7.encode(commandBuffer: commandBuffer,
                 sourceImage: conv6_img,
                 destinationImage: pool7_img)


    let conv8_img = MPSTemporaryImage(commandBuffer: commandBuffer,
                                      imageDescriptor: conv8_id)

    conv8.encode(commandBuffer: commandBuffer,
                 sourceImage: pool7_img,
                 destinationImage: conv8_img)

    let pool9_img = MPSTemporaryImage(commandBuffer: commandBuffer,
                                      imageDescriptor: pool9_id)
    pool9.encode(commandBuffer: commandBuffer,
                 sourceImage: conv8_img,
                 destinationImage: pool9_img)


    let conv10_img = MPSTemporaryImage(commandBuffer: commandBuffer,
                                      imageDescriptor: conv10_id)

    conv10.encode(commandBuffer: commandBuffer,
                 sourceImage: pool9_img,
                  destinationImage: conv10_img)

    let pool11_img = MPSTemporaryImage(commandBuffer: commandBuffer,
                                      imageDescriptor: pool11_id)
    pool11.encode(commandBuffer: commandBuffer,
                 sourceImage: conv8_img,
                 destinationImage: pool11_img)


    let conv12_img = MPSTemporaryImage(commandBuffer: commandBuffer,
                                      imageDescriptor: conv12_id)

    conv12.encode(commandBuffer: commandBuffer,
                 sourceImage: pool11_img,
                  destinationImage: conv12_img)


    let conv13_img = MPSTemporaryImage(commandBuffer: commandBuffer,
                                       imageDescriptor: conv13_id)

    conv13.encode(commandBuffer: commandBuffer,
                  sourceImage: conv12_img,
                  destinationImage: conv13_img)

    let conv14_img = MPSTemporaryImage(commandBuffer: commandBuffer,
                                       imageDescriptor: conv14_id)

    conv14.encode(commandBuffer: commandBuffer,
                  sourceImage: conv13_img,
                  destinationImage: conv14_img)


    // TODO (mt) : now figure out how to do a box prediction !
  }
}
