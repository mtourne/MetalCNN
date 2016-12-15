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
    var srcImage, sftImage : MPSImage
    
    
    // standard neuron and softmax layers are declared
    let alpha : Float = 0.1                 // leaky relu constant
    var relu : MPSCNNNeuronReLU
    
    var softmax : MPSCNNSoftMax
    
    // convolution and fully connected layers
    let conv1, conv3, conv5, conv7, conv9, conv11, conv13, conv14, conv15, conv16 : SlimMPSCNNConvolution
    
    var pool2, pool4, pool6, pool8 : MPSCNNPoolingMax

    
    // MPSImageDescriptor for different mixed layer outputs
    let sid   = MPSImageDescriptor(channelFormat: textureFormat, width: 448, height: 448, featureChannels: 3)
    // XX (mtourne): softmax output, probably not what we need here.
    let sftid = MPSImageDescriptor(channelFormat: textureFormat, width: 1  , height: 1  , featureChannels: 1008)


    
    init(withCommandQueue inputCommandQueue : MTLCommandQueue){
        
        // keep an instance of device and commandQueue around for use
        device = inputCommandQueue.device
        commandQueue = inputCommandQueue
        
        // we will resize the input image to 299x299x3 (input size to inception_v3) size using lanczos
        lanczos = MPSImageLanczosScale(device: device)
        // we will scale pixel values to [-1,1]
        scale = MPSCNNNeuronLinear(device: device!, a: Float(2), b: Float(-1))
        
        // initialize activation layers
        //This filter is called leaky ReLU in CNN literature. Some CNN literature defines classical ReLU as max(0, x). If you want this behavior, simply set the a property to 0.
        relu = MPSCNNNeuronReLU(device: device!, a: alpha)
        softmax = MPSCNNSoftMax(device: device!)
        
        // Initialize MPSImage from descriptors
        srcImage    = MPSImage(device: device!, imageDescriptor: sid)
        sftImage    = MPSImage(device: device!, imageDescriptor: sftid)
        
        // define convolution, pooling and fullyConnected layers and initialize them with proper weights
        // this will occur as a 1 time cost during app launch, which is beneficial to us
        conv1 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 3,
                                      outputFeatureChannels: 16,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv1" ,
                                      padding: false,
                                      strideXY: (1, 1))
        // by default pool should be 'SAME'
        pool2 = MPSCNNPoolingMax(device: device!,
                                 kernelWidth: 2, kernelHeight: 2,
                                 strideInPixelsX: 2, strideInPixelsY: 2)
        
        conv3 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 16,
                                      outputFeatureChannels: 32,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv3" ,
                                      padding: false,
                                      strideXY: (1, 1))
        
        pool4 = MPSCNNPoolingMax(device: device!,
                                 kernelWidth: 2, kernelHeight: 2,
                                 strideInPixelsX: 2, strideInPixelsY: 2)
        
        
        conv5 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 32,
                                      outputFeatureChannels: 64,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv5" ,
                                      padding: false,
                                      strideXY: (1, 1))
        
        pool6 = MPSCNNPoolingMax(device: device!,
                                 kernelWidth: 2, kernelHeight: 2,
                                 strideInPixelsX: 2, strideInPixelsY: 2)
        
        conv7 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 64,
                                      outputFeatureChannels: 128,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv7" ,
                                      padding: false,
                                      strideXY: (1, 1))
        
        pool8 = MPSCNNPoolingMax(device: device!,
                                 kernelWidth: 2, kernelHeight: 2,
                                 strideInPixelsX: 2, strideInPixelsY: 2)
        
        conv9 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 128,
                                      outputFeatureChannels: 256,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv9" ,
                                      padding: false,
                                      strideXY: (1, 1))
        
        conv11 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 256,
                                      outputFeatureChannels: 512,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv11" ,
                                      padding: false,
                                      strideXY: (1, 1))

        conv13 = SlimMPSCNNConvolution(kernelWidth: 3,
                                       kernelHeight: 3,
                                       inputFeatureChannels: 512,
                                       outputFeatureChannels: 1024,
                                       neuronFilter: relu,
                                       device: device,
                                       kernelParamsBinaryName: "conv13" ,
                                       padding: false,
                                       strideXY: (1, 1))
        conv14 = SlimMPSCNNConvolution(kernelWidth: 3,
                                       kernelHeight: 3,
                                       inputFeatureChannels: 1024,
                                       outputFeatureChannels: 1024,
                                       neuronFilter: relu,
                                       device: device,
                                       kernelParamsBinaryName: "conv14" ,
                                       padding: false,
                                       strideXY: (1, 1))
        
        conv15 = SlimMPSCNNConvolution(kernelWidth: 3,
                                       kernelHeight: 3,
                                       inputFeatureChannels: 1024,
                                       outputFeatureChannels: 1024,
                                       neuronFilter: relu,
                                       device: device,
                                       kernelParamsBinaryName: "conv15" ,
                                       padding: false,
                                       strideXY: (1, 1))
        
        // last conv as a fully connected layer (1,1) kernel
        conv16 = SlimMPSCNNConvolution(kernelWidth: 1,
                                       kernelHeight: 1,
                                       inputFeatureChannels: 1024,
                                       outputFeatureChannels: 425,
                                       neuronFilter: nil,
                                       device: device,
                                       kernelParamsBinaryName: "conv16" ,
                                       padding: false,
                                       strideXY: (1, 1))
        
    }
}
