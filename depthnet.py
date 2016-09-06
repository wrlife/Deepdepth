# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:27:48 2016

@author: wrlife
"""
caffe_root = '/home/wrlife/bin/caffe/' 
import caffe,h5py

#caffe.set_device(0)
caffe.set_mode_gpu()

import numpy as np
from pylab import *

import tempfile

from caffe import layers as L
from caffe import params as P
from caffe.coord_map import crop

import os
weights = '/home/wrlife/projects/fcn/fcn.berkeleyvision.org/nyud-fcn32s-color/nyud-fcn32s-color-heavy.caffemodel'#caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
assert os.path.exists(weights)



NUM_STYLE_LABELS = 2

# Helper function for deprocessing preprocessed images, e.g., for display.
def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2

def conv_relu(bottom,nout, ks=3, stride=1, pad=1, group=1,
              param=learned_param,
              weight_filler=dict(type='xavier', std=0.01),
              bias_filler=dict(type='constant', value=0)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(data_layer_params,datalayer, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()
    if train:
        #n.data, n.label=L.HDF5Data(batch_size=30,source=datalayer,ntop=2,shuffle=True)
        n.data, n.label = L.Python(module = 'pascal_multilabel_datalayers', layer = datalayer, 
                                                  ntop = 2, param_str=str(data_layer_params))
    else:
        
   
        n.data = L.Python(module = 'pascal_multilabel_datalayers', layer = datalayer, 
                                               ntop = 1, param_str=str(data_layer_params))#n.data, n.label=L.HDF5Data(batch_size=1000,source=data,ntop=2)
        #n.label=None
    param = learned_param if learn_all else frozen_param
    
    
    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64,pad=53)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)  #112

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2) # 56

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)  #28

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)  #14

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_3) # 7
    
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    

#    if train:        
#        n.drop6 = fc7input = L.Dropout(n.relu6_m,dropout_ratio=0.5,in_place=True)
#    else:
#        fc7input = n.relu6
#        
#        
#    n.fc7_m, n.relu7_m = conv_relu(fc7input, 1, 4096, pad=0, group=1, param=learned_param)
#    
#
#    if train:        
#        n.drop7 = Deconvinput5 = L.Dropout(n.relu7_m,dropout_ratio=0.5,in_place=True)
#    else:
#        Deconvinput5 = n.relu7_m
        
    #n.fc6, n.relu6 = fc_relu(fc6input, 4096, param=param) 
    
    #if train:        
    #    n.drop6 = Deconv5input = L.Dropout(n.relu6,dropout_ratio=0.2,in_place=True)
    #else:
    #    Deconv5input = n.relu6
        
    #n.conv7_1, n.relu7_1 = conv_relu(fc6input, 1, 21, pad=0, group=1, param=param)    

    n.deconv5=L.Deconvolution(n.drop7,
                              convolution_param=dict(kernel_size=7,stride=7,num_output=512,pad=0,group=2,
                                                     weight_filler=dict(type='gaussian', std=0.01),
                                                     #bias_term=False),param = [dict(lr_mult=1, decay_mult=0.00000005)])
                                                     bias_filler=dict(type='constant', value=0)),
                              param=learned_param)   # 7
    
    n.fused_pool5=L.Eltwise(n.deconv5,n.pool5)
    
    n.deconv4=L.Deconvolution(n.fused_pool5,
                              convolution_param=dict(kernel_size=2,stride=2,num_output=512,pad_w=0,pad_h=0,group=2,
                                                     weight_filler=dict(type='gaussian', std=0.01),
                                                     #bias_term=False),param = [dict(lr_mult=1, decay_mult=0.00000005)])
                                                     bias_filler=dict(type='constant', value=0)),
                              param=learned_param)    # 14                        
        
    
    n.fused_pool4=L.Eltwise(n.deconv4,n.pool4)
    
    n.deconv3=L.Deconvolution(n.fused_pool4,
                              convolution_param=dict(kernel_size=2,stride=2,num_output=256,pad_w=0,pad_h=0,group=2,
                                                     weight_filler=dict(type='gaussian', std=0.01),
                                                     #bias_term=False),param = [dict(lr_mult=1, decay_mult=0.00000005)])
                                                     bias_filler=dict(type='constant', value=0)),
                              param=learned_param)  # 28
    
    n.fused_pool3=L.Eltwise(n.deconv3,n.pool3)
    
    n.deconv2=L.Deconvolution(n.fused_pool3,
                              convolution_param=dict(kernel_size=2,stride=2,num_output=128,pad_w=0,pad_h=0,group=1,
                                                     weight_filler=dict(type='gaussian', std=0.01),
                                                     #bias_term=False),param = [dict(lr_mult=1, decay_mult=0.00000005)])
                                                     bias_filler=dict(type='constant', value=0)),
                              param=learned_param) 
    
    n.fused_pool2=L.Eltwise(n.deconv2,n.pool2) # 56
    
    n.deconv1=L.Deconvolution(n.fused_pool2,
                              convolution_param=dict(kernel_size=2,stride=2,num_output=64,pad_w=0,pad_h=0,group=1,
                                                     weight_filler=dict(type='gaussian', std=0.01),
                                                     #bias_term=False),param = [dict(lr_mult=1, decay_mult=0.00000005)])
                                                     bias_filler=dict(type='constant', value=0)),
                              param=learned_param) 
                              
    n.fused_pool1=L.Eltwise(n.deconv1,n.pool1)     #112

    n.deconv1=L.Deconvolution(n.fused_pool1,
                              convolution_param=dict(kernel_size=2,stride=2,num_output=1,pad_w=0,pad_h=0,group=1,
                                                     weight_filler=dict(type='gaussian', std=0.01),
                                                     #bias_term=False),param = [dict(lr_mult=1, decay_mult=0.00000005)])
                                                     bias_filler=dict(type='constant', value=0)),
                              param=learned_param)           
    n.m_score = L.Crop(n.deconv1, n.data,crop_param=dict(axis=2,offset=53))
    #n.fused_pool0=L.Eltwise(n.deconv1,n.conv1_2) # 227    
    
    #n.conv0, n.relu0 = conv_relu(n.m_score, 1)


    #if not train:
        #n.probs = L.Power(n.relu5)
    if train:
        #n.loss = L.EuclideanLoss(n.deconv1, n.label,loss_weight=0.5)
        n.loss=L.Python(n.m_score, n.label,module='myLoss',layer='Gradient_Apperance_Loss',loss_weight=0.5)        
        n.acc = L.Accuracy(n.m_score, n.label)
    # write the net to a temporary file and return its filename
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(n.to_proto()))
        return f.name
        

def style_net(train=True, learn_all=False, subset=None):

    if train:
        split='train'
    else:
        split='test'
        
#    if train:
#        h5data='nyud_train_h5_list.txt'
#    else:
#        h5data='nyud_test_h5_list.txt'
    pascal_root='./'
    data_layer_params = dict(batch_size = 30, im_shape = [120, 120],patch_ratio_w=6,patch_ratio_h=2, split = split, pascal_root = pascal_root,case = train)
    
    return caffenet(data_layer_params,'PascalMultilabelDataLayerSync', train=train,
    #return caffenet(data_layer_params,h5data, train=train,
                    classifier_name='fc8_flickr',
                    learn_all=learn_all)


def disp_preds(net, image, labels, k=5, name='ImageNet'):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    top_k = (-probs).argsort()[:k]
    #print 'top %d predicted %s labels =' % (k, name)
    #print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
    #                for i, p in enumerate(top_k))
    return top_k


def disp_imagenet_preds(net, image):
    disp_preds(net, image, imagenet_labels, name='ImageNet')

def disp_style_preds(net, image):
    disp_preds(net, image, style_labels, name='style')
    
#untrained_style_net = caffe.Net(style_net(train=False, subset='train'),
#                                weights, caffe.TEST)
#
#untrained_style_net.forward()
#style_data_batch = untrained_style_net.blobs['data'].data.copy()
#
#batch_index = 40
#image = style_data_batch[batch_index]
#plt.imshow(deprocess_net_image(image))
