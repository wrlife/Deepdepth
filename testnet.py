import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

caffe.set_mode_gpu()
net = caffe.Net('conv.prototxt',caffe.TEST)

[(k, v.data.shape) for k, v in net.blobs.items()]

[(k, v[0].data.shape, v[1].data.shape) for k, v in net.params.items()]

im = np.array(Image.open('/home/wrlife/bin/caffe/examples/images/cat_gray.jpg'))
im_input=im[np.newaxis,np.newaxis,:,:]
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...]=im_input

net.forward()

#net.save('mymodel.caffemodel')
