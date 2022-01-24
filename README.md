# convolutional-neural-network
Work with CNNs as part of Advanced Honors Computer Science

### Observations on the network

I've tried out a few images of myself and some friends (with permission, of course!) in different trial runs of the convolutional neural network. What I've found is that head-on images, like `eli_w_mask.jpg`, work much better than images from the side, like `jasper_w_mask.jpg`. Scrolling through the dataset, this seems somewhat understandable: the images in the dataset are all very much zoomed in, head-on, and distorted, but not really to the side as seen in some of the sample images. (On the other hand, it makes me less worried about distorting images!) In implementation, being able to extract pictures of the face that are mainly head-on would be very useful.