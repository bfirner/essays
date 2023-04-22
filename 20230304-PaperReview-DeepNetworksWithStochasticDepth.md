Review: Deep Networks with Stochastic Depth
-------------------------------------------

*Note: This isn't a paper review for acceptance into some academic literature. Instead, I'm going to
attempt to write something akin to a book review, but for academic papers. I will also try to keep
things to 1000-2000 words.*

_Deep Networks with Stochastic Depth_ is a paper from the European Conference on Computer Vision
(ECCV) from 2016. According to Google, that paper has been cited nearly 2,000 times, so it must be
good!

The paper presents a [regularization](20230303-WhatIsARegularizer.html) technique that makes it
possible to train deeper artificial neural networks. Four years after
[AlexNet](https://en.wikipedia.org/wiki/AlexNet) people were trying to figure out ways to cram more
parameters into their neural networks. Naive approaches suffered from vanishing gradient problems,
so techniques like batch normalization were being used to keep gradients sane. Regularizers, such as
dropout and loss terms like L2, were used to keep the crazy number of weights from turning into
overfit models of the training set, and the skip connections of residual networks were being used to
keep gradients flowing down ever deeper network architectures.

Enter *Stochastic Depth*, one of those crazy ideas that just might work. Or, as they describe it in
the related work section:

> Similar to Dropout, stochastic depth can be interpreted as training an ensemble of networks, but
with different depths, possibly achieving higher diversity among ensemble members than ensembling
those with the same depth. Different from Dropout, we make the network shorter instead of thinner,
and are motivated by a different problem. Anecdotally, Dropout loses effectiveness when used in
combination with Batch Normalization. Our own experiments with various Dropout rates (on CIFAR-10)
show that Dropout gives practically no improvement when used on 110-layer ResNets with Batch
Normalization.

Seven years later, that's an interesting assertion to see about Dropout. *Is* dropout worse with
batch normalization? Not that I've heard. This is the problem with how rapidly changing the field
has been for the past ten years--techniques for network architectures, parameter update algorithms,
data handling, and so on come and go so quickly so that was common knowledge one year is just
whispers of a bygone era the next.

Dropout is obviously more easily applied than stochastic depth. Stochastic depth can only be applied
when there are existing skip connections so that the DNN's features don't completely change if a
convolution is ignored. Recall that early skip connections were just identities, there were no
squeeze and excitation pathways or anything at all beyond a pooling operation if a downscale was
required. The output from the pathway through the convolution was simply added back to the identity
from the skip, so the kernel outputs were in some sense simply modulating the existing signal.
Dropping them with stochastic depth may feel strange, but it didn't alter the entire DNN.

Nowaday the skip connection in some networks is far more than just an identify though, which makes
the idea of stochastic depth seem odd. It is still in use however, and is mentioned in
[ConNeXt](https://arxiv.org/abs/2201.03545) as one of many techniques slammed together:

>  We use the
AdamW optimizer, data augmentation techniques such as Mixup, Cutmix, RandAugment, Random Erasing,
and regularization schemes including Stochastic Depth and Label Smoothing.

The ConvNeXt paper goes on to say:

> By itself, this enhanced training recipe increased the performance of the ResNet-50 model from 76.1% to 78.8%
(+2.7%), implying that a significant portion of the performance difference between traditional ConvNets and vision
Transformers may be due to the training techniques.

It's important to note that Figure 2 in that paper, which summarizes all of the improvements the
authors make with their large variety of techniques, only plots ImageNet Top1 accuracy from 78% to
82%. The stochastic depth paper shows a drop in test error on ImagNet of only 1% over a "basic"
152-layer ResNet with just their technique.

The stochastic depth paper showed much larger gains in smaller datasets. Indeed, the author's lament
their tiny boost. From the paper it is clear that they think a 1% increase is nothing great:

> In the words of an anonymous reviewer, the current generation
of models for ImageNet are still in a different regime from those of CIFAR. Although there seems to
be no immediate benefit from applying stochastic depth on this particular architecture, it is
possible that stochastic depth will lead to improvements on ImageNet with larger models, which the
community might soon be able to train as GPU capacities increase.

It is telling of how times have changed that the Stochastic Depth authors partially motivates their
work on the speedup during training that is gained when multiple layers are skipped and don't
celebrate a bump in 1% ImageNet accuracy. Nowadays, researchers will apparently do anything to
squeeze blood from a rock and get a tiny increase in accuracy.

I like the stochastic depth paper, it's a simple idea that works and is presented thoroughly in the
paper. I think that today I wouldn't bother looking at anything whose results were mostly based upon
CIFAR-10, but at the time that was okay. There is sufficient analysis done to understand what
stochastic depth is doing to gradients and network weights and you can easily try it out on your own
DNNs with minimal effort.
