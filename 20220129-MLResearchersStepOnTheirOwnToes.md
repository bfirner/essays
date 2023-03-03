Machine Learning Researchers Step on Their Own Toes!
----------------------------------------------------

Breaking news! Sometimes people make things harder than they need to be!

Okay, I'll admit that it's not that surprising.  Any field that heavily relies upon programming
encourages researchers to publish as quickly as possible, which inevitably leads to messy "grad
student" code. In machine learning, where many people look at things as black boxes, the
semi-magical alchemical recipes of papers past often dominate future techniques -- even when the
details were set by a sleep-deprived grad student.

That brings us to today's topic, initializing the weights of your model with the truncated normal
distribution.

You can read about the distribution itself on [wikipedia](https://en.wikipedia.org/wiki/Truncated_normal_distribution), but it isn't very complicated -- just a normal distribution where values that fall beyond an allowed minimum, maximum, or both are discarded and regenerated until they fall into the accepted range.

There is a good case to be made to do this. If you are using activation functions (ReLU, etc) then
there are some regions of those functions with no gradient. Things work best when you force values,
at least at the beginning of training, to be in the area of the function where a gradient exists,
meaning close to 0. At the same time though, you want your weight and bias values to be different.
Now you could just use a uniform distribution, but many things happen to work well with normal
distributions (at least we can justify them better with normal distributions) so they are
preferred. Hence the truncated normal distribution for weight initialization.

I was recently going through a paper, [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545), and
I wanted to play around with the architecture. The initialization used wasn't clear from the paper
so I hopped on over to [their official github](https://github.com/facebookresearch/ConvNeXt) to see what they were using and I found some calls to a function called "trunc\_normal\_" from a package named "timm". I went over to _that_ github page to check out [the implementation](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py).

There is some math and some comments and the code looks good. But I find myself wondering what is
gained from using it? The ConvNext model is initializing with a standard deviation of 0.02. The
default min and max range of the truncated normal being used are -2 and 2. That means that the
truncated normal is only different from a normal distribution after a random value 100 standard
deviations from the mean. Is that common?

You can read up on standard deviations on [wikipedia](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule), but I'll spoil the surpise: 100 standard deviation events are very rare. _Suuuuper_ rare. Five standard deviations is less than one in a million, seven standard deviations is in the hundreds of billions. Now, our deep learning models do have a lot of parameters, but the largest model in this paper is only in the hundreds of millions. It is _extremely_ unlikely that a normal distribution with mean 0 and standard deviation 0.02 will have any values below -2 or above 2.

We could easily remove some mysticism from the code by doing this to our modules instead:

~~~~ {.python}
nn.init.normal_(module.weight, std=0.02)
module.weight.clamp(min=-2, max=2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A bit less mysticism in deep learning would go a long way to dispel the black box mentality that
is pervasive in the field.
