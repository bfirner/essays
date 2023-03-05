What is a Regularizer? Dropout and Stochastic Depth
---------------------------------------------------

I was talking with someone about [this](https://arxiv.org/abs/2102.06171) paper on training neural
networks without normalization when I mentioned that the authors suggested that strong
regularization was required. They happened to be using dropout and [stochastic
depth](https://arxiv.org/abs/1603.09382), which is a silly sounding technique where the non-identity
layers in a residual network are randomly skipped during training.

My conversation partner reacted with outrage: how is that regularization? Regularization should mean
something, they insisted. After all, in linear algebra it generally means imposing a condition that
will cause spareness. Applying an L2 norm, where a loss is applied to non-zero weights will force
weights to zero in the absence of some other gradient, is a well-known regularization technique.

Wikipedia, wonder font of knowledge that it is, has a wishy-washy
[decription](https://en.wikipedia.org/wiki/Regularization_(mathematics)). It describes
regularization as "a process that changes the result answer [sic] to be 'simpler'." The page has
been edited several hundred times since 2005 and that's the best humanity has been able to do. This
isn't a terrible description, but it probably isn't clear how something like dropout serves that
purpose.

### Dropout ###

Let's talk about drop; it's in heavy use and what it does may be unclear to many of its users.

Imagine a stupid simple network with two inputs, A and B, and a single neuron that takes those two
inputs. The desired output is 'A', but, to make things interesting, 70% of the time input B will
have the same value as A.

Let's say that the weights begin like this:

Input  Weight
-----  ------
A      0.5
B      0.5

70% of the time those weights will yield the correct answer, but 30% of the time the output will be
too small. This will drive the weight of A to a higher value. The speed with which that happens is
dependent upon the learning algorithm used.

Let's say that at step 1 the inputs A is 5. B is also 5, so the output of the neuron is $0.5*5 +
0.5*5 = 5$, so the loss is 0.

At step 2 the inputs A is 5. This time B is different from A with a value of 10. The output of the neuron is $0.5*5 +
0.5*10 = 7.5$, so the loss is 2.5. *Both* weights will be decreased, even though ideally only the
weight for B would go down.

Herein lies the issue. The proper solution looks like this:

Input  Weight
-----  ------
A      1.0
B      0.0

However, unless we are already at that solution there will always be some error that pushes the
weight for input B away from 0. If that's the case, then the weight for A will be pushed away from 1.

Regularization solves this problem. If there is an L2 norm that penalizes the weights directly then
B's weight will go towards 0 faster than A's, since A is correlated with the correct output 100% of
the time. In the case of dropout, every time the B input is dropped the weight from A will move
closer to 1. This will push the input weight from B closer to 0 in the next round. If the input from
A is dropped then there will be some movement of the input weight from B, but if we haven't done
something foolish like using a very high learning rate, the input weight from B should eventually be
forced closer to 0. It seems messy, but in a very large network with redundant weights dropout seems
to work well.

Super.

Now let's discuss Stochastic Depth. In this paper the authors take a residual network. Each layer
consists of a set of transformations $f_l$ (in the paper they use convolution, batch norm, ReLU,
convolution, batch norm) and a skip layer that applies an identity (if $f_l$ does not downscale) or
a downscaling operation (such as pooling if $f_1$ does downscale the input). During training the
$f_l$ are removed at random.

How is that a regularizer?

I've left out an important detail. The probability of dropping a layer increases with the layer
number. The very first layer is never dropped, the next layer is drooped with some probability, the
next layer is dropped with increased probability, and so on. This gives the network the opportunity
to learn with all of the layers, but more important features will be pushed into the earlier layers
entirely if possible. Neural networks may normally split a function over multiple successive layers,
but this will force those functions into a more condensed version that fits into the earlier layers.

That creates sparsity, and thus regularization.
