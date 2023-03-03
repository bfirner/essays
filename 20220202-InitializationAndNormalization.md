Initialization and Normalization in Neural Networks
---------------------------------------------------

If you experiment with neural networks it doesn't take much time to realize that normalization can
be critical for learning. Batch normalization, layer normalization of channels, local response
normalization: all of the prevalent neural network architectures have some kind of normalization.
You need to go back to very early networks, such as Lenet^[As a quick side note, Lenet became the
first neural network used in a commercial product when it was used to scan checks at ATM machines.]
from 1989, to find a neural network without explicit normalization layers. Even then, the creators
of Lenet used target values and initialization achieve similar effects.

From the 1989 paper "Backpropagation Applied to Handwritten Zip Code Recognition":

> The nonlinear function used at each node was a scaled hyperbolic tangent. Symmetric functions of
> that kind are believed to yield faster convergence, although the learning can be extremely slow if
> some weights are too small (LeCun 1987). The target values for the output units were chosen within
> the quasilinear range of the sigmoid. This prevents the weights from growing indefinitely and
> prevents the output units from operating in the flat spot of the sigmoid. The output cost function
> was the mean squared error.

This is nothing fancy. The output of a neural network can always be scaled, so if it is beneficial
to scale it to something more suitable for the activation functions then doing so can only help. For
example, if you want your neural network to output some very small or very large numbers, consider
scaling your output so that expected outputs fall into a range that the components of the network
can accommodate. This is similar to the reason why all inputs to a neural network should be
normalized. In fact, inputs to any layer should be normalized, or the weights of a layer should be
adjusted to match the inputs. That brings us to another point from the paper: weight initialization.

Again, from that same paper:

> Before training, the weights were initialized with range values using a uniform distribution
> between -24/$F_i$, and 24/$F_i$ where $F_i$ is the number of inputs (fan-in) of the unit to which
> the connection belongs. This technique tends to keep the total inputs within the operating range
> of the sigmoid.

Nowadays some other activation functions have generally replaced the sigmoid, but the important
concept to learn is that parameter initialization should be done to get the outputs of
neural network components into a range that produces a gradient with the chosen activation function
and architecture. Now, the equation for the sigmoid is $\sigma(x) = \frac{1}{1+e^{-x}}$. Technically
there is no point where a gradient ceases to exist, but practically speaking the gradient becomes
tiny as $x$ moves away from zero. In a world of real numbers this just slows down learning, but
since we are doing neural networks on computers with floating point math we should expect the
gradient to disappear when it becomes too small.

Instead of adjusting the initial weights we could adjust something else -- an abundance of options
is often the case in neural networks. In this case, we could overcome small gradients with an
increase in the initial learning rate (and large gradients by decreasing the learning rate). That
opens up two other problems. First, the learning rate must match all of the parameters in the
network. The outputs were previously normalized, but a very large or small learning rate may not
match. Second, we must always remember that floating point numbers in computer are a real constraint
to neural networks. Often times things that seem to be mathematically feasible are in fact
impossible with floating point numbers.

It is widely repeated that with a low enough learning rate a neural network will converge to a
correct answer, but the proof of that statement has a few conditions that are often overlooked. When
measuring the capacity of a learning model we can use a measure called [Vapnik-Cherveonenkis
dimension](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_dimension). When we measure a
neural network with floating point parameters against a theoretical network with real valued
parameters we see a large drop in the model dimension. It is this practical limitation that makes
regularization so critical for learning.

There is much more to say about the initialization of parameters, normalization of layer outputs,
matching learning parameters to the network, and VC dimensionality. Each of those topics deserves
its own dedicated discussion though, so we'll wrap this up with a quick summary of important points.
First, normalize inputs or normalize your initial weights to match them. Second, normalize outputs
or normalize weights in the last layer to match them. Third, control the gradients in your hidden
layers to enable learning. This means that output values should end up in areas of effective
learning on subsequent activation layers and values shouldn't be so large or small that issues with
floating point math manifest. The typical solution is to simply insert batch normalization layers
into your network, and that solution works -- just remember that other solutions are also possible.
