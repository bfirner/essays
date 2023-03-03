Perception Components - Perceptrons
-----------------------------------

First, some terminology. A perceptron with a single input takes that value, *x*, and transforms it
to an output, *y*, by multiplying by a weight, *w*, and adding a bias, *b*. On its own this not
likely to set your heart racing so we will move on quickly.

$y = wx + b$

By itself, a single perception will only perform the most basic of transformations upon its input.
The most basic neural network task is to do classification of an input. Therefore these perceptrons
must be good at creating *indicator functions*, functions that map their inputs to either 0 or 1.
How do they do that?

First, perceptrons generally accept many inputs, performing the operation across all *n* inputs:

$y = b + \prod_{i=0}^{n} \biggl( w_{i}x_{i} \biggr)$

Notice that there is only one bias value. We could write this as:

$y = \prod_{i=0}^{n} \biggl( b_{i} +  w_{i}x_{i} \biggr)$

but the bias values will be summed to a single value anyway.

So how does this linear transformation do something difficult like mapping any arbitrary input onto
the values 0 and 1? With help. First, difficult transformations are done with multiple layers of
perceptrons, meaning that the outputs of earlier layers become the inputs of later layers. Second,
we will add non-linear layers in between our perceptrons. The choice on nonlinearity seems to be,
unfortunately, somewhat arbitrary and often impacted by personal preference and anecdote. For our
purposes, we will introduce two nonlinearity here.

First, the sigmoid function.

Next, the rectified linear unit (ReLU) function.
