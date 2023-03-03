Perception Components - Introduction
------------------------------------

Probably due to my background doing my PhD in a wireless laboratory, I've always preferred to think
of neural networks as a series of components akin to what we see in a wireless radio. There are
high-pass filters, low-pass filters, matched filters, gain, and so on.

This way of thinking about neural networks also impacts how I would approach the difficult task of
teaching someone how to understand and use them. I wouldn't begin a freshman signal analysis course
with a discussion of how to optimize your tuning network before first introducing the components of
a radio. In the same vein, I will not begin an introduction of neural networks with the mathematics
of gradient descent[^1]; instead I will begin with a discussion of components found in neural networks,
what they do, and how they solve problems. Then when we proceed to outrageously large neural
networks with millions of parameters a student may have some home of understand what it is that all
of those numbers are getting up to.

Typical signal processing systems are concerned with decoding a signal from a noisy channel. Neural
networks are typical used to do a similar task, but are also responsible for *projecting* data from
an input space to an output space. Without getting too far ahead of ourselves, this means that
neural networks can take as input a matrix or vector of any dimensionality and yield any one of
those as an output. This diversity means that neural networks have different kinds of building
blocks than what we see in a signal processing course. Our exploration of these components and their
uses will begin with the lowest dimensional inputs. After we establish the functionality provided by
neural network components in each situation we will increase the dimensionality and complexity of
our examples.

[^1]: Stochastic gradient descent is of course important for neural networks, but it is not
  necessary. There may be other parameter update algorithms that yield the same or better results.

