Paper Review: Limits On Learning Machine Accuracy Imposed by Data Quality
-------------------------------------------------------------------------

This
[paper](https://proceedings.neurips.cc/paper/1994/hash/1e056d2b0ebd5c878c550da6ac5d3724-Abstract.html),
published in 1994 and written by Corinna Cortex, Larry Jackel, and Wan-Ping Chiang, is both a good
read and is still relevant for today's machine learning researchers. As was the case with many
papers of the time, it was written with a clear, direct style to inform colleagues of something
interesting and important. It lacks the overly-complicated equivocation of modern papers, which are
often written by torturing phrases until they lie still on the page to create an illusion of careful
planning and rational design. The paper only cites a mere 8 references, which makes it quite easy to
check the state of research at the time. It's a breath of fresh air compared to modern papers that
have a page or more of references lurking at the end.

The paper describes why learned models exhibit an upper bound on their performance as a function of
their data quality.  The data that they analyzed was from failures in a telecommunication system,
which was relevant to the AT&T network at the time.  It may not feel relevant to a machine learning
researcher in the 2020s, but the paper's methodology is insightful. I will endeavor to both explain
the paper and to show how it is relevant in the modern world of enormous datasets and billion
parameter models.

First, let us begin with the supposition that before you begin swinging your machine learning cudgel
around to smash every problem within reach you, as a rational being capable of complex behavior,
first take a few moments to estimate the upper bound of what is achievable with your statistical
model. For example, if you are attempting to predict the state of a fair coin after flipping it you
would be wise to only promise 50% accuracy from your model since a fair coin is random. Most
problems we try to solve are hopefully less random than that, but it is rare that we can expect 100%
success with any system.

First I'll discuss the contents of the paper. Afterwards, I will draw two examples from modern
machine learning, specifically ImageNet and autonomous vehicles, to illustrate how those ideas can
be applied.

Here is the key point of the paper, as stated at the end of section 1:

> We conjecture that the quality of the data imposes a limiting error rate on any
learning machine of ~25%, so that even with an unlimited amount of data, and an
arbitrarily complex learning machine, the performance for this task will not exceed
~75% correct. This conjecture is supported by experiments.

> The relatively high noise-level of the data, which carries over to a poor performance
of the trained classifier, is typical for many applications: the data collection was
not designed for the task at hand and proved inadequate for constructing high
performance classifiers.

It's a fact of life that all data has noise in it, although what is meant by the term "noise" can
vary from one dataset to another. We would like to believe that clever tricks and bigger models make
this problem go away, but this is, unfortunately, not the case. While we do have better techniques
to deal with mismatches between training and testing data than when this paper was written, we do
not have a method that can completely ignore the presence of noise in our datasets.

Before going farther though, I'm going to reproduce some of the figures from the paper so that we
can all share a common vocabulary for this discussion.

<!-- Uses extention implicit_figures -->
![Figure 1. Larger models decrease error until they learn the training set, but the testing error does not
always follow the same curve.](figures/error_vs_capacity.png)

First, let's talk about the match between training and testing data. Figure 1 shows an idealized
error as we increase our model capacity *if the training set data and test set data do not match*.
This is commonly called "overfitting." Overfitting *cannot happen* when the training and testing
data match. Now, getting training and testing to match completely is impossible, but for practical
purposes if the number of training examples is great enough then the model will not have enough
parameters to simple learn each of those examples individually and it must generalize.

Of course, in modern machine learning we employ gigantic models all of the time, with seemingly no
ill-effects. We can get away with this if we use strong regularizers. These techniques enforce
sparsity in the model, which resists fitting parameters that only apply to small number of examples
from the training set.

![Figure 2. As we increase the size of a dataset, we expect for it to become impossible for the
model to memorize it, so the training loss will increase. We also expect the testing error to
decrease, because it becomes increasingly likely that the training set contains examples that are
very close to the testing set.](figures/error_vs_datasize.png)

One thing that has not changed over time is machine learning's hunger for data. More data means that
we can increase our model size and increases the likelihood that our training data covers all of the
examples in our test set. This leads to a convergence of error between training and testing data, as
shown in Figure 2. Ideally we would increase our training set until all of our errors reach zero.
However...

![Figure 3. Even if we keep increasing our model capacity and dataset size, we may find that our
errors never reach zero. This happens if there is "noise" in our data. Some examples of noise are
incorrect labels and erroneous input data.](figures/error_vs_noise.png)

As illustrated in Figure 3, this is not realistic because our data is always flawed. That doesn't
mean that we can't estimate the error stemming from the quality of our data. Doing so is helpful.
If the errors that will come from the data are below the acceptable errors of the system then we can
ignore them. If not, then we need to find a way to improve our data quality.

Knowing the expected error threshold also helps us to see when there are no more gains to be made
with a dataset. This can save you from wasting a limitless amount of time trying to perform an
impossible task.

Let's jump into those modern examples, starting with ImageNet.

