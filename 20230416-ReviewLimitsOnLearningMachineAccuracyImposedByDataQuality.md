Paper Review: Limits On Learning Machine Accuracy Imposed by Data Quality
-------------------------------------------------------------------------

This
[https://proceedings.neurips.cc/paper/1994/hash/1e056d2b0ebd5c878c550da6ac5d3724-Abstract.html](paper),
published in 1994 and written by Corinna Cortex, Larry Jackel, and Wan-Ping Chiang, is both a good
read and is still relevant for today's machine learning researchers. As was the case with many
papers of the time, it was written with a clear, direct style to inform colleagues of something
interesting and important. It lacks the overly-complicated explanations of modern papers, which are
often written by torturing complicated words until they lie still on the page to create an illusion
of careful planning and rational design. The paper only cites a mere 8 references, makes it quite
easy to check the state of research at the time. It's a breath of fresh air compared to modern
papers with a page or more of references lurking at the end.

The paper describes a theoretical upper bound on the performance of a learned model that predicts
failures in a telecommunication system. It's a topic relevant to the AT&T network at the time but
that may not feel relevant to a machine learning researcher in the 2020s. However, the paper's
methodology is insightful, so I will endeavor to both explain the paper and to show how it is
relevant in the modern world of enormous datasets and billion parameter models.

First, let us begin with the supposition that before you begin swinging your machine learning cudgel
around to smash every problem within reach you first take a few moments to estimate the upper bound
of what is achievable with your statistical model. For example, if you are attempting to predict the
state of a fair coin after flipping it you would be wise to only promise 50% accuracy from your
model.  Most problems we try to solve are hopefully less random than that. I will draw two examples
from modern machine learning, specifically ImageNet and autonomous vehicles, after introducing the
topics of the paper itself.


