Review -- AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents
-------------------------------------------------------------------------------------------

*Note: This isn't a paper review for acceptance into some academic literature. Instead, I'm going to
attempt to write something akin to a book review, but for academic papers. I will also try to keep
things to 1000-2000 words.*

We need data to train neural networks, and we probably need a lot of data to train good robots.
I agree with the paper's authors up to that point.
After that our realities diverged.

The stated goal of the paper is to demonstrate a method to scale up robot deployments during data
collection. In this approach, semi-autonomous robots roam around and looks for things to do.
Vision-language models are used for scene and task understanding and large language
models are used to propose "diverse and novel" tasks. The tasks are then performed from some basic
scripting or via human teleoperation.

The abstract claims that 77,000 robot episodes were collection with over 20 robots running across
multiple buildings. It is later revealed that fewer than 20,000 of those were "successful" trials.

From figures 3 and 4 in the paper, it seems that there was a "break in" phase to the project that
lasted from March to July, at which point the project scaled from 5-7 robots to (briefly) 20. I'll
just look at the numbers from that point in since that should better capture the authors' intent of
a "large scale" data collection system.

The paper states that humans supervised between 3 and 5 robots and they roamed around their
environments. Let's assume that each person was supervising 5 robots and working 8 hour days. This
works out to 2.5 minutes/task collected. That translates into about 110 minutes per successful task.

The paper mentions, in passing, imitation learning and human demonstration.
Imitation learning is dismissed in the last paragraph of page six: "imitation learning methods
require near-optimal data." There is no citation for that, so I guess that they are hoping that the
reader will simply believe it. The assertion is nonsense.

Let's move on to human demonstration; this would be recording a human doing a task, e.g. opening a
drawer and pulling out a bag of chips, and then using that to train the robot. This method is
mentioned in the related work section, but the author's state that "teleoperated data can be far
more diverse and valuable for skill learning." Sure -- but at 110 minutes per successful task, this
isn't the way. Even if every task in their data collection were successful, a human can pull a bag
of chips from a drawer in far less than 2.5 minutes. A system that could transform human actions
into robotic commands would have been far more impactful than this work.

Speaking of impact, let's discuss the cost of this experiment. I'll assume that each robotic arm
cost around \$100k, or around \$2 million in total. Page 11 lists 10 Deepmind engineers as
contributing data. Let's assume that they all cost \$200k/year. At three quarters of a year, the
human cost of data collection was \$1.5 million.
With that much money, could we have made progress on a system that could transform human actions
into control signals for a robot? I think that we ("we" as in humanity) could have, and I believe
that the results would have greater and more far-reaching impact.

We can compare the approach in this paper to a human demonstration system, where we mount
cameras on top of common spaces (kitchen sink, coffee machine, tables, etc) to collect human
activity. A labeller can select those scenes where there is an activity and can draw a bounding box
around the region of interest. That dataset can be used to develop the human activity to robot
command system, perhaps with large language models. This is not an expensive thing to do, although
it is a hard problem to solve.

The most bothersome thing in the paper is the assertion that imitation learning doesn't work,
without even a citation provided. The full complement of robots wasn't even running the entire time,
which makes one wonder how many were taken out when they spilled cans of soda upon their fragile
robot bodies. This is not the way.
