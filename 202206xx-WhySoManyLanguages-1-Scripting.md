Why So Many Programming Languages, Part 1: Interpreted Languages
----------------------------------------------------------------

There are a lot of programming languages. Part of that is surely that humans haven't been
programming for very long and our first programming languages were too closely tied to particular
hardware or made poor mistakes that we've now moved beyond. Even if we restrict ourselves to just
looking at languages created or being actively updated within the last 20 years there are still more
than any particular person would ever need to learn. We don't need this many programming languages,
so why do we have them?

There is more than one reason why a language persists. It could be used in too many existing code
bases to get rid of even if it is reviled by its users (Fortran perhaps) or it could solve a
particular problem so well that it becomes the de-facto standard in some limited domain (shell
scripts for example). Some other languages are required for particular environments required by a
particular group -- programming for Android and iOS fits this category. And finally, some languages
just have inertia behind them. A generation of programmers learned C. Even if most of them created
terrible programs in the beginning, at this point most of that code has been tested, refined, and
has turned into the foundation of many other tools. Replacing it with something more modern would be
incredibly painful and expensive, but wouldn't improve anything for any end-users. Why bother?

* On the topic of replacing old code though, some code is not meant to last too long or to be used
  as the foundation of large projects. This is scripts written in interpreted languages.
* Some people say we use these to save time compiling, but that is silly.
* * Why, with just a couple of flags we can make compiling c++ code build all incremental .o and .d
    files, which vastly reduces compile time.

`
VPATH = src:include

CXX := g++
# The -MD and -MP flags generate dependencies for .o files.
CXXFLAGS := --std=c++20 -Wall -O3 -pedantic -Wextra -MD -MP -Iinclude

SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
DEPFILES := $(OBJECTS:.o=.d)

target: $(OBJECTS)
	g++ $(CXXFLAGS) $^ -o $@
`

However, using a scripting language means that you don't also need to know how to write this
Makefile. Compiled languages tend to have lots of tools written around them (such as Maven for Java
or Make or CMake for C and C++) whereas interpreted languages are ready to get up and go straight
away. Of course in a large enough project we end up with build systems for scripting languages as
well, but first few steps with the interpreted languages feel easier than with a compile language.

Of course there are even easier alternatives. There are a host of useful tools in \*nix and GNU
systems that have barriers to entry so low that many people use them without realizing it. Some,
like bash or awk, are full languages themselves although many people just use the most obvious
features. Other tools, such as sed, bc, or gnuplot, do one thing very well.  Tools like these can be
piped together, with outputs from one program being used as inputs to the next. As long as your task
can be broken down into a series of mostly independent steps these tools are most likely the fastest
and simplest solution available and you work on each step completely independently.

The down side of those task-focused command line utilities is that you must know that they exist and
must also know how to use each one of them. An alternative to them are the general purpose
interpreted languages, such as MATLAB, Perl, Python, Lua, Ruby, or Julia. These languages are
general purpose, meaning that they do not lack critical features required to write most kinds of
programs and support some kind of module system to add features to the language. They may still be
particularly suited to a more narrow range of applications, but their barrier to entry is low and
they can be finessed into working for almost any problem a programmer encounters. A programmer can
always make forward progress when using one of these languages, so they always feel that it is
useful. The combination of low barrier to entry and the feeling of progress is enticing, and many
people continue with their favorite scripting language long past the point when they should have
learned some alternatives.

These interpreted languages aren't just a crutch though. Some tasks can be easily decomposed into
simple subtasks and should be handled in typical \*nix style, with multiple steps piped together.
Once the task has more things to keep track of though, please, please, please, don't keep trying to
solve it in Bash. A nice interpreted language is quick to pick up and it will probably only take you
a few minutes to replace that Bash script you hacked together.

On the other hand, please don't keep trying to use your interpreted language to do complex and error
prone things when you should switch to something that is more formal. Debugging a 1000 line MATLAB
or Python program is bearable, but debugging a 10,000 line program with a weak typing system is a
special kind of torture.


