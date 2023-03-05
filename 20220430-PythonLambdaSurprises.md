Python's Unhappy Surprises
--------------------------

After working at an avionics company, where I used the Ada programming language daily, I decided
that I needed to experience more of the programming world so that I wouldn't back myself into a
corner with an unpopular language. I programmed in Lisp, learned me a Haskell, and played around
with Erlang. I learned about functional programming, closures, and the importance of avoiding
side-effects. I also learned the importance of readable error messages -- Ada error messages are
each documented to the extreme, whereas Haskell error messages read like equations
pulled from random computer science proofs.

At some point, I decided to look at an up and coming language named after an ambush predator that
slowly squeezes the life from its victims, stopping the heart before swallowing its prey whole:
Python.

The Internet was full of positive buzz about it after all. It was apparently super clean, not like
that messy Perl stuff. It also had all of the features of Ruby, but was a fun and inviting language
unlike those Ruby ecosystem which was overflowing with Ruby on Rails zealots. I took a quick look at
the language and it seemed fine. After seeing the use of capilization in Erlang, the use of
whitespace in Python didn't seem revolutionary, but people claimed that it made their code look neat
and clean. I was willing to give it a shot, so I started going through the examples of a popular
Python book to learn the language.

At first I was having a good time. The syntax was fairly clean and nothing was particularly worse
than in most other languages. I even recommended picking up Python to a random person I met on a
train when he mentioned that he wanted to learn some programming to help him with his work --
something dealing with chemistry or biology. Even though I have personally soured on the language, I
would still recommend Python to a non-programmer looking for a language to help automate some
things, at least under some circumstances.

As I started learning more about Python I started to see things that didn't impress me much. That
happens with any language, so it wasn't too surprising. The classes were a bit suspect --
they were obviously just dictionaries with some syntactic sugar on top, but there is nothing wrong
with keeping things simple. The Python itertools module didn't exist yet, so doing some things was
more awkward than I would have liked. Interfacing with C libraries wasn't easy like in Lua. Regex
was more ugly than in Perl. But at least the syntax was clean!

And then I hit the "lambda" function. Apparently you can only have a lambda function if it fits into
a single line. "What is the point of this," I thought? Was this included just because lambda
functions are cool? Is this in the language because functools.partial is too difficult to remember?
Wait, is there a difference between a function created with _lambda_ and one created with _partial_?
In a functional language there would be no difference between them since they would both return a
function. Python is not a functional language though, it is just pretending to be one of the
cool kids with a bit of broken syntax. C++ -- yes, C++, the super uncool language from like, a
million years ago -- does this better. Both std::bind and lambda functions in C++ are
FunctionObjects that capture variables in a scope. How do the Python equivalents capture scope? Do
they act as proper closures and capture scope at all?

Let's take a look at an example.

``` {.python .numberLines}
import functools

# A simple function to use with lambda and partial.
def add(b, c):
    return b + c

def main():
    # The variable that we want to capture.
    a = 2

    # Capture the variable 'a' in the lambda.
    lfunc = lambda b: add(b, a)
    # First, take note that if you would like to bind the 'b' argument instead of the 'c' argument
    # then you are in for a world of annoyance.
    # Putting that aside, this looks like it should be the same as the lambda.
    pfunc = functools.partial(add, c=a)

    # Will this change lfunc or pfunc?
    a = 5

    print("lfunc(10) is ", lfunc(10))
    print("pfunc(10) is ", pfunc(10))

main()
```

To know what numbers will be printed you have to understand whether lambda and partial are capturing
the value of the 'a' variable or a reference to that variable. If the value is captured then the
output would be 2+10. If a reference to the variable is captured then the value would be 5+10. Even if you have no clue about references and values you might reasonably expect that both functions would at least return the same value. Surpise! They do not. The output from the above code is:

<pre>
lfunc(10) is  15
pfunc(10) is  12
</pre>

Cool. It looks like the lambda is capturing 'a' by reference. This means that if the value that is
stored in the variable 'a' changes then the output of lfunc will also change. On the other hand,
pfunc, using functools.partial has captured the value. If you read the python docs it does mention
that the partial function returns something called a _partial object_. This just means that the
function creates an object that is callable, and has internal variables for the function, positional arguments, and keyword arguments. This process transfers the value of 'a' into a new variable, so the value is captured rather than a reference.

Now let us look into the documentation for _lambda_. I will just copy and paste it from the docs (for
python 3.10.1):

<blockquote>
An anonymous inline function consisting of a single expression which is evaluated when the function is called. The syntax to create a lambda function is lambda [parameters]: expression
</blockquote>

Super. So what is an expression?

<blockquote>
A piece of syntax which can be evaluated to some value. In other words, an expression is an accumulation of expression elements like literals, names, attribute access, operators or function calls which all return a value.
</blockquote>

Now here is something fun. Let's change the code slightly:

``` {.python .numberLines}
import functools

# A simple function to use with lambda and partial.
def add(b, c):
    return b + c

def main():
    # The lambda does not form a closure. Instead, it is an unevaluated expression.
    lfunc = lambda b: add(b, a)
    # The variable that we want to capture.
    a = 2

    # First, take note that if you would like to bind the 'b' argument instead of the 'c' argument
    # then you are in for a world of annoyance.
    # Putting that aside, this looks like it should be the same as the lambda.
    pfunc = functools.partial(add, c=a)

    # This changes lfunc, but not pfunc.
    a = 5

    print("lfunc(10) is ", lfunc(10))
    print("pfunc(10) is ", pfunc(10))

main()
```

Notice that the _lambda_ function seems to use 'a' before it exists. This is fine, because the
_lambda_ is actually an unevaluated expression. The above code has the same output as the first code
example. So how about this:

``` {.python .numberLines}
import functools

# A simple function to use with lambda and partial.
def add(b, c):
    return b + c

# What will the lambda do with this?
def thing(fun, a):
    a += 10
    print("fun(10) is ", fun(10))

def main():
    # The lambda does not form a closure. Instead, it is an unevaluated expression.
    lfunc = lambda b: add(b, a)
    # The variable that we want to capture.
    a = 2

    # First, take note that if you would like to bind the 'b' argument instead of the 'c' argument
    # then you are in for a world of annoyance.
    # Putting that aside, this looks like it should be the same as the lambda.
    pfunc = functools.partial(add, c=a)

    # This changes lfunc, but not pfunc.
    a = 5

    print("lfunc(10) is ", lfunc(10))
    print("pfunc(10) is ", pfunc(10))
    thing(lfunc, a)

main()
```

The output is:
<pre>
lfunc(10) is  15
pfunc(10) is  12
fun(10) is  15
</pre>

Wait a moment--somehow the lambda expression is using the value of 'a' from the main() function
even though there is a different 'a' within the _thing_ function. So the lambda actually *does* look
at its scope when searching for variables, even if it doesn't evaluate them at the time. My problem
with these different behaviors is that they are _surprising_, which is not a good feature in a
programming language.

The _principle of least surprise_ is the idea that things should behave in a way that is least
surprising. Inconsistencies are generally surprising, so I find the inconsistent behavior between
lambda and partial to be particularly insulting. More than just being inconsistent within the
language though, this _lambda_ is inconsistent with the rest of the programming world.

Speaking of the rest of the programming world, some core concepts in functional programming are
side-effect free functions and closures.  In a side-effect free world functions can only access
variables that are passed to them or created within the function. This makes those functions much
easier to test. A closure is the scope created when creating a new function, for example a _lambda_
function. In other languages with lambdas it is clear what variables are _captured_ when the lambda
functions are created. The fact that Python lambda functions are expressions that are not evaluated
until called makes then unnecessarily confusing.
