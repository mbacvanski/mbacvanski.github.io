---
layout: small_title
title: A Few Things I Learn Every Day
mathjax: true
---

I try to learn or discover something new every day. Here's a log of all the things that I get excited about!

I usually update this every so often with a few days at once. Dates in `YYYY-MM-DD`.

### 2019-09-29

* [Algorithms](https://stackoverflow.com/questions/4325200/find-the-majority-element-in-array) for finding the majority element of a list in linear time
* [Academictree.org](https://academictree.org/) for family trees of people in academia
* [When to use](https://www.investopedia.com/terms/h/harmonicaverage.asp#:~:text=The%20Basics%20of%20a%20Harmonic%20Mean&text=Harmonic%20means%20are%20often%20used,weight%20to%20each%20data%20point.) the [harmonic mean](https://en.wikipedia.org/wiki/Average#Harmonic_mean)
* Pipeline [pigs](https://en.wikipedia.org/wiki/Pigging) 🐷

### 2019-09-28

* Darwinism suggests that human altruism may have evolved in a way such that groups of early humans who demonstrated more altruism, were more likely to survive in violent conflicts against other groups with less altruistic traits. [This study](https://www.researchgate.net/publication/26268380_Did_Warfare_Among_Ancestral_Hunter-Gatherers_Affect_the_Evolution_of_Human_Social_Behaviors) suggests that [violence](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0126589) among [early humans](https://www.nature.com/articles/nature16477) played a significant role in shaping the social dynamics of groups during the hunter-gatherer time period.
* [Everything We Eat Both Causes And Prevents Cancer](https://academic.oup.com/ajcn/article/97/1/127/4576988)

### 2019-09-27

* Purpleair has a [JSON API](https://www2.purpleair.com/community/faq#hc-access-the-json) 
* [Send SMS](https://github.com/typpo/textbelt) free, alternative to Twilio
* Detect news articles that have corporate sponsors with [this chrome extension](https://github.com/typpo/ad-detector)
* Wildfires have burned [**3.1 million acres**](https://www.fire.ca.gov/incidents/) in California so far this year. And fire season isn't even fully upon us.
* [Spin cleanup](https://sci-hub.se/10.1039/D0CP03745A) for quantum phase estimation
* 👉 [The Missing Semester Of Your CS Education](https://missing.csail.mit.edu/)

### 2019-09-26

* Github [refuses](https://talk.jekyllrb.com/t/katex-not-working-on-github-pages/4468/5) to use [KaTex](https://katex.org/) with Jekyll, switched over to just [mathjax](https://sgeos.github.io/github/jekyll/2016/08/21/adding_mathjax_to_a_jekyll_github_pages_blog.html).
* Installing mongodb on macOS Catalina with SIP means you have to [change the data directory](https://stackoverflow.com/a/58404057/2680053) by `mongod --dbpath=/Users/<your_username>/data/db`
* [Sigterm vs sigkill](https://major.io/2010/03/18/sigterm-vs-sigkill/) and C++ [signal handling](https://www.tutorialspoint.com/cplusplus/cpp_signal_handling.htm)

### 2019-09-25

* How to solve the [closest pair problem](https://en.wikipedia.org/wiki/Closest_pair_of_points_problem) in O(nlog(n)) time
* Cool [bit twiddling tricks](https://graphics.stanford.edu/~seander/bithacks.html) and [crazy bit manipulations](http://realtimecollisiondetection.net/blog/?p=78)
* Classes and inheritance in C++
* Use the `title` attribute instead of `#h1` in this file to make sure the title doesn't appear twice. [Weird things happen](https://github.com/mmistakes/minimal-mistakes/issues/1618#issuecomment-379807051) when you use the `#h1 ` thing.

### 2019-09-24

* How to use malloc in assembly
* Imposter syndrome (I might have it?)

### 2019-09-23

* How memory mapped IO works

### 2019-09-22

* How the [median of medians](https://en.wikipedia.org/wiki/Median_of_medians) algorithm works to produce the approximate median in $$O(n)$$ time
* How to [write a news article](https://www.theguardian.com/science/the-lay-scientist/2010/sep/24/1) about a scientific paper 🙃
* [Quagga](https://www.google.com/search?q=quagga)
* After you add an ssh key to Github, you need to use the ssh url to clone the repo in order to use your ssh credentials.

### 2019-09-21

* How to use partially observable markov decision processes ([POMDP](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process)) to efficiently search for items on a desk using a robotic manipulator, when some items obscure others. While solving for full solutions for a POMPD requires an $$O(n^2)$$ solution, using [Monte-Carlo sampling](https://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps.pdf) can cut this time down dramatically, enabling **[this paper](https://www.ccs.neu.edu/home/camato/publications/icra2019_1664.pdf)** to be able to plan movements of the robotic manipulator in real time. 
* In that paper, they use a POMDP by defining:
	* Set of world states (the objects on the table)
	* Set of actions (moving an object, moving the robot camera location/angle)
	* Set of observations (using R-CNN model to recognize bounding boxes for objects on the table)
	* State transition function, defining the probability distribution of new states after taking some action (moving an item) under the current state
	* Observation function, defining the probability distribution of possible observations (arrangements of obscured objects) given the resulting state and an action. Follows from the state transition function.
	* Reward function, giving a valued-reward for taking certain actions like finding the correct item.

### 2019-09-20

* How to play Among Us
* [Lilly impellers](https://www.treehugger.com/the-lily-impeller-nature-based-design-inspires-game-changing-efficiencies-4863361)

### 2019-09-19

* [Karatsuba's algorithm](https://en.wikipedia.org/wiki/Karatsuba_algorithm) for fast multiplication

### 2020-09-17
* Pointer arithmetic for array access in C++
* How to get Outlook to not delete event invitations after you accept them (it's in settings, under Settings => Calendar => Events and invitations => Delete invitations and responses...)

### 2020-09-16
* Using the stack and safe registers in assembly
* [The Line](https://granta.com/the-line/): a short story by Amor Towles, the author of one of my recent favorite books, *A Gentleman in Moscow*.

### 2020-09-15
* How to use GDB for debugging assembly programs
* You will segfault if you call `printf` on a format string with the wrong specifiers :(

### 2020-09-14
* The difference between function calls and labels in assembly
* SSH port forwarding from Virtualbox to host

### 2020-09-13

* Stoicism and Seneca (added to my reading list)
* The independent set problem and some applications to things like interval scheduling and bipartite matching
* What my would-be-roomates' dorm in Boston looks like (it's nice!)

### 2020-09-12
* How to play codenames
* [Silq](https://silq.ethz.ch/) is a cool and hip high level quantum programming language which has the unique feature of automatic uncomputation
* NORMALIZE PROGRAMMING WITH EXPLICIT PRECONDITIONS AND POSTCONDITIONS!!! 

### 2020-09-11
* What a [disjoint union](https://en.wikipedia.org/wiki/Disjoint_union) is
* The basics of [lambda calculus](https://personal.utdallas.edu/~gupta/courses/apl/lambda.pdf)
* How we can interpret [quantum circuits as higher order functions](https://www.mscs.dal.ca/~selinger/papers/qlambdabook.pdf) (!!!)
* How to [add LaTeX equations to this blog](https://github.com/mbacvanski/mbacvanski.github.io/commit/4aaa8bc833425b62fd3952aa31ae7791f799b3db) so that you can see the lovely next bullet
* One can think of quantum state teleportation as $$ g(f(\vert\phi\rangle)) = \vert\phi\rangle $$, and its inverse of teleporting classical bits, as $$ f(g(x, y)) = (x,y) $$ when $$ f:qbit \rightarrow bit\; \otimes\; bit $$, $$ g:bit\;\otimes\;bit \rightarrow qbit $$. $$f$$ is located at the sender, while $$g$$ is located at the receiver (in state teleportation). ___Beautiful.___

### 2020-09-10
* Checkpointing and restart versus Kubernetes's approach to [drain a node](https://kubernetes.io/docs/tasks/administer-cluster/safely-drain-node/)
* Mainframes still exist (!) and are still used today when huge amounts of memory needs to be accessed by huge amounts of processors, and for high throughput applications
* _"The computer is the tool, computation is the principle."_ (I kinda already knew this but this is a nice way of saying it)
* _“There are at least two kinds of games. One could be called finite, the other, infinite. A finite game is played for the purpose of winning, an infinite game for the purpose of continuing the play.”_ (Finite and Infinite Games, Ballantine, 1986, p. 1.)

### 2020-09-09
* Who the AI research faculty are at Northeastern, what are the labs, and what they work on
* Living in a red haze all day from the wildfire smoke really messes with how you perceive time. I felt exactly like I feel at hackathons, since the luminosity of my room did not change from morning to evening.

### 2020-09-08
* [Lambda](https://arxiv.org/pdf/cs/0404056.pdf) [calculus](https://epubs.siam.org/doi/pdf/10.1137/S0097539703432165) [for](https://www.mscs.dal.ca/~selinger/papers/qlambdabook.pdf) [quantum](https://link.springer.com/content/pdf/10.1007%2F978-3-319-89366-2_19.pdf) [computers](https://cs.stackexchange.com/questions/971/quantum-lambda-calculus) is insane and I need to learn more. Somehow I've managed to not hear about this before. It seems like an area orthogonal to the gate model and that is going to be super important once the hardware of quantum computing is ironed out.
* SecDB is a thing. More, I cannot really say

### 2020-09-07
* Dewey Square by Charlie Parker, on saxophone

### 2020-09-06 
* [Surprising effect](https://blog.janestreet.com/l2-regularization-and-batch-norm/) when you use both L2 Regularization and Batch Norm together

### 2020-09-05
* This [startup](https://replika.ai/) building an AI friend/therapist/romantic partner, and a [great podcast](https://www.youtube.com/watch?v=_AGPbvCDBCk) about it

### 2020-09-03
* How to set up a custom message to greet people on your SSH server

### 2020-09-02
* "Your assumptions are your windows on the world. Scrub them off every once in a while, or the light won't come in." - Isaac Asimov
* Foothill College ran a course on quantum computing. [Here is the book](https://www.fgamedia.org/faculty/loceff/cs_courses/cs_83a/Intro_to_QC_Vol_1_Loceff.pdf), and [here are the lectures](https://www.youtube.com/playlist?list=PLMnoxczUtKqWpKZTwpRBHrif_y-xENTfx).
* The [right packages to remove](https://askubuntu.com/questions/73219/remove-packages-to-transform-desktop-to-server#) to transform a desktop linux image into a headless server 

### 2020-09-01
* More of Dewey Square
* Up until 1820, China was the world's largest economy, accounting for almost 33% of the world's GDP at the time ([source](https://www.everycrsreport.com/reports/RL33534.html)).

### 2020-08-31
* The Raspberry Pi Zero is tinier than you expect and cuter than you expect
* How to play the first page of Dewey Square by Charlie Parker on the saxophone

### 2020-08-30
* How to use Go modules
* How to set up a compressed Docker image from a Go project
* I should look into [Tilt](https://tilt.dev/) for building Kubernetes apps
