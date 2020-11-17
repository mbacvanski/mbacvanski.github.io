---
layout: small_title
title: A Few Things I Learn Every Day
mathjax: true
---

I try to learn or discover something new every day. Here are some of the things that I get excited about!

I usually update this every so often with a few days at once. Dates in `YYYY-MM-DD`.

### 2020-11-16

* [$O(n\log{n})$ multiplication](https://mattermodeling.stackexchange.com/a/1356) and its impact (not too much)
* `/bin/time -v` gives extra info like memory usage, stack sizes, sockets used, etc. Somehow this is different than `time` even though `which time` shows it to be the same 🤔
* [Nuclear Weapons Journal, Issue 2 of 2005.](https://www.lanl.gov/science/weapons_journal/wj_pubs/11nwj2-05.pdf) This is something 12-year old me would have loved.
* How to write a super fast, multithreaded memory allocator using buckets and multiple arenas. Blog post coming soon.

### 2020-11-15

* *"Communism will never work out. There are too many red flags."*
* `#define NDEBUG` to disable asserts in C. Made my own memory allocator almost 40% faster because I had all kinds of asserts sprinkled in there.
* Sometimes you'll need to use `sudo apt-get dist-upgrade` to get `linux-headers` and other common tools.

### 2020-11-14

* *"Infinite boiling softens the stone"*
* How to construct the bebop scale
  * For major scales, add the #5
  * For minor scales, add the major 3rd
  * For dominant 7 scales, add the major 7th

### 2020-11-13

* [This page](https://technerium.ru/en/yadernyi-poezd-prizrak-boevoi-zheleznodorozhnyi-raketnyi-kompleks/) about the train-launched [RT-23 Molodets ICBM](https://en.wikipedia.org/wiki/RT-23_Molodets). In order to fit in a train cart, this rocket featured an inflatable nose cone. Disguised as refrigerator carts, the trains carrying these ICBMS featured a number of innovations to deflect power wires, eject the rocket without launching, and support the immense weight across multiple carts.
* [Weapons and Warfare blog](https://weaponsandwarfare.com/)

### 2020-11-12

* If you try to `mmap` a fractional size of a page, it returns the next largest size possible with whole numbers of pages.
* How do you write an online algorithm to detect a beat drop in music playing in real time?
* Always assert bounds for array access in C!

### 2020-11-11

* Static arrays in C must have a size defined not as a variable, like as `#define size 5`. Apparently constants are variables.
* Are trees aliens?
* Initializing a `pthread_mutex_t` using the macro initializer is fine for static initializations, but if you are doing it afterwards like in a constructor you'll want to use `pthread_mutex_init(&mut, NULL)` [or else](https://sourceware.org/legacy-ml/libc-help/2008-05/msg00072.html) you'll get an error `__pthread_tpp_change_priority: Assertion 'new_prio == -1 || (new_prio >= __sched_fifo_min_prio && new_prio <= __sched_fifo_max_prio)' failed`

### 2020-11-10

* [Datawrapper](https://www.datawrapper.de/) for quickly visualizing data
* [Perry Bible Fellowship](https://pbfcomics.com/)
* Rough idea how [jemalloc](https://people.freebsd.org/~jasone/jemalloc/bsdcan2006/jemalloc.pdf) works, using the idea of multiple arenas of memory allocation that threads are assigned to

### 2020-11-09

* How different waveforms [sound](https://www.perfectcircuit.com/signal/difference-between-waveforms)

* [Self-organizing lists](https://en.wikipedia.org/wiki/Self-organizing_list)

* [Online algorithms](https://en.wikipedia.org/wiki/Online_algorithm) are able to provide useful results even with partial data, and results are updated efficiently as more data is available. 

* Imagine a doubly ordered, doubly linked list that has two orderings: ordering `a`, which `prev_a` and `next_a` define, and ordering `b`, which is kept track of with `prev_b` and `next_b`. You could use this in keeping track of a free list in a memory allocator, where one ordering is allocations by memory address and the other is by size. This would enable constant time freeing with blocks ordered by memory address, while still letting you use a first-fit allocation strategy on allocating the smallest previously free block on new allocation calls. What is the name for this type of data structure?

  ```c
  typedef struct item{
  	struct item* prev_a;
  	struct item* next_a;
    
  	struct item* prev_b;
  	struct item* next_b;
  	
  	void* data;
  }
  ```

* Initialize a `pthread_mutex_t` with `static pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;`

### 2020-11-08

* The size of a struct in C isn't the sum of the sizes of each of its fields; padding is automatically added to [align the struct fields](https://stackoverflow.com/a/119134/2680053).

### 2020-11-07

* `always@(...)` in Verilog describes events that should happen under certain conditions. Two common use cases are to have a sensitivity list inside the parenthesis, or to trigger on every clock cycle. Good reference is [here](https://class.ece.uw.edu/371/peckol/doc/Always@.pdf).
  * `always@(A, B, C)` indicates that the assignments inside the block change whenever any of the parameters in the sensitivity list changes – any time A, B, or C changes.
  * `always@(posedge clock)` updates assignments inside the block at the positive (rising) edge of every clock cycle.
* Amortized time analysis

### 2020-11-06

* It's pretty difficult to divide a line into sevenths without practicing. Is there something fundamental about 7 that makes dividing into 7 pieces difficult? Are there aliens for whom it is easy to divide things into 7 pieces, just like humans find it easy to divide things in halves?
* [Quantum Computing & Quantum Mechanics courses](/blog/qc-qm-courses)

### 2020-11-05

* Pointer arithmetic in C works differently on `void*` and `struct*` or any other pointer type: when you do `+1` on a void pointer, that increments the pointer's address in memory by 1 byte, whereas if you do `+1` on an `int*`, `char*`, or `struct*` that increments the pointer by one whole unit, whether that's an integer, character, or struct.

### 2020-11-04

* *"We are all in the gutter, but some of us are looking at the stars"* — Oscar Wilde
* [How](https://superuser.com/a/294736) a computer reboots itself

### 2020-11-03

* After committing a super large file in Git, and now you have regrets because it won't fit in Github and you can't push it, and now you just want to forget everything you've ever had to do with it: (you may have to put the `-f` flag after `--index-filter`)

  ```bash
  git filter-branch --prune-empty -d /dev/shm/scratch \
    --index-filter "git rm --cached -f --ignore-unmatch path/to/bigfile.csv" \
    --tag-name-filter cat -- --all
  ```

### 2020-11-02

* *"A man who lives fully is prepared to die at any time."* — Mark Twain

* [COVID detection using AI](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9208795) just from cough recordings: looks like a 97.1% accuracy with 100% true negative rate, 16.8% false positive rate.
* Enable AptX & AAC codec on mac: (may need to disconnect & reconnect headphone afterwards)

  ```bash
  sudo defaults write bluetoothaudiod "Enable AptX codec" -bool true
  sudo defaults write bluetoothaudiod "Enable AAC codec" -bool true
  ```

* D-type flip flops implemented using registers in Verilog
* [Akku](https://github.com/jariz/akku) to monitor bluetooth headphone battery level from mac
* [Spinlocks](https://en.wikipedia.org/wiki/Spinlock) are useful when you only expect to wait a very short amount of time, because they prevent your thread from being rescheduled. But they are wasteful if you hold them any longer because they're literally just busy waiting. Also [ticket locks](https://en.wikipedia.org/wiki/Ticket_lock) and [seqlocks](https://en.wikipedia.org/wiki/Seqlock) are pretty cool.
* Up until 2019, in Texas it was illegal to buy an [Erlenmeyer flask](https://en.wikipedia.org/wiki/Erlenmeyer_flask) without a permit, in an attempt to prevent illicit drug sales.
* It's not too hard to build [your own Schlieren imaging setup](https://www.instructables.com/Schlieren-Imaging-How-to-see-air-flow/)!

### 2020-11-01

* Daylight savings in the *fall* is the better one because you get to sleep an extra hour
* Why my mac would refuse to play music through bluetooth audio. Went into Bluetooth Explorer and clicked Tools ➡ Audio Options ➡ Reset audio default.

### 2020-10-31

* Refresher on how to use passport.js's serializeUser, deserializeUser, and local authentication. It's been a while since I've worked with this!
* The [off-price](https://en.wikipedia.org/wiki/Off-price) retail model (TJX is known for)
* [Shibboleth](https://www.google.com/search?q=shibboleth) is a cool word

### 2020-10-30

* Minimum spanning trees and some interesting properties of graph cuts
* Multi-bit wires in verilog
* The story of [Hiroo Onada](https://en.wikipedia.org/wiki/Hiroo_Onoda), Japanese soldier tasked with defending to the death an island in the Phillippines against US invasion during WWII. After all his comrades were killed, Onada spent 29 years hiding in the mountains and occasionally murdering civilians and burning crops – nobody ever told him that the war was over, and his last orders were to never surrender or take his own life. There's a semi-happy ending to this story too.

### 2020-10-29

* How to write a barrier in C using mutexes and condition variables
* If you can't think of anything to say, it probably means you aren't listening closely enough. I've found that if I really think about what the other person is saying, there's always a question I can ask or something related that I can share that moves the conversation forward. When one gets caught up with thinking "what should I say next" it's easy to fall into the habit of just waiting for one's turn to speak, rather than engaging in true conversation. 

### 2020-10-28

* How the internet [looked](https://en.wikipedia.org/wiki/BBN_Technologies#/media/File:InetCirca85.jpg) in 1985
* [Non-demolition quantum measurement](https://en.wikipedia.org/wiki/Quantum_nondemolition_measurement) is in contrast to the usual projective measurement of a system that leaves it in the measured eigenstate. How can this be done, and what is the state that the system takes on after measurement?

### 2020-10-27

* [`Promise.any`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise/any) is in approval! And about time too. Crazy to think that [I asked about it 4 years ago](https://stackoverflow.com/q/39940152/2680053) in 2016 on StackOverflow – the [top voted answer](https://stackoverflow.com/a/39941616/2680053) right now has been a pretty neat hack.
* You shouldn't use `malloc` when creating objects in C++, since `new` takes care of that for you. But `malloc` might still be useful when paired with `realloc`, like for building a dynamically allocated vector, since you can't do that with `new`.
* Is general intelligence equivalent to lossless compression? This is a hot take I've heard floating around.
* The [Efficient Market Hypothesis](https://www.investopedia.com/terms/e/efficientmarkethypothesis.asp) hypothesizes that stock prices always reflect all knowable data about a company, with the conclusion that there are no such things as undervalued or overvalued stocks. If you believe in this you probably believe it's impossible to beat the market by picking stocks.

### 2020-10-26

* How the Bellman-Ford algorithm works

### 2020-10-25

* [**propreantepenultimate**](https://www.google.com/search?q=propreantepenultimate)
* Finally, [film simulation recipes](https://fujixweekly.com/2020/10/25/film-simulation-recipe-cards/) for Fujifilm! Recipes for Kodachrome, Ilford HP5, Fujifilm Acros, and more.

### 2020-10-24

* [`kebab-case`](https://stackoverflow.com/a/17820138/2680053)
* [Procedurally generated Chinese landscapes](https://github.com/LingDong-/shan-shui-inf)
* `#pragma once` instead of include guards! Why did no one tell me about this earlier. Although it's not universally available in all compilers.

### 2020-10-23

* Currency arbitrage is the name for that thing
* [Great video](https://www.youtube.com/watch?v=WCUNPb-5EYI) on RNNs!
* [Bongo Cat](https://bongo.cat/)
* Install MacOS in Virtualbox with [this incredible shell script](https://github.com/myspaghetti/macos-virtualbox)

### 2020-10-22

* [Negative edge weights](https://cs.stackexchange.com/questions/14248/what-is-the-significance-of-negative-weight-edges-in-a-graph) on graphs could be useful in modeling the free energy of chemical reactions or cost/profit of driving a taxi. Negative edge weights open up the possibility of having negative cycles, where following a cycle ends up with a sum negative weight – this doesn't really make sense, and the [Bellman-Ford](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm) algorithm detects these.

### 2020-10-21

* `systemctl disable influxdb` to prevent a service from starting automatically on boot, effective next stop event (reboot).
* Controversy over the [Mpemba effect](https://en.wikipedia.org/wiki/Mpemba_effect), where hot water freezes faster than cold water (sometimes????)
* [Sleep Sort](https://www.geeksforgeeks.org/sleep-sort-king-laziness-sorting-sleeping/): sort a list by literally doing nothing
* Build nice command line UIs with Go using [bubbletea](https://github.com/charmbracelet/bubbletea)

### 2020-10-20

* Djikstra's algorithm
* Basics of Verilog
* [`ctrl-z`](https://superuser.com/questions/476873/what-is-effect-of-ctrl-z-on-a-unix-linux-application) to suspend a process, and `bg` and `fg` to manage processes. [Here](https://en.wikipedia.org/wiki/Job_control_(Unix)#Implementation) is how Unix implements them with a job table.
* The `c` shell, `csh`, [feels more like normal programming](https://en.wikipedia.org/wiki/C_shell#More_like_C) than bash

### 2020-10-19

* [Saudade](https://www.google.com/search?q=saudade) and [Solastalgia](https://pubmed.ncbi.nlm.nih.gov/18027145/)

### 2020-10-18

* NVIDIA [video codec](https://developer.nvidia.com/nvidia-video-codec-sdk) using AI
* Data lakes in organizations easily suffer from data quality issues when the applications dumping data are not responsible for the quality of their data, the team managing and using the data lake is unable to resolve data quality issues originating upstream.

### 2020-10-17

* Algorithms for finding strongly connected components of graphs in linear time
* Applying breadth-first search to bipartite coloring
* [bat](https://github.com/sharkdp/bat) is a cat clone that does syntax highlighting and looks nice
* Fuzzy find files with [fzf](https://github.com/junegunn/fzf)
* [Exa](https://github.com/ogham/exa) is a cool replacement for `ls`, written in Rust

### 2020-10-16

* Topological sort of a graph in linear time, using discovery and finish times of depth-first search

### 2020-10-15

* On Unix, pipes run processes concurrently: when you run `echo hi | tail`, the shell uses forking to run both `echo hi` and `tail` at the same time, and the `stdout` file descriptor of the `echo` is the same as the file descriptor for the `stdin` of `tail`. The two ends of the pipe are in different processes, and link the `stdout` of the left hand process to the `stdin` of the right hand process. This example behaves the same as if I did `tail <(echo hi)`, but this would first run `echo hi` and then pass that output to `tail`.
  * This is also the reason why when you do `ps -ef | grep cron`, the `grep` command you are running shows up. It would not show up in the output of `ps` if `grep` were running after `ps` completed. But because we are using piping, the `grep` process is started at the same time, and doing its work on the `stdout` of `ps`, which of course will include `grep`. 
* Put `set follow-fork-mode child` and `set detach-on-fork off` in the gdb window in order to debug forked processes inside the CLion debugger
  * Note: this doesn't really work very well but I don't have the time to figure out why. When a child process exits basically the whole program exits 😟

### 2020-10-14

* [Minterms](https://en.wikipedia.org/wiki/Canonical_normal_form)
* Somehow disabling 3d acceleration in VirtualBox for a Ubuntu 20.04 vm fixes problems with the UI being incredibly laggy
* [How to use `execlp`](https://stackoverflow.com/a/21559499/2680053), whose man page documentation is incredibly vague

### 2020-10-13

* The inventor of the theremin, Leo Theremin, also happened to invent [a bugging device](https://en.wikipedia.org/wiki/The_Thing_(listening_device)) concealed within a gift given by the Soviet Union to the US Ambassador's house in Moscow. 
* [Clara Rockmore](spotify:artist:5WKWtfP2aDQAOwAvhxquPR), who plays the theremin
* From September 19 until today, I have been writing dates on this page as 2019 instead of 2020 🙃 Thanks to Kirpal who told me about this!

### 2020-10-12

* About one third of press releases are exaggerated, and these get magnified in news stories. Exact results [here](https://www.bmj.com/content/349/bmj.g7015), go take a look.

* Uninstall all adobe junk applications and processes with [CC Cleaner](https://helpx.adobe.com/photoshop-elements/kb/elements-installation-error-CC-cleaner-tool.html)

* [Mask off](https://www.youtube.com/watch?v=NudlLKd3WW4) 

* [News embargoes](https://www.scientificamerican.com/article/how-the-fda-manipulates-the-media/) mean that science reporters can only report in the ways that  publishers like the FDA allow. Embargoed news has been around since the 1920s, when it meant that science journals would offer reporters early access to papers and contact information of authors, under the condition that the reporters can only publish their story after the embargo time period expires. This would enable reporters extra time to research their story, without the fear that their story would get published by someone else first.
  
  The **close-hold embargo** forbids reporters from contacting outside sources for commentary about news. This essentially prevents all independent reporting, allowing the journal or organization to dictate what the news says. The FDA has a history of doing this.

### 2020-10-11

* Cookies are passé. Browser fingerprinting is à la mode. Check if your browser is leaking your fingerprint at [Panopticlick](https://panopticlick.eff.org/), a project by the EFF.
* I should read [Gödel, Escher, Bach](https://en.wikipedia.org/wiki/G%C3%B6del,_Escher,_Bach)

### 2020-10-10

* `mongodump` and `mongorestore` have a `--gzip` flag to zip dumped `bson` documents
* The [story](https://pitchfork.com/thepitch/how-a-long-lost-indian-disco-record-won-over-crate-diggers-and-cracked-the-youtube-algorithm/) behind [Aaj Shanibar](https://www.youtube.com/watch?v=HRIbkvzDSpo)
* [Glyptodonts](https://en.wikipedia.org/wiki/Glyptodont) used to roam prehistoric Earth until probably the end of the ice age
* *“The object of art is not to reproduce reality, but to create a reality of the same intensity.”* ーAlberto Giacometti

### 2020-10-09

* [Difference](https://askubuntu.com/questions/445384/what-is-the-difference-between-apt-and-apt-get) between `apt` and `apt-get`
* How to [resize a virtualbox disk](https://askubuntu.com/a/558215/546794) image using `gparted`
* What "[DAO](https://en.wikipedia.org/wiki/Data_access_object)" means (somehow I've never seen someone call it that before)
* [Saol](https://www.schick-toikka.com/saol-standard) is an awesome font

### 2020-10-08

* How to move from one key to another using common chords

### 2020-10-07

* How to play tones through a speaker connected to a GPIO pin.
  * How would you play music, chords, or multiple tones at once?

### 2020-10-06

* [Neat-URL](https://github.com/Smile4ever/Neat-URL) removes trackers from your URLs
* [Boop](https://github.com/IvanMathy/Boop) to easily do text things! Lots of useful and fun shortcuts. Hard to describe but pretty awesome.
* [QuickJS](https://bellard.org/quickjs/) is a JavaScript runtime that starts much faster than the v8 engine used in Node.js. Could be useful for serverlessor CLI apps that need to start and stop very quickly. [Elsa](https://github.com/elsaland/elsa) is a QuickJS wrapper written in Go.
* [DUF](https://github.com/muesli/duf) (Disk Usage Free) utility
* Herbie Hancock gave [lectures](https://www.youtube.com/watch?v=xSFMkJQKigk) at Harvard!

### 2020-10-05

* C [order of operations](http://unixwiz.net/techtips/reading-cdecl.html) in type declarations

### 2020-10-04

* [Windcatchers](https://en.wikipedia.org/wiki/Windcatcher) are an ancient and effective way to ventilate and cool a large space, and are still great techniques to use today. [Yakhchāl](https://en.wikipedia.org/wiki/Yakhch%C4%81l)s are related and also pretty cool.
* [Gnome sort](https://en.wikipedia.org/wiki/Gnome_sort) is an easy in-place sorting algorithm

### 2020-10-03

* Put commands in `/etc/rc.local` to have them run every time the system boots! Using the `@reboot` descriptor in your crontab doesn't always work, depending on the system-specific implementation of cron (it didn't on the Intel DE1-SoC board's distribution of Linux).
* Use `cron` to schedule recurring tasks, use `anacron` to schedule recurring tasks on machines that aren't powered on all the time. Any tasks that were missed while the computer was off, will be run when it turns on again with `anacron`.
* [Rectangle](https://github.com/rxhanson/Rectangle) for a super clean way to manage window tiling on MacOS

### 2020-10-02

* Handy guide to [printf](https://www.cypress.com/file/54441/download) formatting

### 2020-10-01

* [Linear probing](https://en.wikipedia.org/wiki/Linear_probing) for hashtables
* [Paper](sci-hub.se/10.1017/jfm.2020.720) about the effects of ventilation on indoor spread of COVID-19. Many common ventilation approaches actually increase the risk of COVID-19 exposure indoors.

### 2020-09-30

* [Survey](https://uw-se-2020-class-profile.github.io/profile.pdf) of Waterloo software engineering majors on all sorts of topics, ranging from salary to personal habits. (over 100 pages long!)
* `:set keywordprg=cppman` in vim gives you access to C++ function documentation when you press `K` over a function name
* [Shigeo Sekito](https://en.wikipedia.org/wiki/Shigeo_Sekito), a Japanese [electone](https://en.wikipedia.org/wiki/Electone) player whose song ["ザ・ワード II"](spotify:track:3OCxOUUH3FUf8xk0RuWADJ) was sampled in Travis Scott and Quavo's song "How U Feel". Sekito's enigmatic and diverse body of works seem to be heavily influenced by jazz, blues, bop, funk, and even a bit of electronica emerging at the time. You can hear his style progress and develop, especially in his 1985 album [_アーティスティック・エレクトーン_](spotify:album:0GWl71lcfvNjbjGwv11L0X). I absolutely love his sound, and especially the [title song](spotify:track:7CKAlVKKGnMW2GiIrZyuOw) – and the album just gets better.

### 2020-09-29

* [Algorithms](https://stackoverflow.com/questions/4325200/find-the-majority-element-in-array) for finding the majority element of a list in linear time
* [Academictree.org](https://academictree.org/) for family trees of people in academia
* [When to use](https://www.investopedia.com/terms/h/harmonicaverage.asp#:~:text=The%20Basics%20of%20a%20Harmonic%20Mean&text=Harmonic%20means%20are%20often%20used,weight%20to%20each%20data%20point.) the [harmonic mean](https://en.wikipedia.org/wiki/Average#Harmonic_mean)
* Pipeline [pigs](https://en.wikipedia.org/wiki/Pigging) 🐷

### 2020-09-28

* Darwinism suggests that human altruism may have evolved in a way such that groups of early humans who demonstrated more altruism, were more likely to survive in violent conflicts against other groups with less altruistic traits. [This study](https://www.researchgate.net/publication/26268380_Did_Warfare_Among_Ancestral_Hunter-Gatherers_Affect_the_Evolution_of_Human_Social_Behaviors) suggests that [violence](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0126589) among [early humans](https://www.nature.com/articles/nature16477) played a significant role in shaping the social dynamics of groups during the hunter-gatherer time period.
* [Everything We Eat Both Causes And Prevents Cancer](https://academic.oup.com/ajcn/article/97/1/127/4576988)

### 2020-09-27

* Purpleair has a [JSON API](https://www2.purpleair.com/community/faq#hc-access-the-json) 
* [Send SMS](https://github.com/typpo/textbelt) free, alternative to Twilio
* Detect news articles that have corporate sponsors with [this chrome extension](https://github.com/typpo/ad-detector)
* Wildfires have burned [**3.1 million acres**](https://www.fire.ca.gov/incidents/) in California so far this year. And fire season isn't even fully upon us.
* [Spin cleanup](https://sci-hub.se/10.1039/D0CP03745A) for quantum phase estimation
* 👉 [The Missing Semester Of Your CS Education](https://missing.csail.mit.edu/)

### 2020-09-26

* Github [refuses](https://talk.jekyllrb.com/t/katex-not-working-on-github-pages/4468/5) to use [KaTex](https://katex.org/) with Jekyll, switched over to just [mathjax](https://sgeos.github.io/github/jekyll/2016/08/21/adding_mathjax_to_a_jekyll_github_pages_blog.html).
* Installing mongodb on macOS Catalina with SIP means you have to [change the data directory](https://stackoverflow.com/a/58404057/2680053) by `mongod --dbpath=/Users/<your_username>/data/db`
* [Sigterm vs sigkill](https://major.io/2010/03/18/sigterm-vs-sigkill/) and C++ [signal handling](https://www.tutorialspoint.com/cplusplus/cpp_signal_handling.htm)

### 2020-09-25

* How to solve the [closest pair problem](https://en.wikipedia.org/wiki/Closest_pair_of_points_problem) in O(nlog(n)) time
* Cool [bit twiddling tricks](https://graphics.stanford.edu/~seander/bithacks.html) and [crazy bit manipulations](http://realtimecollisiondetection.net/blog/?p=78)
* Classes and inheritance in C++
* Use the `title` attribute instead of `#h1` in this file to make sure the title doesn't appear twice. [Weird things happen](https://github.com/mmistakes/minimal-mistakes/issues/1618#issuecomment-379807051) when you use the `#h1 ` thing.

### 2020-09-24

* How to use malloc in assembly
* Imposter syndrome (I might have it?)

### 2020-09-23

* How memory mapped IO works

### 2020-09-22

* How the [median of medians](https://en.wikipedia.org/wiki/Median_of_medians) algorithm works to produce the approximate median in $$O(n)$$ time
* How to [write a news article](https://www.theguardian.com/science/the-lay-scientist/2010/sep/24/1) about a scientific paper 🙃
* [Quagga](https://www.google.com/search?q=quagga)
* After you add an ssh key to Github, you need to use the ssh url to clone the repo in order to use your ssh credentials.

### 2020-09-21

* How to use partially observable markov decision processes ([POMDP](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process)) to efficiently search for items on a desk using a robotic manipulator, when some items obscure others. While solving for full solutions for a POMPD requires an $$O(n^2)$$ solution, using [Monte-Carlo sampling](https://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps.pdf) can cut this time down dramatically, enabling **[this paper](https://www.ccs.neu.edu/home/camato/publications/icra2020_1664.pdf)** to be able to plan movements of the robotic manipulator in real time. 
* In that paper, they use a POMDP by defining:
  * Set of world states (the objects on the table)
  * Set of actions (moving an object, moving the robot camera location/angle)
  * Set of observations (using R-CNN model to recognize bounding boxes for objects on the table)
  * State transition function, defining the probability distribution of new states after taking some action (moving an item) under the current state
  * Observation function, defining the probability distribution of possible observations (arrangements of obscured objects) given the resulting state and an action. Follows from the state transition function.
  * Reward function, giving a valued-reward for taking certain actions like finding the correct item.

### 2020-09-20

* How to play Among Us
* [Lilly impellers](https://www.treehugger.com/the-lily-impeller-nature-based-design-inspires-game-changing-efficiencies-4863361)

### 2020-09-19

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
