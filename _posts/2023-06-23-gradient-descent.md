---
layout: blog_outline
title: Gradient Descent in Disguise 
mathjax: true
---

Last semester I found myself sitting on a kitchen floor at a house party in Cambridge. To my left, a bag of boxed wine, and to my right, a very inebriated guest vigorously explaining to me how doing a Ph.D. is really just about learning to do the scientific method really well. I’ve been thinking about him ever since.

The scientific method I learned in middle school was a dreary flowchart – a discrete set of steps, a cycle that you take grudgingly once or at most twice in order to show on your poster board presentation so your teacher can check off the box `[x] Followed scientific method?` on the grading rubrik. Create a hypothesis, conduct an experiment, then re-evaluate your hypothesis — with a bunch of silly steps in between like “research the topic area” and “analyze data.” It wasn't that deep.

Recently I’ve come to realize that the scientific method is far more than our middle school teachers chalked it up to be. Instead of being disjointed and discrete steps, the scientific method is a rigorous way to approach optimality, in the continuous landscape of the world we live in. Optimization is the study of this very idea, and it’s an idea that flows beneath everything from science, engineering, business, and living a fulfilling life. Within the field of optimization, gradient descent is one of the OG ([Original Gradient]([Urban Dictionary: OG](https://www.urbandictionary.com/define.php?term=OG)) way to find optima of complicated continuous functions that you can’t solve analytically. Gradient descent is widely used, easy to implement, and fairly intuitive. Gradient descent is how neural networks are trained (but [not always!](/blog/forward-forward-nn)). We can take some big ideas from gradient descent and apply them to areas from developing software to learning a new skill, from building a startup to understanding how our universe works.

## Gradient Descent

Optimization is about making progress and moving towards an optimal solution. Gradient descent does this in a fun and intuitive way. 

1. Understand what your goal is, and formulate it as a “cost” or “goodness” function that you want to minimize.
2. Repeat:
   1. Look around your starting point in each of the directions. Where does the function slope up? Where does it slope down?
   2. Move in the direction that minimizes badness (and maximizes goodness)

Like gradient descent, most things in life are both about the journey (moving in a good direction) and the destination (your (hopefully) global optima).

### Cost functions

Before you can set out to make progress, you have to know how what you’re looking for. What is your goal, and how will you measure it? Life is a continuous space, so binary goals are less useful than measurable metrics. How can you quantify both your progress so far, as well as what’s remaining to get you to where you want?

### Compute the gradient

This is the key to making efficient progress: you want to know very precisely in which direction to go towards your goal. The next action you take should best minimize your cost function. It’s convenient when your function is differentiable, but that’s not always easy to do, both in real life and in optimization problems. When you can’t take the derivative directly, you have to estimate. The simplest way is to take small steps in each direction around your current position, and see how the landscape changes very locally around you. Finite-difference approaches to estimate derivatives do just this. 

### Descend the gradient

The gradient will tell you in which direction to go next. How far you move depends on a parameter often called “step size”. Intuitively, step size is how far you move before you stop and re-evaluate where you’re at — and you may realize that from this new vantage point, a new direction may be better. Move too far and you’re likely to put in unnecessary work or never converge, overshooting your goal with every step you take. Move too little and you’ll take a very long time to reach your goal. Step size can depend on how confident you are in your gradient estimate, or how experienced you are in what you’re doing.

## How To Do It

### Keep it simple.

The simpler your cost function, the easier it is to estimate your gradient, and therefore the more accurate your direction will be. Reduce your dimensionality. Finding a gradient in two dimensions is far easier and more accurate than finding the gradient in 2,000 dimensions. Don’t start by building a generic product for everybody — pick a niche and excel in it. **How can you simplify your goals? Identify what’s truly most important, and don’t get distracted.**  

### Invest in understanding.

The more precisely you understand your environment and local landscape, the better you can model the game you’re playing, the better you can estimate your gradient. In artificial intelligence, this means to visualize your data, agent, and policy as much as possible. In startups, this might mean to talk to customers continuously to validate your assumptions as they change. **Spending time building tools to help you better understand the problem and your progress isn’t sexy in the short term, but often ends up being far more worthwhile than we think.**

### Move quickly, in small steps.

Most of the time, the journey is equally important as (and often more important than) the destination. In science, small steps are easy to understand and therefore easy to reproduce, validate, and rigorously prove. A great leap forward that skips over intermediate steps lacks the rigor that many small steps, each bringing incremental improvement, lends to the scientific process. In startups, being able to show validated learning (link) means showing your gradient descent steps. In software engineering, making incremental, easily validatable changes reduces the chance of bugs compared to large monolithic pull requests. Moving in small step sizes forces you to be precise, and allows you to show that each step is both necessary and meaningful. **Big steps are tempting, but risky. We are often more confident than we should be.** 

### Validate your cost function, early and often.

You can be on track, just the wrong track. Taking a step back to ensure you’re on the right track, in the grandest scheme of things, can make the greatest differences. Look around and check if your cost function truly represents what you want. In ML, this means cross validation. In startups, this means collecting real feedback on an MVP and new ideas. In life, this means taking a step back to re-evaluate your priorities, assess if they have changed, and if your happiness and well-being now lies somewhere else. **Ask if you’re making real-world progress, not just in your model of the world.**

 
