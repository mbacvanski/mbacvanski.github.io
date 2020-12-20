---
layout: blog_outline
title: Proof of two nifty properties of Hermitian operators
mathjax: true
---

Given a Hermitian operator $\hat{H}$:

# 1. The Eigenvalues are Real

Let's say we apply our Hermitian operator $\hat{H}$ to a state $\vert\Psi\rangle$. We're going to get some scalar of an eigenvalue of $\hat{H}$, call this $h$. Also consider what happens when we take the $\dagger$ operator, the hermitian conjugate. We get these two equations:

$$
\hat{H}\vert\psi\rangle = h\vert\psi\rangle \qquad\text{and}\qquad \langle\psi\vert\hat{H} = \langle\psi\vert h^*
$$

Now let's take the inner product of the left equation with $\langle\psi\vert$, and the inner product of the right equation with $\vert\psi\rangle$. 

$$
\langle\psi\vert\hat{H}\vert\psi\rangle = h\langle\psi\vert\psi\rangle \qquad\text{and}\qquad \langle\psi\vert\hat{H}\vert\psi\rangle = \langle\psi\vert\psi\rangle h^*
$$

The left hand side of $\langle\psi\vert\hat{H}\vert\psi\rangle$ is the same so we can combine these statements.

$$
\begin{align}
h\langle\psi\vert\psi\rangle &= \langle\psi\vert\psi\rangle h^*\\
h\langle\psi\vert\psi\rangle - \langle\psi\vert\psi\rangle h^* &= 0\\
(h-h^*)\langle\psi\vert\psi\rangle &= 0
\end{align}
$$

So we see that in order for this to be $0$, either $(h-h^*)=0$ or $\langle\psi\vert\psi\rangle = 0$. It wouldn't make sense that our state $\vert\psi\rangle$ has a norm (length) of 0, and we can just define our state such that this is not the case. So the only thing left is that $h-h^\*=0$, where $h$ is an eigenvalue of $\hat{H}$ and $h^\*$ is its complex conjugate. 

If $h=a+bi$, then $h^*=a-bi$ by definition of complex conjugate. So because $h-h^\*=0$, $h$ must be real (have an imaginary component of 0). ✔️

# 2. The Eigenstates are Orthogonal

Let's look at what happens when we have two eigenvalue equations, one with a bra and one with a ket: $\hat{H}\vert\psi_a\rangle$ and $\langle\psi_b\vert\hat{H}$.

$$
\hat{H}\vert\psi_a\rangle=h_a \vert\psi_a\rangle \qquad\text{and}\qquad \langle\psi_b\vert\hat{H}=\langle\psi_b\vert h_b
$$

Now take the inner product of the left equation with $\langle\psi_b\vert$ and the inner product of the right equation with $\vert\psi_a\rangle$.

$$
\langle\psi_b\vert\hat{H}\vert\psi_a\rangle=h_a\langle\psi_b\vert\psi_a\rangle \qquad\text{and}\qquad \langle\psi_b\vert\hat{H}\vert\psi_a\rangle=\langle\psi_b\vert\psi_a\rangle h_b
$$

So once again we can combine these statements.

$$
h_a\langle\psi_b\vert\psi_a\rangle =\langle\psi_b\vert\psi_a\rangle h_b\\
(h_b-h_a)\langle\psi_b\vert\psi_a\rangle=0
$$

In order for this to be true either $(h_b-h_a)=0$ or $\langle\psi_b\vert\psi_a\rangle=0$. So the eigenvalues must either by identical, or the inner product between the eigenstates must be 0. If the eigenvalues are not degenerate ($h_a \neq h_b$), then of course our eigenstates are orthogonal. But what if our eigenvalues are degenerate? 

If the eigenvalues are equal $h_a=h_b=h$, then any state in the basis of $\vert\psi_a\rangle$ and $\vert\psi_b\rangle$, which we will denote as $(a\vert\psi_a\rangle + b\vert\psi_b\rangle)$ will also have an eigenvalue of our $h$. 

$$
\begin{align}
\hat{H}(a\vert\psi_a\rangle+b\vert\psi_b\rangle)&=a\hat{H}\vert\psi_a\rangle+b\hat{H}\vert\psi_b\rangle\\
&=ah_a\vert\psi_a\rangle+bh_b\vert\psi_b\rangle\\
&= ah\vert\psi_a\rangle+bh\vert\psi_b\rangle\\
&=h(a\vert\psi_a\rangle+b\vert\psi_b\rangle)
\end{align}
$$

What this means is that if our eigenvalues are identical, then there are an infinite number of eigenstates that correspond to that eigenvalue. Not all these eigenstates are orthogonal to each other, but we can simply choose a pair that are orthogonal. ✔️
