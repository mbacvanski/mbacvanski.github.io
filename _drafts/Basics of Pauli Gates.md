# Basics of Pauli Gates

The Pauli gates are three of the most fundamental gates used in quantum computation. 

Quantum logic gates must observe several properties. We will see that these properties are upheld by the Pauli gates. We begin by examining some key matrix operators and properties.

## Complex Conjugate

We take the complex conjugate of a complex number by swapping the sign of the imaginary component. The complex conjugate is often written as a *.
$$
(a+bi)^* = (a-bi)
$$

## Hermitian Adjoint Operator

The adjoint operator is also referred to as the dagger operator. It works by taking the complex conjugate, and then the transpose of the matrix or vector it is applied to.

- The Hermitian conjugate of a bra is the corresponding ket:
  $$
  |\psi\rangle^\dagger = \langle\psi|
  $$

* The Hermitian conjugate of a complex number is its complex conjugate:

- The Hermitian conjugate of the Hermitian conjugate is itself:
  $$
  (A^\dagger)^\dagger = A
  $$

## Quantum Gates Must be Unitary

A matrix is unitary if it does not alter the length of a vector that is multiplied by it. In other words, a unitary matrix is one where when it is multiplied by its adjoint, it is equal to the identity – applying the inverse of a unitary matrix effectively undoes the previous application of it.
$$
||U \cdot |\psi\rangle|| = |||\psi\rangle|
$$

$$
U^\dagger U = I
$$

where $I$ is the identity matrix.