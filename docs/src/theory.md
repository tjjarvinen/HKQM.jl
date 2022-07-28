# Theory

This section is intended to be a self contained
reference to theory.

## Green's Functions

Consider an equation with a linear operator $\mathcal{L}$

```math
\begin{equation}
\mathcal{L}u(x)=f(x)
\end{equation}
```

from which we would like to solve $u(x)$. This can be done with a help of a special function called as Green's function $G(x,s)$ that is defined as

```math
\mathcal{L}G(x,s)=\delta(x-s)
```

where $\delta(x-s)$ is Dirac's delta function.
Multiplying this with $f(s)$ and integrating over $s$ leads to

```math
\int\mathcal{L}G(x,s)f(s)ds=\int\delta(x-s)f(s)ds=f(x)
```

Plugging this definition of $f(x)$ to Eq. (1) leads to

```math
\mathcal{L}u(x)=f(x)=\int\mathcal{L}G(x,s)f(s)ds
```

Changing order of operating to $x$ and integration of $s$

```math
\mathcal{L}u(x)=\mathcal{L}\left(\int G(x,s)f(s)ds\right)
```

leads to solultion

```math
\begin{equation}
u(x)=\int G(x,s)f(s)ds
\end{equation}
```

### Poisson equation

Poisson equation has general form

```math
\nabla^2 \varphi = f
```

it has a [Green's function](https://en.wikipedia.org/wiki/Green%27s_function) (3D)

```math
\begin{equation}
G(r_1,r_2)=\frac{-1}{4\pi|r_1-r_2|}
\end{equation}
```

### Helmholtz equation

Inhomogenous Helmholtz equation is defined as 

```math
(\nabla^2 + k^2)\varphi = f
```

and has a [Green's function](https://en.wikipedia.org/wiki/Green%27s_function) (3D)

```math
\begin{equation}
G(r_1,r_2)= e^{-ik|r_1-r_2|} \frac{-1}{4\pi|r_1-r_2|}
\end{equation}
```

## Electronic integrals

A charge density $\rho$ creates a electric potential $\phi$ acordingly Poisson equation (Gauss's Law)

```math
\nabla^2\phi = -\frac{\rho}{\varepsilon_0}
```

Electric potential can be solved using Poisson equation Green's function Eq. (3)

```math
\begin{equation}
\phi(r_1) = -\int G(r_1,r_2)\frac{\rho(r_2)}{\varepsilon_0} dr_2 = \int \frac{\rho(r_2)}{4\pi \varepsilon_0 |r_1-r_2|}dr_2 
\end{equation}
```

Electric potential can then be used to calculate ineraction
between two charge densities $\rho_1$ and $\rho_2$ by
first calculating electric potential from one charge
density and then integrating together with other charge
density

```math
\begin{align*}
\phi_1(r) = & \int \frac{\rho_1(r_1)}{4\pi \varepsilon_0 |r-r_1|}dr_1 \\
E_{interaction} = & \int \phi_1(r) \rho_2(r) dr
\end{align*}
```

## Schrödinger equation

General form of time independent Schrödinger equation is

```math
-\frac{\hbar^2\nabla^2}{2m}\psi + V\psi = E\psi
```

It can be changed to Helmholtz equation from

```math
\left( \nabla^2 + \frac{2m}{\hbar^2} \right)\psi = \frac{2m}{\hbar^2}\psi
```

with using

```math
\begin{align*}
k = & \frac{ \sqrt{2 mE} }{\hbar} \\
f = & \frac{2m}{\hbar^2}\psi
\end{align*}
```

Green's function can then be used to find a solution to Schrödinger equation

```math
\psi(r_1) = -\frac{m}{2\pi\hbar^2}\int \frac{\exp(-ik|r_1-r_2|)}{|r_1-r_2|}V(r_2)\psi(r_2)dr_2
```

Which is a complex due to $e^{-ik|r_1-r_2|}$ term. If energy is negative, the $i$ in front $k$ can be moved inside the square root in $k$ and make the equation real

```math
\begin{equation}
\psi(r_1) = -\frac{m}{2\pi\hbar^2}\int \frac{\exp(-\frac{ \sqrt{-2 mE} }{\hbar}|r_1-r_2|)}{|r_1-r_2|}V(r_2)\psi(r_2)dr_2
\end{equation}
```

This equation is iterative in nature due to it needing energy as an input and because wavefunction is present on both sides of the equation.

## Implementation

### Reduce dimensionality of integrals

All integrals above are 3-dimensional and thus expensive. It is thus very desirable to reduce the dimensionality of the integrals.

To do this consider standard [Gaussian integral](https://en.wikipedia.org/wiki/Gaussian_integral)

```math
\begin{equation}
    \int_{-\infty}^{\infty} e^{-at^2}dt = \sqrt{\frac{\pi}{a}}
\end{equation}
```

By taking $a=r^2$ this equation can be manipulated to

```math
\begin{equation}
    \frac{1}{r} = \frac{2}{\sqrt{\pi}} \int_{0}^{\infty}e^{-r^2t^2}dt.
\end{equation}
```

This form allow us to get rid of square root in distance calculation $r=\sqrt{x^2+y^2+z^2}$, by just separating $r^2$ term

```math
r^2 = x^2 + y^2 + z^2.
```

With this the 3-dimensional integral in Eq. (5) can be changes to four 1-dimensional integrals

```math
\phi(r_1) = \frac{1}{2\varepsilon_0\pi^{3/2}} \int_{0}^{\infty}dt\int \rho(r_2) e^{-\Delta r^2t^2}dr_2 \nonumber \\
```

```math
\phi(r_1) = \frac{1}{2\varepsilon_0\pi^{3/2}} \int_{0}^{\infty}dt\int \rho(x_2,y_2,z_2) e^{-\Delta x^2t^2}e^{-\Delta y^2t^2}e^{-\Delta z^2t^2}dx_2dy_2dz_2 \nonumber
```

By introducing

```math
T(t,\Delta x) = e^{-\Delta x^2 t^2} = e^{-(x_1-x_2)^2t^2}
```

the integral takes form

```math
\phi(r_1) = \frac{1}{2\varepsilon_0\pi^{3/2}}
\int_{0}^{\infty}dt\int_{-\infty}^{\infty} dx_2 T(t, \Delta x)
\int_{-\infty}^{\infty}dy_2 T(t, \Delta y)
\int_{-\infty}^{\infty} dz_2 T(t, \Delta z)\rho(r_2).
```

This form has four one dimensional integrals that can be
calculated in serial fashion. Meaning that the computational complexity
was reduced from $N^3$ to $4N$

### Helmholtz Equation

Helmholtz equation differs from Poisson equation by additional
$\exp (-kr)$ therm

```math
\frac{\exp (-kr)}{r}.
```

With a standard Laplace transformation [3]

```math
\int_{0}^{\infty}\exp(-sp)\frac{\exp(-\frac{a^2}{4p})}{\sqrt{\pi p}}dp
 = \frac{\exp(-a\sqrt{s})}{\sqrt{s}}
```

and supstituting $p=t^2$, $s=r^2$ and $a=k$ we get

```math
    \frac{\exp (-kr)}{r} = \frac{2}{\sqrt{\pi}}
    \int_{0}^{\infty}\exp(-\frac{k^2}{4t^2}-t^2r^2)dt
```

that is separaple in variable $r$. Using the $T$ that was defined above leads to

```math
    \frac{\exp (-kr)}{r} = \frac{2}{\sqrt{\pi}}
    \int_{0}^{\infty}\exp(-\frac{k^2}{4t^2})
    T(t,\Delta x)T(t,\Delta y)T(t,\Delta z)dt.
```

This leads to final equation

```math
    \psi(r_1) = -\frac{m}{\pi^{3/2}\hbar^2}\int_{0}^{\infty}dt\exp(-\frac{k^2}{4t^2}) \int_{-\infty}^{\infty}dx_2 T(t, \Delta x)
    \int_{-\infty}^{\infty}dy_2 T(t, \Delta y)
    \int_{-\infty}^{\infty}dz_2 T(t, \Delta z)
    V(r_2)\psi(r_2)dr_2
```

that is almost the same as for Poisson equation.
With only differences being that charge density is
replaced with $V\psi$ and the extra $\exp(-\frac{k^2}{4t^2})$-term
for *t*-integration.

## Basis

The most accurate way to do general numerical integration is by using
[Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature).
The integral is performed by calculating values of function in certain points that and summing up with a weight, which gives integral of Legendre polynomial over an interval [a,b]

```math
\int_{a}^{b}f(x)dx \approx \sum_{i=1}^N\omega_{i}f(x_{i}).
```

The basis is thus Gauss-Legendre polynomials, which are divided to elements
to control the number of polynomials and spacing in different locations.

At the current implementation the elements are cubic and contain the same
number of Gauss-Legendre points. But the theory does not make this kind of
restrictions. It is possible to define elements and Gauss-Lagrange points
separately for x-, y- and z-coordinates. But currently due to ease of
implementation the program has symmetric implementation. Later of the will
be support for more complicated ones.

## Tensor Representation

The basis is formed of elements and Gauss-Legendre polynomial points each
of wich have x-,y- and z-coordinates. Thus wavefunctions, densities, etc.
are six dimensional tensors.

```math
\rho(x,y,z) = \rho_{\alpha\beta\gamma IJK}
```

Where $\alpha$, $\beta$ and $\gamma$ are indices for Gauss-Lagrange points
and $I$, $J$ and $K$ are indices for elements.

An integral is then a tensor operation, using Einstein's notation

```math
\int\rho(r)dr = \omega_{\alpha I}\omega_{\beta J}\omega_{\gamma I}\rho_{\alpha\beta\gamma IJK}.
```

Where $\omega$ is Gauss-Legendre quadrature weight.

In code this is easily calculated using [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) that impelements
[tensor networks](https://www.tensors.net/)

```julia
@tensor ω[α,I]*ω[β,J]*ω[γ,K]*ρ[α,β,γ,I,J,K]
```

### Poisson Equation

In tensor representation the $T$ is a 5-rank tensor

```math
T(t,x_1,x_2) = T_{\alpha\alpha'II't}
```

The calculation is done so that position coordinates are contracted first
and the integral over t-coordinate is calculated on the way.
This is necessary as otherwise the memory needed for intermediate steps
would be too much.

To do this a 4th rank tensor is taken out from $T$

```math
\tilde{T}_{\alpha\alpha'II'}^i = T_{\alpha\alpha'II't_i}
```

this leads to

```math
\phi_{\alpha\beta\gamma IJK}=
\frac{1}{2\varepsilon_0\pi^{3/2}}\sum_{i=1}^{N}\omega_{i}
\tilde{T}_{\alpha\alpha'II'}^i
\tilde{T}_{\beta\beta'JJ'}^i
\tilde{T}_{\gamma\gamma'KK'}^i
\rho_{\alpha'\beta'\gamma' I'J'K'}
```

### Helmholtz Equation in Schrödinger Equation Case

The representation is the same except for the extra term and potential
energy.

The calculation is done by first calculating energy to get $k$. After that
the potential energy and the initial state are multiplyed

```math
\tilde{\psi} = V\psi.
```

This leads to a following tensor expression

```math
\phi_{\alpha\beta\gamma IJK}=
-\frac{m}{\pi^{3/2}\hbar^2}
\sum_{i=1}^{N}\omega_{i}\exp(-\frac{k^2}{4t^2})
\tilde{T}_{\alpha\alpha'II'}^i
\tilde{T}_{\beta\beta'JJ'}^i
\tilde{T}_{\gamma\gamma'KK'}^i
\tilde{\psi}_{\alpha'\beta'\gamma' I'J'K'}
```

In practice the extra term $\exp(-\frac{k^2}{4t^2})$ is included in
the integration weight $w_i$ and the constant term in front
($\frac{m}{\pi^{3/2}\hbar^2}$) is ignored,
because the wavefunction needs to be normalized in any case.

## Hartree-Fock

Hartree-Fock equation is an eigen-value equation for Fock matrix

```math
F_{ij} = h_{ij} + 2J_{ij} + K_{ij}.
```

Where $h_{ij}$ is one particle Hamiltonian

```math
h = \frac{1}{2}\nabla^2 + V,
```

$J$ is Coulomb operator

```math
J = \int\frac{\sum_{k}|\psi_{k}(r_2)|^2}{|r_1-r_2|}dr_2
```

and K is exhange operator

```math
K_{ij}\psi_{j} = \sum_{k}\psi_{k}\int\frac{\psi_{j}^{*}(r_2)\psi_{k}(r_2)}{|r_1-r_2|}dr_2.
```

Defining

```math
\tilde{V} = V + 2J + K
```

leads to

```math
F = \frac{1}{2}\nabla^2 + \tilde{V},
```

which is in same form as Schrödinger equation in above and thus can be
solved the same way.

With Helmholtz equation there is only need to calculate the eigen-value
of orbital, to calculta $k$

```math
<\psi_{i}|F_{ii}|\psi_{i}> 
```

and applying $\tilde{V}$ to orbitals once per orbital. The Fock matrix is thus only calculated diagonally to occupied orbitals.

This has two consequences. First, scaling on number of orbitals is reduced
to $O(N^2)$. Secondly, orbitals can be updated on parallel, followed by an
ortogonalization. Meaning that the orbital optimization scales linearry on number of orbitals.

## Magnetic Field Calculations

In magnetic field the Hamiltonian for a charged particle is [4]

```math
H = \frac{1}{2m}(-i\hbar\nabla+qA)^2 + V.
```

Helmholtz equation needs that $\nabla^2$ is separated to its own term.
This leads to the following expression

```math
H = -\frac{\hbar^2}{2m}\nabla^2 +i\hbar q\nabla A -i\hbar qA\nabla
    + q^2A^2 + V.
```

Defining

```math
\tilde{V} = +i\hbar q\nabla A -i\hbar qA\nabla + q^2A^2 + V 
          = -[p, A] + q^2A^2 + V
```

leads to general Schrödinger equation type expression

```math
H = -\frac{\hbar^2}{2m}\nabla^2 + \tilde{V},
```

which can be solved in the normal way. With the only exception that
the wavefunction is now complex.

## References

[1] M. H. Kalos; Phys. Rev. **128**, (1962)
[https://doi.org/10.1103/PhysRev.128.1791](https://doi.org/10.1103/PhysRev.128.1791)

[2] Harrision, et al.; J. Chem. Phys. **121**, 11587 (2004);
[https://doi.org/10.1063/1.1791051](https://doi.org/10.1103/PhysRev.128.1791)

[3] Schum's outlines Mathematical Handbook of Formulas and Tables,
2nd edition; edit. Murray R. Spiegel and John Liu;
Mc Graw Hill; 1999; ISBN 0-07-116765-X

[4] Lectures on Quantum Mechanics, 2nd edition; Steven Weinberg;
Cambridge University press; 2015; ISBN 978-1-107-11166-0
