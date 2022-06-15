# Theory

This section is intended to be a self contained
reference to theory.

## Green's Functions

Consider an equation with a linear operator $\mathcal{L}$

$$
\begin{equation}
\mathcal{L}u(x)=f(x)
\end{equation}
$$

from which we would like to solve $u(x)$. This can be done with a help of a special function called as Green's function $G(x,s)$ that is defined as

$$ \mathcal{L}G(x,s)=\delta(x-s) $$

where $\delta(x-s)$ is Dirac's delta function.
Multiplying this with $f(s)$ and integrating over $s$ leads to

$$ \int\mathcal{L}G(x,s)f(s)ds=\int\delta(x-s)f(s)ds=f(x) $$

Plugging this definition of $f(x)$ to Eq. (1) leads to

$$
\mathcal{L}u(x)=f(x)=\int\mathcal{L}G(x,s)f(s)ds
$$

Changing order of operating to $x$ and integration of $s$

$$
\mathcal{L}u(x)=\mathcal{L}\left(\int G(x,s)f(s)ds\right)
$$

leads to solultion

$$
\begin{equation}
u(x)=\int G(x,s)f(s)ds
\end{equation}
$$

### Poisson equation

Poisson equation has general form

$$
\nabla^2 \varphi = f
$$

it has a [Green's function](https://en.wikipedia.org/wiki/Green%27s_function) of from

$$
\begin{equation}
G(r_1,r_2)=\frac{-1}{4\pi|r_1-r_2|}
\end{equation}
$$

### Helmholtz equation

Inhomogenous Helmholtz equation is defined as 

$$
(\nabla^2 + k^2)\varphi = f
$$

and [Green's function](https://en.wikipedia.org/wiki/Green%27s_function) of form (3D)

$$
\begin{equation}
G(r_1,r_2)= e^{-ik|r_1-r_2|} \frac{-1}{4\pi|r_1-r_2|}
\end{equation}
$$

## Electronic integrals

A charge density $\rho$ creates a electric potential $\phi$ acordingly Poisson equation (Gauss's Law)

$$
\nabla^2\phi = -\frac{\rho}{\varepsilon_0}
$$

Electric potential can be solved using Poisson equation Green's function Eq. (3)

$$
\begin{equation}
\phi(r_1) = -\int G(r_1,r_2)\frac{\rho(r_2)}{\varepsilon_0} dr_2 = \int \frac{\rho(r_2)}{4\pi \varepsilon_0 |r_1-r_2|}dr_2 
\end{equation}
$$

Electric potential can then be used to calculate ineraction
between two charge densities $\rho_1$ and $\rho_2$ by
first calculating electric potential from one charge
density and then integrating together with other charge
density

$$
\begin{align*}
\phi_1(r) = & \int \frac{\rho_1(r_1)}{4\pi \varepsilon_0 |r-r_1|}dr_1 \\
E_{interaction} = & \int \phi_1(r) \rho_2(r) dr
\end{align*}
$$

## Schrödinger equation

General form of time independent Schrödinger equation is

$$
-\frac{\hbar^2\nabla^2}{2m}\psi + V\psi = E\psi
$$

It can be changed to Helmholtz equation from

$$
\left( \nabla^2 + \frac{2m}{\hbar^2} \right)\psi = \frac{2m}{\hbar^2}\psi
$$

with using

$$
\begin{align*}
k = & \frac{ \sqrt{2 mE} }{\hbar} \\
f = & \frac{2m}{\hbar^2}\psi
\end{align*}
$$

Green's function can then be used to find a solution to Schrödinger equation

$$
\psi(r_1) = -\frac{m}{2\pi\hbar^2}\int \frac{\exp(-ik|r_1-r_2|)}{|r_1-r_2|}V(r_2)\psi(r_2)dr_2
$$

Which is a complex due to $e^{-ik|r_1-r_2|}$ term. If energy is negative, the $i$ in front $k$ can be moved inside the square root in $k$ and make the equation real

$$
\begin{equation}
\psi(r_1) = -\frac{m}{2\pi\hbar^2}\int \frac{\exp(-\frac{ \sqrt{-2 mE} }{\hbar}|r_1-r_2|)}{|r_1-r_2|}V(r_2)\psi(r_2)dr_2
\end{equation}
$$

This equation is iterative in nature due to it needing energy as an input and because wavefunction is present on both sides of the equation.

# Implementation

## Reduce dimensionality of integrals

All integrals above are 3-dimensional and thus expensive. It is thus very desirable to reduce the dimensionality of the integrals.

To do this consider standard [Gaussian integral](https://en.wikipedia.org/wiki/Gaussian_integral)

$$
\begin{equation}
    \int_{-\infty}^{\infty} e^{-at^2}dt = \sqrt{\frac{\pi}{a}}
\end{equation}
$$

By taking $a=r^2$ this equation can be manipulated to

$$
\begin{equation}
    \frac{1}{r} = \frac{2}{\sqrt{\pi}} \int_{0}^{\infty}e^{-r^2t^2}dt
\end{equation}
$$

This form allow us to get rid of square root in distance calculation $r=\sqrt{x^2+y^2+z^2}$, by just separating $r^2$ term

$$
r^2 = x^2 + y^2 + z^2
$$

With this the 3-dimensional integral in Eq. (5) can be changes to four 1-dimensional integrals

$$
\begin{align}
\phi(r_1) = & \frac{1}{2\varepsilon_0\pi^{3/2}} \int_{0}^{\infty}dt\int \rho(r_2) e^{-\Delta r^2t^2}dr_2 \nonumber \\
= & \frac{1}{2\varepsilon_0\pi^{3/2}} \int_{0}^{\infty}dt\int \rho(x_2,y_2,z_2) e^{-\Delta x^2t^2}e^{-\Delta y^2t^2}e^{-\Delta z^2t^2}dx_2dy_2dz_2
\end{align}
$$

By introducing

$$
\begin{equation}
T(x_1,x_2) = e^{-(x_1-x_2)^2t^2}
\end{equation}
$$

The integral takes form

$$
\begin{equation}
\phi(r_1) = \frac{1}{2\varepsilon_0\pi^{3/2}} \int_{0}^{\infty}dt\int dx_2 T(x_1,x_2) \int dy_2 T(y_1,y_2)  \int dz_2\rho(r_2) T(z_1,z_2)
\end{equation}
$$

# References

[1] M. H. Kalos; Phys. Rev. **128**, (1962); [https://doi.org/10.1103/PhysRev.128.1791]()

[2] Harrision, et al.; J. Chem. Phys. **121**, 11587 (2004);  [https://doi.org/10.1063/1.1791051]()