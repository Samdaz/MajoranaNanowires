![alt text](Logo.png)

MajoranaNanowires is a numerical package for Python to model and simulate Majorana nanowire devices and other related semiconductor/superconductor heterostructures. Simulations describe nanowires supporting Majorana bound states (or other related structures) using tight-binding Hamiltonians. It is also possible to include electrostatic interactions with an arbitrary environment through finite element techniques.

---
<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#examples">Examples</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#installation">Installation</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

---

## Key features

The main features that this package includes are:
 <ul>
  <li> It allows to describe Majorana nanowires using the Kitaev model or the Lutchyn-Oreg one. It is also possible to use more sophisticated 8-band k.p models, although with some limitations.</li>
  <li> The wire can be described using a 1D Hamiltonian (for an effective single-channel description of the wire), 2D (for the description of the section of the wire) or 3D (for a full description of the wire).</li>
  <li> Any of them can be done in position or momentum space.</li>
  <li> It allows to simulate the electrostatic environment which tipically surrounds these nanostructures.</li>
  <li> It is possible to include self-consistent interactions between the electrons inside the nanowire, and/or between the electrons in the nanowire with the surrounding media.</li>
</ul> 


## Examples

Work in progress.


## Benchmarks

Work in progress.


## Documentation

Work in progress. 


## Installation




## Credits and requirements

This package was initially created by Samuel D. Escribano as a result of his PhD thesis. Apart from built-in and standard scientific Python modules (*Numpy* and *Scipy*), MajoranaNanowires relies on some other open-source packages, in particular:

* [Pfaffian](https://arxiv.org/abs/1102.3440)— this package allows to compute efficiently the Pfaffian of a matrix. MajoranaNanowires package uses it to compute the topological invariant corresponding to a 1D Hamiltonian. This package is already included in MajoranaNanowires, so no further installation is needed.

* [Fenics](https://fenicsproject.org/)— this package uses finite element methods to solve aritrary partial differential equations. MajoranaNanowires package uses it to solve the Poisson equation for an specific electrostatic environment. Please visit the project webpage to install it.


## License
MajoranaNanowires (Quantum Simulation Package) is released under the MIT License.



## How to cite
If you use this package to perform simulations for scientific results, please consider citing it. You can download [here](https://github.com/Samdaz/MajoranaNanowires/blob/master/MajoranaNanowiresQSP.bibtex) the citation in BibTex format, or just mention our repository with its corresponding DOI number (i.e. Github repository *MajoranaNanowires Quantum Simulation Package*, DOI:...).





