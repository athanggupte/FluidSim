# FluidSim: SPH-based fluid simulation

This project is a submission to CSCI-596: Scientific Computing and Visualization, a course offered at the University of Southern California.

![image](https://github.com/athanggupte/FluidSim/assets/45060273/236418b5-157b-45cb-9b8f-86a88db3f7c1)

## About

This project implements a fluid simulation using the Smoothed Particle Hydrodynamics (SPH) method, a mesh-free Lagrangian technique that approximates fluids as a collection of particles. The simulation is written in C++ and CUDA, and uses OpenGL for visualization.

## Features

- Interactive simulation of fluid dynamics with realistic physics
- Real-time rendering of fluid particles with lighting and shading effects
- User interface to control simulation parameters and camera movement
- Performance optimization using parallel computing and spatial hashing

## Theory

This project is based on several fundamental concepts in fluid dynamics and numerical simulation. Here is a brief overview:

### Smoothed Particle Hydrodynamics (SPH)

Smoothed Particle Hydrodynamics (SPH) is a computational method used for simulating the mechanics of continuum media, such as solid mechanics and fluid flows. It is a meshfree particle method based on Lagrangian formulation, and has been widely applied to different areas in engineering and science. The SPH method and its recent developments include the need for meshfree particle methods, approximation schemes of the conventional SPH method, and numerical techniques for deriving SPH formulations for partial differential equations.

### Navier-Stokes Equations

The Navier-Stokes equations are partial differential equations which describe the motion of viscous fluid substances. They were named after French engineer and physicist Claude-Louis Navier and the Irish physicist and mathematician George Gabriel Stokes. They mathematically express momentum balance and conservation of mass for Newtonian fluids. They arise from applying Isaac Newton's second law to fluid motion, together with the assumption that the stress in the fluid is the sum of a diffusing viscous term (proportional to the gradient of velocity) and a pressure term.

### Weakly Compressible SPH Formulation of Pressure Force

In the context of SPH, the pressure force of a particle is determined by a symmetric SPH formulation to preserves linear and angular momentum. The weak compressibility assumption is introduced, which directly considers the pressure as a function of the density. This approach eliminates the need for evolving the pressure at the sound speed and significantly increases the allowed time steps.

## Usage

The simulation window has a user interface that shows the current simulation parameters and allows you to adjust them.  
Use the mouse to orbit around the simulation region, and scroll to zoom in and out.
