
![ascfd logo](https://github.com/user-attachments/assets/4d73c58c-e9b8-457e-993d-4614eb6b06d3#gh-dark-mode-only)
![ascfd logo light](https://github.com/user-attachments/assets/c0e90a31-abcf-47ce-b9cb-ff35a7b556ba#gh-light-mode-only)

---
<p align="center">
  <i align="center">A next-generation, hybrid MHD code for Hall-Effect Thruster simulation and design.</i>
</p>

![ascfd banner 2](https://github.com/user-attachments/assets/f08a45b3-4e36-41eb-a532-98c68c6b8db3#gh-dark-mode-only)
![ascfd banner light](https://github.com/user-attachments/assets/dad2f09c-c803-4399-add4-1c4e95a7b65b#gh-light-mode-only)

---

**Built by the DeGrendele Simulation Lab.**

Weiping Li, Phoenix Do, Efthimios Gkatzimas, Matthew Fang, Rhea Haridas, Aditya Kaul, Akhil Muthyala, Ashita Pant, Ammar Sajwani, Vunal Jinasundera, and Chris DeGrendele.

## Abstract

Effectively simulating Hall-Effect ion thrusters (HETs) is critical for enabling deep space travel and requires advanced Magnetohydrodynamics (MHD) methods. We aim to build a reliable and precise MHD HET simulation which improves upon existing kinetic Particle-In-Cell (PIC) and hybrid models in both accuracy and efficiency that can be used to optimize HET designs with minimal material cost. Building from our existing 2D Euler code, a refined PIC method will allow us to accurately track the detailed kinetic behavior of the particles, while an optimized hybrid flux scheme will improve computational efficiency. Key engineering tools, such as thrust calculation and embedded boundary support, will also be implemented to aid applied design usage. 

## Architecture
### MHD Fluid Electron Loop

Good for efficiency and analyzing general behavior.
1. Iterates through grid
2. Iterates through variables
3. Calculates HLLD fluxes
4. Updates conserved variables

### Particle-in-Cell Ion Loop

Good for analyzing anomalies and small-scale physics.
1. Iterates through particles
2. Solves ion equations
3. Moves and updates particles

### Logistical Modules

To model physical thruster aspects:
- Embedded boundaries
- Gas injection
  
For applied design usage:
- Thrust calculations
- Wall degradation
- Key performance metrics

Coupling modules to keep electron and ion loops synchronized.

## Timeline
![revised timeline dark](https://github.com/user-attachments/assets/fc83833b-0fd2-4baa-a20e-aad88cd6f444#gh-dark-mode-only)
![revised timeline light](https://github.com/user-attachments/assets/cd92f100-2744-440f-a921-2ea560c8b7b4#gh-light-mode-only)

---
![cse](https://github.com/user-attachments/assets/d91a05a4-8a67-4879-8e39-fde85355afc9#gh-dark-mode-only)
![cse light](https://github.com/user-attachments/assets/e3ee81c7-ac71-428e-a380-4190bcb09378#gh-light-mode-only)

