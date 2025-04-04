![ascfd logo](https://github.com/user-attachments/assets/3ca911f8-6cb8-489e-93fd-e65bcb236f54)
---
<p align="center">
  <i align="center">A next-generation, hybrid MHD code for Hall-Effect Thruster simulation and design.</i>
</p>

![ascfd banner 2](https://github.com/user-attachments/assets/f08a45b3-4e36-41eb-a532-98c68c6b8db3)
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
![timeline](https://github.com/user-attachments/assets/1730f596-9a32-4b8d-81d4-9d76f6445a7a)

---
![cse](https://github.com/user-attachments/assets/d91a05a4-8a67-4879-8e39-fde85355afc9)

