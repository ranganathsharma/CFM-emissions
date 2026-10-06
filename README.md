# CFM-emissions

Emission modelling of vehicles using Car-Following Models.

This repository contains the code and results for an emissions modelling study
of human-driven and autonomous vehicles using Car-Following Models (CFM) and
Multi-anticipative Car-Following Models (MCFM).

The framework is demonstrated using the Intelligent Driver Model (IDM), the Full
Velocity Difference Model (FVDM), and their respective multi-anticipative
extensions:

- IDM/MIDM: https://www.sciencedirect.com/science/article/pii/S100757041400402X
- FVDM/MFVDM: https://www.sciencedirect.com/science/article/pii/S0378437122007543?via%3Dihub

The overall study framework is shown below.

![Emission design of experiment](Emission%20design%20of%20experiment.png)

The leader vehicle dynamics dictate the movement of the following vehicles in
the CFM class of models. These dynamics can be altered to represent urban,
rural, and highway driving conditions. This study uses standard drive cycles
from around the world to represent those leader dynamics. The study also
quantifies how follower-vehicle emissions differ from leader-vehicle emissions,
capturing the role of traffic-flow modelling in emissions estimation.

Vehicle emissions are estimated using the established microscopic emissions
model PHEMLight:

https://sumo.dlr.de/docs/Models/Emissions/PHEMlight.html

Fuel type, traffic-flow model, and model parameter values are varied to quantify
total emissions and their uncertainty under different driving conditions.

## Repository Structure

`Demo/`

A step-by-step demonstration of estimating CO2, NOx, PM, and fuel consumption
rates. The workflow is explained in `Demo/README.md`.

`Manuscript_results/`

Code and generated figures used for the manuscript results.

`Validation/`

Validation exercises used to check that the simulation and model outputs are
reliable. These are important checks, but they are not exhaustive; additional
validation is recommended for full simulation studies.

This work is carried out by Ranganatha Belagumba Ramachandra, Dr. Bidisha Ghosh, Dr. Vikram
Pakrashi and Dr. Siddartha Mounisai Middela. This repository is provided to promote 
reproducible research in transportation. The authors acknowledge the support provided by 
the RERITE tutorial in making this repository more 
accessible and reader friendly (https://www.rerite.org/itsc24-rr-tutorial/). 