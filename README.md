# CFM-emissions
Emission modelling of vehicles using Car Following Models

This repository has the codes and results for the Emission modelling study for human driven and Autonomous Vehicles through Car Following and Multi-anticipative Car Following Models (CFM and MCFM). 

The framework of the study is demonstrated on Intelligent Driver Model (IDM) and Full Velocity Difference Model (FVDM) and their respective multi-anticipative extensions found at https://www.sciencedirect.com/science/article/pii/S100757041400402X and https://www.sciencedirect.com/science/article/pii/S0378437122007543?via%3Dihub. The framework of the study is presented below

<img width="1356" height="979" alt="doe" src="https://github.com/user-attachments/assets/a7da42b6-2d59-4fb0-9e99-46a7b9914596" />

The leader vehicle dynamics dictates the movement of following vehicles in CFM class of models and can be altered to represent urban, rural and highway type of driving conditions. We have utilized standard drive cycles used across the world to represent these dynamics. Further, the study quantifies the difference in follower vehicle emission from the leader vehicle capturing the aspect of traffic flow modelling in emission estimation. 

The emissions are estimated for each vehicle using the well established microscopic emission model - PHEMLight (https://sumo.dlr.de/docs/Models/Emissions/PHEMlight.html). The fuel type and the traffic flow model and the parameter values are varied to quantify the total emissions and its uncertainty under different conditions. 

The repository is structured as follows:

Codes:     Codes for calculating and plotting the results

Results:   Results for each model and condition

Data:      Input data of drive cycles

other files

Note: This project utilizes PHEMlight, a licensed product. Ownership of PHEMlight rests solely with its developers, and it is not distributed or modified here. No proprietary or sensitive information related to PHEMlight is disclosed in this repository. Use of PHEMlight requires a valid license obtained directly from the rights holder.
