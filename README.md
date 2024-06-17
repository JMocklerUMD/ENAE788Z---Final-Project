# ENAE788Z Final Project
A final project for ENAE788Z: Decision-Making under Uncertainty at the University of Maryland, Dept. of Aerospace Engineering. 

### Overview
The goal of this work was to apply to methods from the class to a novel problem, culimating in a report and presentation. In this project, I applied a reinforcement-learning strategy to optimally learning traffic signalling at a four-way intersection. The project culimated in an 8-page IEEE-style report titled "Reinforcement Learning of a Four-Way Traffic Signal System" 

Read the paper and the powerpoint presentation in the "final report and presentations" folder!

### Problem formulation and key results
This work modelled a four-way intersection (with no turns for initial simplicity) as a Markov decision process (MDP). Then, the MDP was learned using a home-coded maximum likelihood reinforcement learning algorithm. Using the maximum of the state-action value function, the traffic signal was optimally determined from the length of traffic at the approaches. A few key images are detailed below. 

![training_plot](https://github.com/JMocklerUMD/ENAE788Z---Final-Project/assets/150191399/519d127d-b0c5-4088-8015-161c25ab2c8e)

*Fig 1: Training plot of the Reinforcement Learning. The training loss is directly proportional to the total queue lengths at the intersection.*

![LearnedvsConv_Flows_EW](https://github.com/JMocklerUMD/ENAE788Z---Final-Project/assets/150191399/eb41dda6-ddaf-4aa6-aad2-7cd96c3b2c9b)

*Fig 2: Execution of the learned policy, where the action taken (or the traffic signal deployed) is the argmax of the state-action value function. Figure shows the learned policy vs capacity analysis used commonly in practice to design traffic signals in the E/W-bound approach*

![LearnedvsConv_Flows_NS](https://github.com/JMocklerUMD/ENAE788Z---Final-Project/assets/150191399/1939d3e2-3783-4361-8298-99c2d492f354)

*Fig 3: Execution of the learned policy, where the action taken (or the traffic signal deployed) is the argmax of the state-action value function for the N/S-bound approach*

