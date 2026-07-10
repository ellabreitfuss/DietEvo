# DietEvo

Conceptual diagram:

Network
   │
   ▼
Norms ───────┐
             │
Attitudes ───┤
             ▼
        Intention
             │
             ▼
        Behavior
         │      │
         ▼      ▼
Individual   Environmental
 Health        Health


# Food Transition Agent-Based Model (ABM)

An agent-based model (ABM) to investigate how **individual health motivations**, **environmental motivations**, and **social norms** interact to drive dietary transitions in a household and friendship network.

The model combines the **Theory of Planned Behavior (TPB)** with a **social network model** to study the emergence of sustainable dietary behavior.

---

# Research Question

Can environmental health improve when interventions focus **only on individual health**, rather than explicitly promoting environmentally friendly diets?

The model explores how health-oriented behavior may diffuse through social networks and generate environmental co-benefits.

---

# Model Overview

The model consists of two interacting layers:

1. **Network Layer**
2. **Agent-Based Behavioral Layer**

---

## Network Layer

Each node represents one individual.

Agents are connected by

- Household relationships
- Friendship relationships

The network is static during the simulation.

Social influence is transmitted through these network connections.

Household members exert a stronger influence than friends.

Current default weights:

| Relationship | Weight |
|--------------|-------:|
| Household | 0.80 |
| Friends | 0.20 |

---

## Agent Layer

Each agent contains the following state variables.

### Environmental dimension

- Environmental attitude
- Environmental norm
- Environmental perceived behavioral control (PBC)
- Environmental intention
- Environmental behavior

---

### Individual health dimension

- Health attitude
- Health norm
- Health perceived behavioral control (PBC)
- Health intention
- Health behavior

---

### System variables

Each agent additionally stores

- Individual health status

The model contains one global variable

- Environmental health status

---

# Behavioral Model

The behavioral model is based on the **Theory of Planned Behavior (TPB).**

For each dimension:

Attitude
↓
Norm
↓
Perceived Behavioral Control (PBC)
↓
Intention
↓
Behavior

---

## Intention

Intentions are calculated as

\[
I=w_AA+w_NN+w_{PBC}PBC
\]

where

- Attitudes
- Subjective norms
- Perceived behavioral control

jointly determine behavioral intention.

---

## Behavior

Behavior depends on

- Intention
- PBC
- Behavioral threshold

Behavior only changes if

\[
|B_{t-1}-B^{target}| \ge x
\]

where

\[
B^{target}=I\cdot PBC
\]

This represents behavioral inertia and switching costs.

---

# Social Influence

Norms are updated from the observed behavior of connected agents.

Each time step

1. Household behavior is averaged.
2. Friendship behavior is averaged.
3. Both are combined using predefined network weights.

Norm update:

\[
N_t=(1-m)N_{t-1}+mDN
\]

where

- \(DN\) = descriptive social norm
- \(m\) = motivation to comply

---

# Health Feedback

Individual health status depends on

- previous health
- health behavior
- environmental health
- stochastic noise

Poor health increases health attitudes.

Health attitudes are cumulative and do not decrease.

---

# Environmental Feedback

Environmental health depends on

- previous environmental health
- average environmental behavior
- stochastic noise

Environmental attitudes adapt to the current environmental health status.

Poor environmental health increases environmental concern.

Improving environmental health reduces environmental concern.

---

# Feedback Loops

The model contains both reinforcing and balancing feedback loops.

## Reinforcing

Behavior
↓
Social norms
↓
Intentions
↓
Behavior

---

## Balancing (Health)

Poor health
↓
Health attitude
↓
Health behavior
↓
Improved health

---

## Balancing (Environment)

Poor environmental health
↓
Environmental attitude
↓
Environmental behavior
↓
Improved environmental health

---

# Initialization

Agents are initialized with

- attitudes
- norms

sampled from a truncated Gaussian distribution

- mean = 0.5
- standard deviation = 0.2

Values are

- clipped to [0,1]
- discretized to 0.1 increments

---

# Simulation Order

Each simulation step follows

1. Update environmental health
2. Update individual health
3. Update attitudes
4. Update norms
5. Calculate intentions
6. Update behaviors

---

# Project Structure

```
Food-Transition-ABM/

│
├── main.py
│
├── network/
│   ├── households.py
│   └── friendships.py
│
├── simulation/
│   ├── agents.py
│   └── model.py
│
└── README.md
```

---

# Running the Model

Clone the repository

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python main.py
```

---

# Dependencies

- Python 3.11+
- NumPy
- NetworkX
- Matplotlib

---

# Future Extensions

Potential future developments include
- Evolutionary dynamics (reproduction of agents and with that norms)
- Add global influence (internet/media)
- Food availability and prices
- Policy interventions
- Heterogeneous TPB parameters
- Empirical calibration
- Validation with dietary survey data

---

# Citation

If you use this model, please cite

```
Author (2025)

Food Transition Agent-Based Model
GitHub Repository
```
