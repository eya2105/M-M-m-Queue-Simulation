# M/M/m Queue Simulation

This project simulates an **M/M/m queueing system** and compares the **empirical** distribution of clients over time with the **theoretical** distribution calculated using queueing theory formulas.

It visualizes:
- System state over time
- Probability distribution of the number of clients
- Comparison between simulation and analytical results

---

## Features

- Continuous-time simulation of M/M/m queues
- Computation of theoretical stationary distribution
- Visualization of state trajectory and probability distributions
- Empirical vs theoretical mean comparison

---

## Model Description

- **M/M/m** represents a queueing system with:
  - Poisson arrivals (`λ`)
  - Exponential service times (`μ`)
  - `m` parallel servers

The system is stable if:  
> ρ = λ / (m × μ) < 1

---

## Requirements

- Python 3.x
- NumPy
- Matplotlib

Install the dependencies using:

```bash
pip install numpy matplotlib
