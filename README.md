# Drone Delivery Optimization with PuLP

This repository contains a mixed-integer linear programming (MILP) model for a multi-warehouse drone delivery assignment problem implemented with PuLP.

## Problem Overview

The model assigns customers to drones and assigns each active drone to a single warehouse. The objective is to minimize total operating and deployment cost while satisfying customer coverage, drone payload capacity, flight endurance, and mission-time constraints.

This implementation deliberately models each customer service as an **out-and-back trip from the assigned warehouse**. It is therefore a mission-assignment model rather than a full vehicle-routing model with customer-to-customer arcs.

## Main Decisions

The model determines:

- which drones are activated,
- which warehouse each active drone is deployed from,
- which customers are served by each drone.

## Constraints

The optimization includes the following constraints:

- every customer is served exactly once,
- each drone is deployed from at most one warehouse,
- customer assignments are linked to warehouse deployment,
- drone payload capacity cannot be exceeded,
- total out-and-back flight time cannot exceed drone endurance,
- total mission time cannot exceed an operational upper bound.

## Objective Function

The objective minimizes:

1. variable operating cost based on round-trip travel time, and
2. fixed warehouse deployment cost for active drones.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python drone_delivery_optimization.py
```

The script uses PuLP's CBC solver through `PULP_CBC_CMD` and prints the solver status, objective value, warehouse selected for each active drone, assigned customers, load utilization, and flight-time utilization.

## Important Modeling Note

An earlier formulation of this problem used route-selection variables, customer assignment variables, warehouse deployment variables, time variables, and ordering variables without fully linking them. In particular, the warehouse-dependent travel-time term was aggregated incorrectly, route variables did not represent actual routes, and the ordering variables did not define a valid subtour-elimination structure.

The current formulation removes those disconnected variables and uses a consistent three-index assignment variable `serve[w,d,c]`, which directly links the selected warehouse, drone, and customer.

If sequential customer-to-customer routing is required, the model should be extended to a true VRP/MILP formulation with arc variables such as `x[d,i,j]` and appropriate flow and subtour-elimination constraints.

## License

This project is released under a custom non-commercial license. Commercial use is prohibited without prior written permission from the copyright holder. See [LICENSE](LICENSE) for details.
