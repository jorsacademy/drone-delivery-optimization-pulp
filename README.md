# Multi-Depot Drone Delivery Optimization with PuLP

This repository contains a mixed-integer linear programming (MILP) model for a multi-depot drone vehicle-routing problem implemented in Python with PuLP.

The model is no longer a simple warehouse-to-customer assignment model. It now constructs explicit customer-to-customer routes by using directed arc variables, so each active drone starts at one selected warehouse, visits its assigned customers in sequence, and returns to the same warehouse.

## Problem Scope

The optimization simultaneously decides:

- which drones are activated,
- which warehouse each active drone uses as its depot,
- which drone serves each customer,
- the order in which customers are visited,
- which directed flight arcs are used,
- customer service start times,
- cumulative payload along each route.

## Objective

The objective minimizes total system cost:

1. variable drone operating cost proportional to travel time, and
2. fixed warehouse deployment cost for each active drone.

## Constraints

The MILP includes:

- exact customer coverage,
- one selected warehouse per active drone,
- one departure from and one return to the selected warehouse,
- customer flow conservation,
- payload-capacity limits,
- route-duration limits,
- customer time windows,
- time propagation between consecutive visits,
- return-to-depot timing,
- load-based MTZ subtour elimination.

The MTZ structure prevents disconnected customer cycles that do not connect to the selected warehouse.

## Travel-Time Data

All warehouses and customers have two-dimensional coordinates. A complete travel-time matrix is generated from ceiling-rounded Euclidean distance. This produces consistent travel times for warehouse-to-customer, customer-to-customer, and customer-to-warehouse arcs.

The coordinates are illustrative educational data and can be replaced by real distances or travel times from another source.

## Solver

The model is solved with CBC through PuLP:

```python
solver = PULP_CBC_CMD(msg=False, timeLimit=60)
model.solve(solver)
```

The solver is therefore used to evaluate the complete MILP rather than only applying a heuristic construction procedure.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python drone_delivery_optimization.py
```

## Output

For an optimal solution, the script reports:

- solver status,
- objective value,
- selected warehouse for every active drone,
- ordered route,
- customers served,
- payload utilization,
- travel and service time,
- route-time utilization,
- variable travel cost,
- deployment cost,
- customer arrival times and time windows,
- total payload and total system cost.

A post-solve validation function independently checks customer coverage, depot return, duplicate visits, payload capacity, and route-duration feasibility.

## Model Structure

The main binary variables are:

- `deploy[w,d]`: drone `d` is deployed from warehouse `w`,
- `active[d]`: drone `d` is used,
- `assign[d,c]`: drone `d` serves customer `c`,
- `arc[d,i,j]`: drone `d` directly travels from node `i` to node `j`.

Continuous variables are used for customer arrival times and cumulative route load.

See [MODEL.md](MODEL.md) for the mathematical formulation.

## Modeling Improvements Over the Earlier Version

The earlier formulation contained route variables that did not actually represent customer-to-customer movement, warehouse-dependent travel terms that were aggregated incorrectly, and ordering variables that were not connected to a valid routing structure.

The current formulation replaces those disconnected decisions with explicit directed arc variables and route-flow equations. Warehouse selection, customer assignment, timing, payload, and subtour elimination are now linked to the same route structure.

## Educational Use

This repository is intended as an operations-research teaching example. The data set is small enough for students to inspect but the formulation contains the main components of a real multi-depot capacitated vehicle-routing model with time windows.

## License

This project is distributed under a custom non-commercial license. Commercial use, resale, paid integration, commercial redistribution, or use as part of a paid commercial product or service is prohibited without prior written permission from the copyright holder.

See [LICENSE](LICENSE) for the complete terms.
