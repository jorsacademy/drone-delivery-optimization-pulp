from pulp import (
    LpBinary,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)


# ============================================================
# Data
# ============================================================
WAREHOUSES = ["W1", "W2", "W3"]
DRONES = ["D1", "D2", "D3", "D4", "D5"]
CUSTOMERS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]

capacity = {"D1": 10, "D2": 12, "D3": 8, "D4": 15, "D5": 10}
max_flight_time = {"D1": 20, "D2": 25, "D3": 18, "D4": 30, "D5": 22}
operating_cost = {"D1": 5, "D2": 6, "D3": 4, "D4": 7, "D5": 5}

customer_demand = {
    "C1": 2,
    "C2": 3,
    "C3": 1,
    "C4": 4,
    "C5": 2,
    "C6": 3,
    "C7": 2,
    "C8": 1,
    "C9": 3,
    "C10": 2,
}

# One-way travel time from each warehouse to each customer.
travel_time = {
    ("W1", "C1"): 5,
    ("W1", "C2"): 7,
    ("W1", "C3"): 6,
    ("W1", "C4"): 8,
    ("W1", "C5"): 9,
    ("W1", "C6"): 6,
    ("W1", "C7"): 7,
    ("W1", "C8"): 5,
    ("W1", "C9"): 8,
    ("W1", "C10"): 6,
    ("W2", "C1"): 6,
    ("W2", "C2"): 5,
    ("W2", "C3"): 7,
    ("W2", "C4"): 6,
    ("W2", "C5"): 8,
    ("W2", "C6"): 7,
    ("W2", "C7"): 6,
    ("W2", "C8"): 9,
    ("W2", "C9"): 7,
    ("W2", "C10"): 8,
    ("W3", "C1"): 8,
    ("W3", "C2"): 6,
    ("W3", "C3"): 5,
    ("W3", "C4"): 7,
    ("W3", "C5"): 6,
    ("W3", "C6"): 9,
    ("W3", "C7"): 8,
    ("W3", "C8"): 7,
    ("W3", "C9"): 6,
    ("W3", "C10"): 5,
}

# Fixed deployment cost if a drone is activated from a warehouse.
warehouse_deployment_cost = {
    ("W1", "D1"): 8,
    ("W1", "D2"): 8,
    ("W1", "D3"): 7,
    ("W1", "D4"): 9,
    ("W1", "D5"): 8,
    ("W2", "D1"): 7,
    ("W2", "D2"): 7,
    ("W2", "D3"): 6,
    ("W2", "D4"): 8,
    ("W2", "D5"): 7,
    ("W3", "D1"): 9,
    ("W3", "D2"): 8,
    ("W3", "D3"): 7,
    ("W3", "D4"): 9,
    ("W3", "D5"): 8,
}

# Maximum total mission time for any active drone.
MAX_MISSION_TIME = 30


# ============================================================
# Model
# ============================================================
model = LpProblem("Drone_Delivery_Optimization", LpMinimize)

# z[w,d] = 1 if drone d is deployed from warehouse w.
z = LpVariable.dicts(
    "deploy",
    [(w, d) for w in WAREHOUSES for d in DRONES],
    cat=LpBinary,
)

# y[w,d,c] = 1 if customer c is served by drone d deployed from warehouse w.
y = LpVariable.dicts(
    "serve",
    [(w, d, c) for w in WAREHOUSES for d in DRONES for c in CUSTOMERS],
    cat=LpBinary,
)

# a[d] = 1 if drone d is activated.
a = LpVariable.dicts("active", DRONES, cat=LpBinary)


# ============================================================
# Objective
# ============================================================
# Each assigned customer is modeled as an out-and-back trip from the drone's
# assigned warehouse. This is a mission-assignment model, not a customer-to-
# customer routing model.
model += (
    lpSum(
        2 * travel_time[w, c] * operating_cost[d] * y[w, d, c]
        for w in WAREHOUSES
        for d in DRONES
        for c in CUSTOMERS
    )
    + lpSum(
        warehouse_deployment_cost[w, d] * z[w, d]
        for w in WAREHOUSES
        for d in DRONES
    )
)


# ============================================================
# Constraints
# ============================================================

# Every customer must be served exactly once.
for c in CUSTOMERS:
    model += (
        lpSum(y[w, d, c] for w in WAREHOUSES for d in DRONES) == 1,
        f"customer_assignment_{c}",
    )

# Each drone can be deployed from at most one warehouse.
for d in DRONES:
    model += (
        lpSum(z[w, d] for w in WAREHOUSES) == a[d],
        f"single_warehouse_{d}",
    )

# A drone can serve a customer from a warehouse only if it is deployed there.
for w in WAREHOUSES:
    for d in DRONES:
        for c in CUSTOMERS:
            model += (
                y[w, d, c] <= z[w, d],
                f"deployment_link_{w}_{d}_{c}",
            )

# Payload capacity of each drone.
for d in DRONES:
    model += (
        lpSum(
            customer_demand[c] * y[w, d, c]
            for w in WAREHOUSES
            for c in CUSTOMERS
        )
        <= capacity[d] * a[d],
        f"capacity_{d}",
    )

# Total out-and-back flight time must remain within each drone's endurance.
for d in DRONES:
    model += (
        lpSum(
            2 * travel_time[w, c] * y[w, d, c]
            for w in WAREHOUSES
            for c in CUSTOMERS
        )
        <= max_flight_time[d] * a[d],
        f"flight_endurance_{d}",
    )

# Additional operational mission-time limit.
for d in DRONES:
    model += (
        lpSum(
            2 * travel_time[w, c] * y[w, d, c]
            for w in WAREHOUSES
            for c in CUSTOMERS
        )
        <= MAX_MISSION_TIME * a[d],
        f"mission_time_{d}",
    )


# ============================================================
# Solve
# ============================================================
solver = PULP_CBC_CMD(msg=False)
model.solve(solver)


# ============================================================
# Report
# ============================================================
print(f"Solver status: {LpStatus[model.status]}")

if LpStatus[model.status] == "Optimal":
    print(f"Objective value: {value(model.objective):.2f}\n")

    for d in DRONES:
        if value(a[d]) > 0.5:
            selected_warehouse = next(
                w for w in WAREHOUSES if value(z[w, d]) > 0.5
            )
            served_customers = [
                c
                for c in CUSTOMERS
                if value(y[selected_warehouse, d, c]) > 0.5
            ]

            total_load = sum(customer_demand[c] for c in served_customers)
            total_flight_time = sum(
                2 * travel_time[selected_warehouse, c] for c in served_customers
            )

            print(f"Drone {d}")
            print(f"  Warehouse: {selected_warehouse}")
            print(f"  Customers: {served_customers}")
            print(f"  Load: {total_load} / {capacity[d]}")
            print(
                f"  Flight time: {total_flight_time} / {max_flight_time[d]}"
            )
            print()
else:
    print("No optimal solution was found for the current data and constraints.")
