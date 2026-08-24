"""Multi-depot drone routing MILP implemented with PuLP.

The model assigns each customer to exactly one drone, selects one warehouse for
any active drone, constructs a complete warehouse-to-customers-to-warehouse
route, enforces payload capacity, route-duration limits, and customer time
windows, and eliminates disconnected customer subtours.
"""

from math import ceil, hypot

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
# Sets and data
# ============================================================
WAREHOUSES = ["W1", "W2", "W3"]
DRONES = ["D1", "D2", "D3", "D4", "D5"]
CUSTOMERS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]
NODES = WAREHOUSES + CUSTOMERS

# Coordinates are used only to generate a complete deterministic travel-time
# matrix. One coordinate unit is treated as one unit of flight time.
coordinates = {
    "W1": (0, 0),
    "W2": (12, 0),
    "W3": (6, 10),
    "C1": (2, 2),
    "C2": (4, 1),
    "C3": (6, 3),
    "C4": (9, 2),
    "C5": (11, 4),
    "C6": (3, 7),
    "C7": (6, 7),
    "C8": (8, 8),
    "C9": (10, 7),
    "C10": (5, 9),
}

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

# Service time spent at each customer.
service_time = {
    "C1": 1,
    "C2": 1,
    "C3": 1,
    "C4": 2,
    "C5": 1,
    "C6": 2,
    "C7": 1,
    "C8": 1,
    "C9": 2,
    "C10": 1,
}

# Customer service must begin within [earliest, latest].
time_windows = {
    "C1": (0, 24),
    "C2": (0, 24),
    "C3": (2, 28),
    "C4": (0, 24),
    "C5": (3, 30),
    "C6": (0, 28),
    "C7": (4, 32),
    "C8": (5, 34),
    "C9": (4, 32),
    "C10": (6, 36),
}

payload_capacity = {"D1": 8, "D2": 10, "D3": 7, "D4": 12, "D5": 9}
max_route_time = {"D1": 38, "D2": 42, "D3": 36, "D4": 48, "D5": 40}
operating_cost = {"D1": 5, "D2": 6, "D3": 4, "D4": 7, "D5": 5}

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


def build_travel_time_matrix():
    """Return integer Euclidean travel times between every distinct node pair."""
    matrix = {}
    for i in NODES:
        for j in NODES:
            if i == j:
                continue
            xi, yi = coordinates[i]
            xj, yj = coordinates[j]
            matrix[i, j] = ceil(hypot(xi - xj, yi - yj))
    return matrix


travel_time = build_travel_time_matrix()

# Valid directed route arcs. Warehouse-to-warehouse arcs are intentionally
# excluded because a drone starts and ends at one selected warehouse.
ARCS = (
    [(w, c) for w in WAREHOUSES for c in CUSTOMERS]
    + [(i, j) for i in CUSTOMERS for j in CUSTOMERS if i != j]
    + [(c, w) for c in CUSTOMERS for w in WAREHOUSES]
)


# ============================================================
# Input validation
# ============================================================
def validate_data():
    """Fail early when model data is incomplete or internally inconsistent."""
    assert set(coordinates) == set(NODES)
    assert set(customer_demand) == set(CUSTOMERS)
    assert set(service_time) == set(CUSTOMERS)
    assert set(time_windows) == set(CUSTOMERS)
    assert set(payload_capacity) == set(DRONES)
    assert set(max_route_time) == set(DRONES)
    assert set(operating_cost) == set(DRONES)

    for c in CUSTOMERS:
        earliest, latest = time_windows[c]
        assert 0 <= earliest <= latest
        assert customer_demand[c] > 0
        assert service_time[c] >= 0

    for d in DRONES:
        assert payload_capacity[d] >= max(customer_demand.values())
        assert max_route_time[d] > 0
        assert operating_cost[d] >= 0

    for w in WAREHOUSES:
        for d in DRONES:
            assert (w, d) in warehouse_deployment_cost


validate_data()


# ============================================================
# MILP model
# ============================================================
model = LpProblem("Multi_Depot_Drone_Routing", LpMinimize)

# deploy[w,d] = 1 if drone d is based at warehouse w.
deploy = LpVariable.dicts(
    "deploy",
    [(w, d) for w in WAREHOUSES for d in DRONES],
    cat=LpBinary,
)

# active[d] = 1 if drone d is used.
active = LpVariable.dicts("active", DRONES, cat=LpBinary)

# assign[d,c] = 1 if drone d serves customer c.
assign = LpVariable.dicts(
    "assign",
    [(d, c) for d in DRONES for c in CUSTOMERS],
    cat=LpBinary,
)

# x[d,i,j] = 1 if drone d directly flies from node i to node j.
x = LpVariable.dicts(
    "arc",
    [(d, i, j) for d in DRONES for i, j in ARCS],
    cat=LpBinary,
)

# arrival[d,c] is service start time at customer c. It is forced to zero when
# the customer is not assigned to drone d.
arrival = LpVariable.dicts(
    "arrival",
    [(d, c) for d in DRONES for c in CUSTOMERS],
    lowBound=0,
)

# load_after[d,c] tracks cumulative payload delivered after visiting c. It
# also acts as an MTZ-style ordering variable for subtour elimination.
load_after = LpVariable.dicts(
    "load_after",
    [(d, c) for d in DRONES for c in CUSTOMERS],
    lowBound=0,
)


# ============================================================
# Objective function
# ============================================================
travel_cost = lpSum(
    operating_cost[d] * travel_time[i, j] * x[d, i, j]
    for d in DRONES
    for i, j in ARCS
)

deployment_cost = lpSum(
    warehouse_deployment_cost[w, d] * deploy[w, d]
    for w in WAREHOUSES
    for d in DRONES
)

model += travel_cost + deployment_cost, "total_operating_and_deployment_cost"


# ============================================================
# Constraints
# ============================================================
# Every customer is served exactly once by exactly one drone.
for c in CUSTOMERS:
    model += (
        lpSum(assign[d, c] for d in DRONES) == 1,
        f"serve_once_{c}",
    )

# An active drone is deployed from exactly one warehouse; an inactive drone
# is deployed from none.
for d in DRONES:
    model += (
        lpSum(deploy[w, d] for w in WAREHOUSES) == active[d],
        f"single_depot_{d}",
    )

# For each possible warehouse, selected deployment creates exactly one route
# departure and exactly one route return for that drone.
for d in DRONES:
    for w in WAREHOUSES:
        model += (
            lpSum(x[d, w, c] for c in CUSTOMERS) == deploy[w, d],
            f"depot_departure_{w}_{d}",
        )
        model += (
            lpSum(x[d, c, w] for c in CUSTOMERS) == deploy[w, d],
            f"depot_return_{w}_{d}",
        )

# Customer flow conservation links route arcs to customer assignment.
for d in DRONES:
    for c in CUSTOMERS:
        incoming = (
            lpSum(x[d, w, c] for w in WAREHOUSES)
            + lpSum(x[d, i, c] for i in CUSTOMERS if i != c)
        )
        outgoing = (
            lpSum(x[d, c, w] for w in WAREHOUSES)
            + lpSum(x[d, c, j] for j in CUSTOMERS if j != c)
        )
        model += incoming == assign[d, c], f"incoming_flow_{d}_{c}"
        model += outgoing == assign[d, c], f"outgoing_flow_{d}_{c}"

# Total demand assigned to a drone cannot exceed its payload capacity.
for d in DRONES:
    model += (
        lpSum(customer_demand[c] * assign[d, c] for c in CUSTOMERS)
        <= payload_capacity[d] * active[d],
        f"payload_capacity_{d}",
    )

# Route duration includes flight time and service time.
for d in DRONES:
    model += (
        lpSum(travel_time[i, j] * x[d, i, j] for i, j in ARCS)
        + lpSum(service_time[c] * assign[d, c] for c in CUSTOMERS)
        <= max_route_time[d] * active[d],
        f"route_time_{d}",
    )

# Customer time windows. Unassigned arrival variables are exactly zero.
for d in DRONES:
    for c in CUSTOMERS:
        earliest, latest = time_windows[c]
        model += (
            arrival[d, c] >= earliest * assign[d, c],
            f"earliest_service_{d}_{c}",
        )
        model += (
            arrival[d, c] <= latest * assign[d, c],
            f"latest_service_{d}_{c}",
        )

# Big-M for time propagation. This is deliberately derived from the data
# instead of using an arbitrary extremely large constant.
max_latest = max(latest for _, latest in time_windows.values())
max_service = max(service_time.values())
max_leg = max(travel_time.values())
TIME_M = max(max_route_time.values()) + max_latest + max_service + max_leg

# Time propagation from the selected warehouse to the first customer.
for d in DRONES:
    for w in WAREHOUSES:
        for c in CUSTOMERS:
            model += (
                arrival[d, c]
                >= travel_time[w, c] - TIME_M * (1 - x[d, w, c]),
                f"time_from_depot_{d}_{w}_{c}",
            )

# Time propagation between consecutive customers.
for d in DRONES:
    for i in CUSTOMERS:
        for j in CUSTOMERS:
            if i == j:
                continue
            model += (
                arrival[d, j]
                >= arrival[d, i]
                + service_time[i]
                + travel_time[i, j]
                - TIME_M * (1 - x[d, i, j]),
                f"time_between_{d}_{i}_{j}",
            )

# The final customer must be able to return to the selected warehouse within
# the route-duration limit.
for d in DRONES:
    for c in CUSTOMERS:
        for w in WAREHOUSES:
            model += (
                arrival[d, c]
                + service_time[c]
                + travel_time[c, w]
                <= max_route_time[d] + TIME_M * (1 - x[d, c, w]),
                f"return_deadline_{d}_{c}_{w}",
            )

# Load-based MTZ constraints eliminate disconnected customer-only cycles and
# provide an interpretable cumulative-load variable.
max_demand = max(customer_demand.values())
for d in DRONES:
    cap = payload_capacity[d]
    load_m = cap + max_demand

    for c in CUSTOMERS:
        model += (
            load_after[d, c] >= customer_demand[c] * assign[d, c],
            f"load_lower_{d}_{c}",
        )
        model += (
            load_after[d, c] <= cap * assign[d, c],
            f"load_upper_{d}_{c}",
        )

    for i in CUSTOMERS:
        for j in CUSTOMERS:
            if i == j:
                continue
            model += (
                load_after[d, j]
                >= load_after[d, i]
                + customer_demand[j]
                - load_m * (1 - x[d, i, j]),
                f"mtz_load_{d}_{i}_{j}",
            )


# ============================================================
# Solve
# ============================================================
solver = PULP_CBC_CMD(msg=False, timeLimit=60)
model.solve(solver)


# ============================================================
# Solution utilities
# ============================================================
def selected_warehouse_for(drone):
    """Return the selected warehouse for an active drone."""
    for w in WAREHOUSES:
        if value(deploy[w, drone]) is not None and value(deploy[w, drone]) > 0.5:
            return w
    return None


def build_route(drone, warehouse):
    """Reconstruct an ordered route from the binary arc solution."""
    route = [warehouse]
    current = warehouse
    max_steps = len(CUSTOMERS) + 2

    for _ in range(max_steps):
        next_nodes = [
            j
            for i, j in ARCS
            if i == current
            and value(x[drone, i, j]) is not None
            and value(x[drone, i, j]) > 0.5
        ]
        if not next_nodes:
            raise RuntimeError(f"Broken route for {drone}: no successor after {current}.")

        nxt = next_nodes[0]
        route.append(nxt)
        if nxt == warehouse:
            return route
        current = nxt

    raise RuntimeError(f"Route reconstruction exceeded the safe step limit for {drone}.")


def route_statistics(drone, route):
    """Calculate route-level load, travel time, service time, and cost."""
    customers = [node for node in route if node in CUSTOMERS]
    route_travel = sum(travel_time[i, j] for i, j in zip(route, route[1:]))
    route_service = sum(service_time[c] for c in customers)
    route_load = sum(customer_demand[c] for c in customers)
    route_cost = operating_cost[drone] * route_travel
    return customers, route_load, route_travel, route_service, route_cost


def validate_solution(routes):
    """Perform simple post-solve checks independent of the MILP constraints."""
    visited = [c for route in routes.values() for c in route if c in CUSTOMERS]
    assert sorted(visited) == sorted(CUSTOMERS), "Customers are not covered exactly once."

    for d, route in routes.items():
        warehouse = route[0]
        assert route[-1] == warehouse, f"{d} does not return to its start warehouse."
        customers, load, travel, service, _ = route_statistics(d, route)
        assert len(customers) == len(set(customers)), f"{d} visits a customer more than once."
        assert load <= payload_capacity[d] + 1e-6, f"{d} violates payload capacity."
        assert travel + service <= max_route_time[d] + 1e-6, f"{d} violates route time."


# ============================================================
# Report
# ============================================================
status = LpStatus[model.status]
print(f"Solver status: {status}")

if status == "Optimal":
    routes = {}
    total_travel_time = 0
    total_service_time = 0
    total_payload = 0
    variable_cost_value = 0.0
    deployment_cost_value = 0.0

    print(f"Objective value: {value(model.objective):.2f}\n")

    for d in DRONES:
        if value(active[d]) is None or value(active[d]) <= 0.5:
            continue

        warehouse = selected_warehouse_for(d)
        route = build_route(d, warehouse)
        routes[d] = route

        customers, load, travel, service, variable_cost = route_statistics(d, route)
        fixed_cost = warehouse_deployment_cost[warehouse, d]

        total_travel_time += travel
        total_service_time += service
        total_payload += load
        variable_cost_value += variable_cost
        deployment_cost_value += fixed_cost

        print(f"Drone {d}")
        print(f"  Warehouse: {warehouse}")
        print(f"  Route: {' -> '.join(route)}")
        print(f"  Customers served: {customers}")
        print(f"  Payload: {load} / {payload_capacity[d]}")
        print(f"  Travel time: {travel}")
        print(f"  Service time: {service}")
        print(f"  Route time: {travel + service} / {max_route_time[d]}")
        print(f"  Variable travel cost: {variable_cost:.2f}")
        print(f"  Deployment cost: {fixed_cost:.2f}")

        for c in customers:
            print(
                f"    {c}: arrival={value(arrival[d, c]):.2f}, "
                f"window={time_windows[c]}, demand={customer_demand[c]}"
            )
        print()

    validate_solution(routes)

    print("System totals")
    print(f"  Total payload delivered: {total_payload}")
    print(f"  Total travel time: {total_travel_time}")
    print(f"  Total service time: {total_service_time}")
    print(f"  Travel operating cost: {variable_cost_value:.2f}")
    print(f"  Deployment cost: {deployment_cost_value:.2f}")
    print(f"  Verified objective: {variable_cost_value + deployment_cost_value:.2f}")
    print("  Post-solve validation: passed")
else:
    print("No proven optimal solution was found within the current model and solver settings.")
