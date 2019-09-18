from pywr.core import Model, Input, Link, Output
from pywr.parameters import ArrayIndexedParameter
from .json_utils import add_node, update_node, add_connection
import numpy as np
from scipy import stats


def make_simple_resource_zone(data, name, supply_loc=9, supply_scale=1, demand_loc=8, demand_scale=3):
    """
    Make a very simply WRZ with a supply, WTW and demand.


    """
    # Generate supply side maximum flow
    max_flow = stats.norm.rvs(loc=supply_loc, scale=supply_scale, size=1)[0]
    add_node(data, name="Supply-{}".format(name), type="input", cost=-10, max_flow=max(max_flow, 0))

    # No flow constraint on WTW
    add_node(data, name="WTW-{}".format(name), type="link", cost=1)

    # Generate demand side maximum flow
    max_flow = stats.norm.rvs(loc=demand_loc, scale=demand_scale, size=1)[0]
    add_node(data, name="Demand-{}".format(name), type="output", cost=-500, max_flow=max(max_flow, 0))

    add_connection(data, "Supply-{}".format(name), "WTW-{}".format(name))
    add_connection(data, "WTW-{}".format(name), "Demand-{}".format(name))


def make_simple_resource_zone_direct(model, name, supply_loc=9, supply_scale=1, demand_loc=8, demand_scale=3,
                                     supply_time_varying=True):
    """
    Make a very simply WRZ with a supply, WTW and demand.


    """
    # Generate supply side maximum flow
    if supply_time_varying:
        max_flow = stats.norm.rvs(loc=supply_loc, scale=supply_scale, size=len(model.timestepper))
        max_flow[max_flow < 0] = 0.0
        max_flow = ArrayIndexedParameter(model, max_flow)
    else:
        max_flow = stats.norm.rvs(loc=supply_loc, scale=supply_scale, size=1)[0]
        max_flow = max(max_flow, 0)

    inpt = Input(model, name="Supply-{}".format(name), cost=-10, max_flow=max_flow)

    # No flow constraint on WTW
    link = Link(model, name="WTW-{}".format(name), cost=1)

    # Generate demand side maximum flow
    max_flow = stats.norm.rvs(loc=demand_loc, scale=demand_scale, size=1)[0]
    otpt = Output(model, name="Demand-{}".format(name), cost=-500, max_flow=max(max_flow, 0))

    inpt.connect(link)
    link.connect(otpt)


def make_simple_connections(data, number_of_resource_zones, density=10, loc=15, scale=5):
    num_connections = (number_of_resource_zones ** 2) * density // 100 // 2

    connections = np.random.randint(number_of_resource_zones, size=(num_connections, 2))
    max_flow = stats.norm.rvs(loc=loc, scale=scale, size=num_connections)

    added = []

    for (i, j), mf in zip(connections, max_flow):
        if (i, j) in added or i == j:
            continue
        name = "Transfer {}-{}".format(i, j)
        add_node(data, name=name, type="link", max_flow=max(mf, 0), cost=1)

        add_connection(data, "WTW-{}".format(i), name)
        add_connection(data, name, "WTW-{}".format(j))

        added.append((i, j))


def make_simple_connections_direct(model, number_of_resource_zones, density=10, loc=15, scale=5):
    num_connections = (number_of_resource_zones ** 2) * density // 100 // 2

    added = []
    while len(added) < num_connections:
        i, j = np.random.randint(number_of_resource_zones, size=2)
        mf = stats.norm.rvs(loc=loc, scale=scale, size=1)
        if (i, j) in added or i == j:
            continue
        name = "Transfer {}-{}".format(i, j)

        link = Link(model, name=name, max_flow=max(mf, 0), cost=1)

        from_name = "WTW-{}".format(i)
        try:
            from_node = model.nodes[from_name]
        except TypeError:
            from_node = model.node[from_name]

        to_name = "WTW-{}".format(j)
        try:
            to_node = model.nodes[to_name]
        except TypeError:
            to_node = model.node[to_name]

        from_node.connect(link)
        link.connect(to_node)
        added.append((i, j))


def make_simple_model(number_of_resource_zones=1, connection_density=10, solver=None):
    data = {
        "metadata": {
            "title": "Simple 1",
            "description": "A very simple example.",
            "minimum_version": "0.1"
        },
        "timestepper": {
            "start": "2015-01-01",
            "end": "2015-02-1",
            "timestep": 1
        },
        "nodes": [],
        "edges": []
    }

    for i in range(number_of_resource_zones):
        make_simple_resource_zone(data, "{}".format(i))

    make_simple_connections(data, number_of_resource_zones, density=connection_density)
    return Model.load(data, solver=solver)


def make_simple_model_direct(number_of_resource_zones=1, connection_density=10, solver=None, time_varying=True):
    model = Model(solver=solver)

    for i in range(number_of_resource_zones):
        make_simple_resource_zone_direct(model, "{}".format(i), supply_time_varying=time_varying)

    make_simple_connections_direct(model, number_of_resource_zones, density=connection_density)
    return model