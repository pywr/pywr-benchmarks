from pywr.core import Model
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



class StaticNetwork:
    params = [
        [10, 50],        # Number of zones
        [2, 5],     # Connection density
        ['glpk', ]  # solver
    ]
    param_names = [
        'number_of_resource_zones',
        'connection_density',
        'solver'
    ]

    def setup(self, nz, density, solver):
        np.random.seed(1337)
        self.model = make_simple_model(nz, density, solver)
        self.model.setup()

    def teardown(self, nz, density, solver):
        del self.model

    def time_run(self, nz, density, solver):
        self.model.run()

    def time_setup(self, nz, density, solver):
        self.model.dirty = True
        self.model.setup()


class AggregatedParameter:
    params = [
        [10, 100, 500],
    ]
    param_names = [
        'number_of_scenarios',
    ]

    def setup(self, number_of_scenarios):
        from pywr.core import Scenario

        data = {
            "metadata": {
                "title": "Simple 1",
                "description": "A very simple example.",
                "minimum_version": "0.1"
            },
            "timestepper": {
                "start": "2015-01-01",
                "end": "2015-02-1",
                "timestep": 7
            },
            "nodes": [
                {
                    "type": "input",
                    "name": "input",
                },
                {
                    "type": "link",
                    "name": "link"
                },
                {
                    "type": "output",
                    "name": "output",
                    "cost": -10,
                    "max_flow": "max_flow_param"
                }
            ],
            "edges": [
                ["input", "link"],
                ["link", "output"]
            ],
            "parameters": {
                "max_flow_param": {
                    "type": "aggregated",
                    "agg_func": "sum",
                    "parameters": [
                        {
                            "type": "constant",
                            "value": 5.0
                        },
                        {
                            "type": "constant",
                            "value": 5.0
                        },
                        {
                            "type": "constant",
                            "value": 5.0
                        }
                    ]
                }
            }
        }

        m = Model.load(data, )
        Scenario(m, name='benchmark', size=number_of_scenarios)

        m.setup()
        self.model = m

        self.param = m.parameters['max_flow_param']
        self.ts = m.timestepper.current

    def time_calc_values(self, number_of_scenarios):
        self.param.calc_values(self.ts)

    def teardown(self, number_of_scenarios):
        del self.model