from pywr.core import Model, Input, Link, Output
from .random_network import make_simple_model_direct
import numpy as np
import os


class StaticNetwork:
    """ Bench mark a randomly generated steady state network. """

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
        self.model = make_simple_model_direct(nz, density, solver)
        try:
            self.model.setup()
        except:
            pass
        self.model.reset()

    def teardown(self, nz, density, solver):
        del self.model

    def time_run(self, nz, density, solver):
        self.model.run()

    def time_setup(self, nz, density, solver):
        self.model.dirty = True
        self.model.setup()


def load_xml_model(filename=None, data=None):
    """Load a test model and check it"""
    import xml.etree.ElementTree as ET
    if data is None:
        path = os.path.join(os.path.dirname(__file__), 'models', filename)
        with open(path, 'r') as f:
            data = f.read()
    else:
        path = None
    xml = ET.fromstring(data)
    model = Model.from_xml(xml, path=path)
    return model


class Simple1:
    """ Benchmark `simple1.json` test model.

    For older version of Pywr this benchmark falls back to using `sample1.xml`. This
    model is very simple and is mostly a throughput test.
    """
    params = [
        [1, 10, 100, 500],
    ]
    param_names = [
        'number_of_scenarios',
    ]

    def setup(self, number_of_scenarios):

        directory = os.path.join(os.path.dirname(__file__), "models")

        try:
            with open(os.path.join(directory, 'simple1.json')) as fh:
                m = Model.load(fh)
        except:
            m = load_xml_model(os.path.join(directory, 'simple1.xml'))

        if number_of_scenarios > 1:
            from pywr.core import Scenario
            Scenario(m, name='benchmark', size=number_of_scenarios)

        try:
            m.setup()
        except:
            pass

        self.model = m

    def time_run(self, number_of_scenarios):
        self.model.reset()
        for i in range(100):
            self.model.step()


class DemandSaving1:
    """ Benchmark `demand_saving1.json` test model.

    This model includes a more complex combination of `Parameter` objects. Useful
    to find regressions in the calculation of these.
    """
    params = [
        [1, 10, 100, 500],
    ]
    param_names = [
        'number_of_scenarios',
    ]

    def setup(self, number_of_scenarios):

        directory = os.path.join(os.path.dirname(__file__), "models")

        with open(os.path.join(directory, 'demand_saving1.json')) as fh:
            m = Model.load(fh)

        if number_of_scenarios > 1:
            from pywr.core import Scenario
            Scenario(m, name='benchmark', size=number_of_scenarios)

        if hasattr(m, 'setup'):
            m.setup()
        self.model = m

    def time_run(self, number_of_scenarios):
        self.model.run()
