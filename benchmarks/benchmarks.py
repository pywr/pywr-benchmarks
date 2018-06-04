from pywr.core import Model, Input, Link, Output
from .random_network import make_simple_model_direct
import numpy as np
import os
import json
import pandas

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

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
        [1, 2, 5, 10, 20, 40, 80, 160, 320],
    ]
    param_names = [
        'number_of_scenarios',
    ]

    def setup(self, number_of_scenarios):

        try:
            with open(os.path.join(MODELS_DIR, 'simple1.json')) as fh:
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
        [1, 2, 5, 10, 20, 40, 80, 160, 320],
    ]
    param_names = [
        'number_of_scenarios',
    ]

    def setup(self, number_of_scenarios):

        with open(os.path.join(MODELS_DIR, 'demand_saving1.json')) as fh:
            m = Model.load(fh)

        if number_of_scenarios > 1:
            from pywr.core import Scenario
            Scenario(m, name='benchmark', size=number_of_scenarios)

        if hasattr(m, 'setup'):
            m.setup()
        self.model = m

    def time_run(self, number_of_scenarios):
        self.model.run()


class SimpleThames:
    """ Benchmark the simple model based on the Thames inflow.

    This benchmark uses external data
    """
    params = [
        [1, 2, 5, 10, 20, 40, 80, 160],
    ]
    param_names = [
        'number_of_scenarios',
    ]

    @staticmethod
    def load_climate_change_dataframe(slice=None):
        df = pandas.read_hdf(
            os.path.join(MODELS_DIR, "Thames @ Kingston - GLM - Climate Change Naturalised (Daily).h5"))
        if slice is not None:
            df = df.iloc[:, slice].copy()
        return df

    @staticmethod
    def add_variables(model, new_volume=0.0, desalination_capacity=0.0):
        from pywr.parameters import ConstantParameter, DataFrameParameter, AggregatedParameter
        # Make the variables scaled between 0 and 1.0
        # This ensures the EA has an easier time perturbing the variables
        new_volume = ConstantParameter(model, value=new_volume, scale=100e3, name='additional volume',
                                       lower_bounds=0.0, upper_bounds=1.0, is_variable=True, )

        total_volume = AggregatedParameter(model, [new_volume, ConstantParameter(model, 200e3)], agg_func='sum',
                                           name='total volume')

        model.nodes['reservoir1'].max_volume = total_volume

        desal_capacity = ConstantParameter(model, desalination_capacity, lower_bounds=0.0, upper_bounds=1.0,
                                           scale=2e3, is_variable=True, name='desalination capacity')
        model.nodes['desalination1'].max_flow = desal_capacity

    @staticmethod
    def add_recorders(model, deficit_agg_func='mean'):
        from pywr.recorders import TotalDeficitNodeRecorder

        total_deficit = TotalDeficitNodeRecorder(model, model.nodes["demand1"], name="total deficit",
                                                 agg_func=deficit_agg_func)

    def setup(self, number_of_scenarios, start_date='1970-01-01', end_date='1980-01-01'):

        with open(os.path.join(MODELS_DIR, 'simple_thames.json')) as fh:
            data = json.load(fh)

        # Remove any existing recorders
        if 'recorders' in data:
            del data['recorders']

        # Load the model
        m = Model.load(data, path=MODELS_DIR)

        m.timestepper.start = start_date
        m.timestepper.end = end_date

        if number_of_scenarios is not None:
            from pywr.core import Scenario
            from pywr.parameters import DataFrameParameter
            df = self.load_climate_change_dataframe(slice(0, number_of_scenarios))
            try:
                size = df.shape[1]
            except IndexError:
                size = 1

            climate_scenario = Scenario(m, name='climate change', size=size)
            m.nodes["catchment1"].flow = DataFrameParameter(m, df, scenario=climate_scenario)

        self.add_variables(m)
        self.add_recorders(m)

        m.check()
        m.setup()
        self.model = m

    def time_run(self, number_of_scenarios):
        self.model.run()
