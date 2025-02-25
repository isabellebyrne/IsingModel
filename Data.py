import numpy as np

class SingleSimData:
    """
    Object for storing data collected in a single simulation run at a given temperature. Data is collected after each step.
    1. Magnetizations: List
    2. Energies: List
    3. Heat Capacities: List
    4. Temperature: Float
    5. Steps (x-values): List
    """

    def __init__(self, temperature):
        self.magnetizations = []
        self.energies = []
        self.heat_capacities = []
        self.temperature = temperature

    def add_data(self, energy, magnetization, heat_capacity):
        """Add energy, magnetization, and heat capacity data for a single step."""
        self.magnetizations.append(magnetization)
        self.energies.append(energy)
        self.heat_capacities.append(heat_capacity)

    def get_data(self):
        """Retrieve collected data as a dictionary."""
        return {
            "magnetizations": self.magnetizations,
            "energies": self.energies,
            "heat_capacities": self.heat_capacities,
            "temperature": self.temperature,
            "n_steps": np.arange(len(self.magnetizations))
        }

    def get_data_at_step(self, step):
        """Retrieve data at a specific step."""
        return {
            "step": step,
            "magnetization": self.magnetizations[step],
            "energy": self.energies[step],
            "heat_capacity": self.heat_capacities[step]
        }


class MultiSimData:
    """
    Object for storing data collected over multiple simulation runs at different temperatures. Data is collected after each simulation is complete. 
    1. Magnetizations: List
    2. Energies: List
    3. Heat Capacities: List
    4. Temperatures (x-values): List
    """

    def __init__(self):
        self.temperatures = []
        self.magnetizations = []
        self.energies = []
        self.heat_capacities = []

    def add_data(self, temperature, energy, magnetization, heat_capacity):
        """ Add equlibrium energy, magnetization, and heat capacity data for a single simulation."""
        self.temperatures.append(temperature)
        self.magnetizations.append(magnetization)
        self.energies.append(energy)
        self.heat_capacities.append(heat_capacity)

    def get_data(self):
        """Retrieve collected data as a dictionary."""
        return {
            "temperatures": self.temperatures,
            "magnetizations": self.magnetizations,
            "energies": self.energies,
            "heat_capacities": self.heat_capacities
        }

    def get_data_at_temperature(self, temperature):
        """Retrieve data at a specific temperature."""
        index = np.argmin(np.abs(np.array(self.temperatures) - temperature))
        return {
            "temperature": self.temperatures[index],
            "magnetization": self.magnetizations[index],
            "energy": self.energies[index],
            "heat_capacity": self.heat_capacities[index]
        }
