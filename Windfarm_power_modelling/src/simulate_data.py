
from floris import FlorisModel
import pandas as pd

### Configure & Run Simulation
def two_turbine_simulation(fmodel: FlorisModel, 
                            x_turb2: int = None,
                            y_turb2: int = None,
                            wind_speeds: list = None,
                            wind_directions: list = None,
                            turbulence_intensities: list = None):
                            
    # Set the farm layout to have 8 turbines irregularly placed
    layout_x = [0, x_turb2]
    layout_y = [0, y_turb2]
    fmodel.set(layout_x=layout_x, layout_y=layout_y)

    # set wind
    fmodel.set(wind_speeds=wind_speeds, 
        wind_directions=wind_directions, 
        turbulence_intensities=turbulence_intensities)

    fmodel.run()

    ### Get Results
    turbine_powers = fmodel.get_turbine_powers() / 1000.0
    farm_power = fmodel.get_farm_power() / 1000.0

    return turbine_powers, farm_power


def two_turbine_simulation_data_generation(fmodel: FlorisModel,
                                            x_range: tuple, 
                                            y_range: tuple, 
                                            wind_speeds: list, 
                                            wind_directions: list, 
                                            turbulence_intensities: list, 
                                            dtype):
    
    # Create an empty DataFrame with the required columns and dtype
    data = pd.DataFrame(columns=['x_turb2', 'y_turb2', 'wind_speed', 'wind_direction', 
                                 'turbulence_intensity', 'turbine1_power', 'turbine2_powers', 'farm_power'],
                        dtype=dtype)

    # Loop over the given range for x_turb2 and y_turb2
    for x_turb2 in range(x_range[0], x_range[1], x_range[2]):
        for y_turb2 in range(y_range[0], y_range[1], y_range[2]):
            # Simulate the turbine powers and farm power
            turbine_powers, farm_power = two_turbine_simulation(fmodel, 
                                                                x_turb2=x_turb2, 
                                                                y_turb2=y_turb2,
                                                                wind_speeds=wind_speeds, 
                                                                wind_directions=wind_directions, 
                                                                turbulence_intensities=turbulence_intensities)

            # Create a new row with the simulation results
            new_row = pd.DataFrame({'x_turb2': [x_turb2], 
                                    'y_turb2': [y_turb2],
                                    'wind_speed': wind_speeds,
                                    'wind_direction': wind_directions,
                                    'turbulence_intensity': turbulence_intensities,
                                    'turbine1_power': turbine_powers[0][0],
                                    'turbine2_powers': turbine_powers[0][1], 
                                    'farm_power': farm_power[0]},
                                    dtype=dtype)
            data = pd.concat([data, new_row], ignore_index=True)
    
    return data




