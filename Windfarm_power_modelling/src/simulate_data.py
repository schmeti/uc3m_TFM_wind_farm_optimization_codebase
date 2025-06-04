
from floris import FlorisModel
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


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

            # Loop through all combinations of wind speeds, wind directions, and turbulence intensities
            for i, wind_direction in enumerate(np.unique(wind_directions)):
                for j, wind_speed in enumerate(np.unique(wind_speeds)):
                    for k, turbulence_intensity in enumerate(np.unique(turbulence_intensities)):
                        # Create a new row with the simulation results
                        new_row = pd.DataFrame({'x_turb2': [x_turb2], 
                                                'y_turb2': [y_turb2],
                                                'wind_speed': [wind_speed],
                                                'wind_direction': [wind_direction],
                                                'turbulence_intensity': [turbulence_intensity],
                                                'turbine1_power': [turbine_powers[i][0]],
                                                'turbine2_powers': [turbine_powers[i][1]], 
                                                'farm_power': [farm_power[i]]},
                                                dtype=dtype)
                        data = pd.concat([data, new_row], ignore_index=True)
    
    return data


def generate_wind_direction_distribution(mu=260, sd=10, wind_speed=8, turbulence_intensity=0.06, step=10, nquantile=None,epsilon = 1e-6, plot=True):
    if nquantile is not None:
        quantiles = np.linspace(epsilon, 1-epsilon, nquantile + 1)
        lower_bounds = norm.ppf(quantiles[:-1], loc=mu, scale=sd) % 360
        upper_bounds = norm.ppf(quantiles[1:], loc=mu, scale=sd) % 360
        angles = []
        for lb, ub in zip(lower_bounds, upper_bounds):
            if lb < ub:
                angle_range = np.linspace(lb, ub, 100)
            else:
                angle_range = np.linspace(lb, ub + 360, 100) % 360
            weights = norm.pdf(angle_range, loc=mu, scale=sd)
            expected_angle = np.angle(np.sum(np.exp(1j * np.deg2rad(angle_range)) * weights), deg=True) % 360 
            angles.append(expected_angle)
        angles = np.array(angles)
    else:
        angles = np.arange(0, 360, step)
    angles = angles[~np.isnan(angles)]

    def circular_dist(x, mu):
        d = np.abs(x - mu) % 360
        return np.minimum(d, 360 - d)

    distances = circular_dist(angles, mu)
    densities = norm.pdf(distances, loc=0, scale=sd)
    probabilities = densities / np.sum(densities)

    wind_df = pd.DataFrame({
        'wind_direction': angles,
        'probability': probabilities
    })

    wind_df.insert(0, 'x_turb2', np.nan)
    wind_df.insert(1, 'y_turb2', np.nan)
    wind_df['wind_speed'] = wind_speed
    wind_df['turbulence_intensity'] = turbulence_intensity
    wind_df = wind_df[wind_df['probability'] > 0.001].reset_index(drop=True) # remove very low probabilities

    # Move to last col
    prob_col = wind_df.pop('probability')
    wind_df['probability'] = prob_col

    # Plot
    if plot:
        plt.figure(figsize=(10, 5))
        if nquantile is not None:
            for lb, ub, prob in zip(lower_bounds, upper_bounds, wind_df['probability']):
                # Handle wrap-around at 360 degrees
                if lb < ub:
                    plt.bar(x=lb + (ub - lb) / 2, height=prob, width=(ub - lb), align='center',
                            color='lightgrey', edgecolor='black', linewidth=1)
                else:
                    # Bar wraps around 360
                    width1 = 360 - lb
                    width2 = ub
                    plt.bar(x=lb + width1 / 2, height=prob, width=width1, align='center',
                            color='lightgrey', edgecolor='black', linewidth=1)
                    plt.bar(x=ub / 2, height=prob, width=width2, align='center',
                            color='lightgrey', edgecolor='black', linewidth=1)
            for angle in angles:
                plt.axvline(angle, color='darkgrey', linestyle='--', alpha=0.7)
            
        else:
            bar_width = step
            plt.figure(figsize=(10, 5))
            plt.bar(wind_df['wind_direction'], wind_df['probability'], width=bar_width, align='center', color='lightgrey', edgecolor='black')

        plt.title(f'mean={mu}°, sd={sd}°')
        plt.xlabel('Wind Direction (Degrees)')
        plt.ylabel('Probability')
        plt.xticks(np.arange(0, 361, 30))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    return wind_df