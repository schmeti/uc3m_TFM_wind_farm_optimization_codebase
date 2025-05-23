import matplotlib.pyplot as plt
import pyomo.environ as pyo
import numpy as np

def plot_two_turbine_results(data, zfeature='farm_power', model_opt=None,dpoint_size = None):
    from scipy.interpolate import griddata

    x = data['x_turb2']
    y = data['y_turb2']
    z = data[zfeature]
    xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='linear')

    plt.figure(figsize=(20, 6))

    plt.contourf(xi, yi, zi, levels=100, cmap='viridis')
    cbar = plt.colorbar(label='Total Farm Power', orientation='horizontal', pad=-0.2, aspect=50, shrink=0.8)
    cbar.ax.set_xlabel('Power', labelpad=10)
    if dpoint_size is not None:
        plt.scatter(x, y, color='black', s=dpoint_size, label='Data Points', alpha=0.7, zorder=5)

    if model_opt is not None:
        circle = plt.Circle((0, 0), 100, color='black', fill=False, linestyle='-', label='Min Distance Constraint', linewidth=4)
        plt.gca().add_artist(circle)

        plt.scatter(0, 0, color='red', label='Turbine 1', s=500, edgecolor='black', zorder=10)
        optimal_x = pyo.value(model_opt.x['x_turb2'])
        optimal_y = pyo.value(model_opt.x['y_turb2'])
        plt.scatter(optimal_x, optimal_y, color='blue', label='Optimal Turbine 2', s=500, edgecolor='black')

    plt.axis('scaled')
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.9), ncol=3)
    plt.xlabel('x_turb2')
    plt.ylabel('y_turb2')
    plt.grid(True)
    plt.show()
