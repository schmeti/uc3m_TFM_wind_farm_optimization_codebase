def plot_two_turbine_results(data, zfeature='farm_power', model_opt=None, dpoint_size=None, ax=None):
    import matplotlib.pyplot as plt
    import pyomo.environ as pyo
    import numpy as np
    from scipy.interpolate import griddata

    x = data['x_turb2']
    y = data['y_turb2']
    z = data[zfeature]
    xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='linear')

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 6))
    else:
        fig = ax.figure

    contour = ax.contourf(xi, yi, zi, levels=100, cmap='viridis')
    cbar = fig.colorbar(contour, ax=ax, label='Total Farm Power', orientation='horizontal', pad=0.15, aspect=50, shrink=0.8)
    cbar.ax.set_xlabel('Power', labelpad=10)
    if dpoint_size is not None:
        ax.scatter(x, y, color='black', s=dpoint_size, label='Data Points', alpha=0.7, zorder=5)

    if model_opt is not None:
        circle = plt.Circle((0, 0), model_opt.min_dist, color='black', fill=False, linestyle='-', label='Constraints', linewidth=4)
        ax.add_artist(circle)
        ax.axhline(model_opt.y_max, color='black', linestyle='-', linewidth=4)
        ax.axvline(model_opt.x_max, color='black', linestyle='-', linewidth=4)
        ax.scatter(0, 0, color='red', label='Turbine 1', s=500, edgecolor='black', zorder=10)
        optimal_x = pyo.value(model_opt.x['x_turb2'])
        optimal_y = pyo.value(model_opt.x['y_turb2'])
        ax.scatter(optimal_x, optimal_y, color='blue', label='Optimal Turbine 2', s=500, edgecolor='black')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.legend(loc='upper right')
    ax.set_xlabel('x_turb2')
    ax.set_ylabel('y_turb2')

    if ax is None:
        return fig
