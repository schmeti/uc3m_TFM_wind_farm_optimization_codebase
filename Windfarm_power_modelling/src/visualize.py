def plot_two_turbine_results(data, zfeature='farm_power', model_opt=None, dpoint_size=None, wind_df = None, ax=None):
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
        
        if sum(1 for k in model_opt.x if k[0] == 'x_turb2') > 1:
            optimal_x = pyo.value(model_opt.x['x_turb2',0])
            optimal_y = pyo.value(model_opt.x['y_turb2',0])
        else:
            optimal_x = pyo.value(model_opt.x['x_turb2'])
            optimal_y = pyo.value(model_opt.x['y_turb2'])
        ax.scatter(optimal_x, optimal_y, color='blue', label='Optimal Turbine 2', s=500, edgecolor='black')
    
    if wind_df is not None:
        # Get current axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        wind_df = wind_df[wind_df['probability'] > 0.001] 

        for i, angle in enumerate(list(wind_df['wind_direction'].unique())):
            if (angle <= 90 and angle >= 0) or (angle <= 270 and angle >= 180):
                
                radians = np.deg2rad(90-angle)  
                
                # Calculate intersection points with plot boundaries
                # First try to intersect with right edge (x = xlim[1])
                x_right = xlim[1]
                y_right = x_right * np.tan(radians)
                
                # Then try to intersect with top edge (y = ylim[1])
                y_top = ylim[1]
                x_top = y_top / np.tan(radians) if np.tan(radians) != 0 else xlim[1]
                
                # Determine which intersection is within bounds
                if abs(y_right) <= ylim[1]:
                    # Use right edge intersection
                    x_end = x_right 
                    y_end = y_right
                    ha = 'left' if x_end > 0 else 'right'
                    va = 'center'
                else:
                    # Use top edge intersection
                    x_end = x_top
                    y_end = y_top
                    ha = 'center'
                    va = 'bottom' if y_end > 0 else 'top'



                # Plot the line, highlight the max probability angle in red
                max_prob_idx = wind_df['probability'].idxmax()
                max_angle = wind_df.loc[max_prob_idx, 'wind_direction']
                color = 'red' if angle == max_angle else 'gray'
                ax.plot([0, x_end], [0, y_end], linestyle='--', color=color)
                
                # Place the angle label at the end point
                ax.text(x_end+5, y_end, f'{round(angle, 1)}°', ha=ha, va=va, color='gray')
            
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.legend(loc='upper right')
    ax.set_xlabel('x_turb2')
    ax.set_ylabel('y_turb2')

    if ax is None:
        return fig
