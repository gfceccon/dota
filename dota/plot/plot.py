import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union, Dict, Any
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings

# Set style
plt.style.use('default')
sns.set_palette("husl")


class DotaPlotter:
    """
    Comprehensive plotting class for Dota project with support for:
    - 2D and 3D scatter plots
    - Bar plots
    - Grid plots
    - Customizable layouts and styling
    """
    
    def __init__(
        self,
        figsize: Tuple[float, float] = (12, 8),
        style: str = 'seaborn-v0_8',
        color_palette: str = 'husl',
        dpi: int = 100
    ):
        """
        Initialize the DotaPlotter
        
        Args:
            figsize: Default figure size (width, height)
            style: Matplotlib style to use
            color_palette: Seaborn color palette
            dpi: Figure DPI for quality
        """
        self.figsize = figsize
        self.style = style
        self.color_palette = color_palette
        self.dpi = dpi
        
        # Set default styling
        try:
            plt.style.use(self.style)
        except OSError:
            plt.style.use('default')
            warnings.warn(f"Style '{self.style}' not found, using 'default'")
        
        sns.set_palette(self.color_palette)
    
    def scatter_2d(
        self,
        x: Union[List, np.ndarray, pd.Series],
        y: Union[List, np.ndarray, pd.Series],
        c: Optional[Union[List, np.ndarray, pd.Series]] = None,
        s: Union[int, List, np.ndarray] = 50,
        alpha: float = 0.7,
        title: str = "2D Scatter Plot",
        xlabel: str = "X axis",
        ylabel: str = "Y axis",
        colorbar_label: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        show_grid: bool = True,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Create a 2D scatter plot
        
        Args:
            x, y: Data for x and y axes
            c: Colors for points (optional)
            s: Size of points
            alpha: Transparency
            title: Plot title
            xlabel, ylabel: Axis labels
            colorbar_label: Label for colorbar if c is provided
            figsize: Figure size override
            save_path: Path to save the plot
            show_grid: Whether to show grid
            **kwargs: Additional arguments for scatter plot
        
        Returns:
            Tuple of (figure, axes)
        """
        fig_size = figsize or self.figsize
        fig, ax = plt.subplots(figsize=fig_size, dpi=self.dpi)
        
        scatter = ax.scatter(x, y, c=c, s=s, alpha=alpha, **kwargs)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        # Add colorbar if colors are provided
        if c is not None:
            cbar = plt.colorbar(scatter, ax=ax)
            if colorbar_label:
                cbar.set_label(colorbar_label, fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def scatter_3d(
        self,
        x: Union[List, np.ndarray, pd.Series],
        y: Union[List, np.ndarray, pd.Series],
        z: Union[List, np.ndarray, pd.Series],
        c: Optional[Union[List, np.ndarray, pd.Series]] = None,
        size: Union[int, List, np.ndarray] = 50,
        alpha: float = 0.7,
        title: str = "3D Scatter Plot",
        xlabel: str = "X axis",
        ylabel: str = "Y axis",
        zlabel: str = "Z axis",
        colorbar_label: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        elev: float = 20,
        azim: float = 45,
        **kwargs
    ) -> Tuple[Figure, Any]:
        """
        Create a 3D scatter plot
        
        Args:
            x, y, z: Data for x, y, and z axes
            c: Colors for points (optional)
            size: Size of points
            alpha: Transparency
            title: Plot title
            xlabel, ylabel, zlabel: Axis labels
            colorbar_label: Label for colorbar if c is provided
            figsize: Figure size override
            save_path: Path to save the plot
            elev, azim: 3D view angles
            **kwargs: Additional arguments for scatter plot
        
        Returns:
            Tuple of (figure, 3d_axes)
        """
        fig_size = figsize or self.figsize
        fig = plt.figure(figsize=fig_size, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Filter out conflicting parameters
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in ['s', 'c', 'alpha']}
        
        scatter = ax.scatter(x, y, z, alpha=alpha, **plot_kwargs)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # Use getattr to safely access 3D methods
        if hasattr(ax, 'set_zlabel'):
            getattr(ax, 'set_zlabel')(zlabel, fontsize=12)
        
        # Set viewing angle
        if hasattr(ax, 'view_init'):
            getattr(ax, 'view_init')(elev=elev, azim=azim)
        
        # Add colorbar if colors are provided
        if c is not None:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=30)
            if colorbar_label:
                cbar.set_label(colorbar_label, fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def bar_plot(
        self,
        categories: Union[List, np.ndarray, pd.Series],
        values: Union[List, np.ndarray, pd.Series],
        colors: Optional[Union[str, List]] = None,
        title: str = "Bar Plot",
        xlabel: str = "Categories",
        ylabel: str = "Values",
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        horizontal: bool = False,
        show_values: bool = True,
        rotation: int = 45,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Create a bar plot
        
        Args:
            categories: Category names
            values: Values for each category
            colors: Colors for bars
            title: Plot title
            xlabel, ylabel: Axis labels
            figsize: Figure size override
            save_path: Path to save the plot
            horizontal: Whether to create horizontal bar plot
            show_values: Whether to show values on bars
            rotation: Rotation angle for category labels
            **kwargs: Additional arguments for bar plot
        
        Returns:
            Tuple of (figure, axes)
        """
        fig_size = figsize or self.figsize
        fig, ax = plt.subplots(figsize=fig_size, dpi=self.dpi)
        
        if horizontal:
            bars = ax.barh(categories, values, color=colors, **kwargs)
            ax.set_xlabel(ylabel, fontsize=12)
            ax.set_ylabel(xlabel, fontsize=12)
        else:
            bars = ax.bar(categories, values, color=colors, **kwargs)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            
            # Rotate labels if needed
            if rotation != 0:
                ax.tick_params(axis='x', rotation=rotation)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y' if not horizontal else 'x')
        
        # Add value labels on bars
        if show_values:
            for bar in bars:
                if horizontal:
                    width = bar.get_width()
                    ax.text(width + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                           f'{width:.1f}', ha='left', va='center', fontsize=10)
                else:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + max(values) * 0.01,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def stacked_bar_plot(
        self,
        categories: Union[List, np.ndarray, pd.Series],
        y1: Union[List, np.ndarray, pd.Series],
        y2: Union[List, np.ndarray, pd.Series],
        colors: Optional[Tuple[str, str]] = None,
        labels: Optional[Tuple[str, str]] = None,
        title: str = "Stacked Bar Plot",
        xlabel: str = "Categories",
        ylabel: str = "Values",
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        horizontal: bool = False,
        show_values: bool = True,
        show_legend: bool = True,
        rotation: int = 45,
        invert_colors: bool = False,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Create a stacked bar plot with y2 stacked on top of y1
        
        Args:
            categories: Category names
            y1: Values for bottom bars
            y2: Values for top bars (stacked on y1)
            colors: Tuple of colors for (y1, y2) bars
            labels: Tuple of labels for (y1, y2) in legend
            title: Plot title
            xlabel, ylabel: Axis labels
            figsize: Figure size override
            save_path: Path to save the plot
            horizontal: Whether to create horizontal stacked bar plot
            show_values: Whether to show values on bars
            show_legend: Whether to show legend
            rotation: Rotation angle for category labels
            **kwargs: Additional arguments for bar plot
        
        Returns:
            Tuple of (figure, axes)
        """
        fig_size = figsize or self.figsize
        fig, ax = plt.subplots(figsize=fig_size, dpi=self.dpi)
        
        # Default colors and labels
        default_colors = ('#3498db', '#e74c3c')  # Blue and red
        colors = colors or default_colors
        if(invert_colors):
            colors = (colors[1], colors[0])
        labels = labels or ('Bottom', 'Top')
        
        if horizontal:
            # Horizontal stacked bars
            bars1 = ax.barh(categories, y1, color=colors[0], label=labels[0], **kwargs)
            bars2 = ax.barh(categories, y2, left=y1, color=colors[1], label=labels[1], **kwargs)
            ax.set_xlabel(ylabel, fontsize=12)
            ax.set_ylabel(xlabel, fontsize=12)
            
            # Add value labels if specified
            if show_values:
                for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                    # Label for y1 (bottom/left part)
                    width1 = bar1.get_width()
                    ax.text(width1/2, bar1.get_y() + bar1.get_height()/2,
                           f'{y1[i]:.1f}', ha='center', va='center', fontsize=9, 
                           color='white', fontweight='bold')
                    
                    # Label for y2 (top/right part)
                    width2 = bar2.get_width()
                    ax.text(width1 + width2/2, bar2.get_y() + bar2.get_height()/2,
                           f'{y2[i]:.1f}', ha='center', va='center', fontsize=9, 
                           color='white', fontweight='bold')
        else:
            # Vertical stacked bars
            bars1 = ax.bar(categories, y1, color=colors[0], label=labels[0], **kwargs)
            bars2 = ax.bar(categories, y2, bottom=y1, color=colors[1], label=labels[1], **kwargs)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            
            # Rotate labels if needed
            if rotation != 0:
                ax.tick_params(axis='x', rotation=rotation)
            
            # Add value labels if specified
            if show_values:
                for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                    # Label for y1 (bottom part)
                    height1 = bar1.get_height()
                    ax.text(bar1.get_x() + bar1.get_width()/2, height1/2,
                           f'{y1[i]:.1f}', ha='center', va='center', fontsize=9, 
                           color='white', fontweight='bold')
                    
                    # Label for y2 (top part)
                    height2 = bar2.get_height()
                    ax.text(bar2.get_x() + bar2.get_width()/2, height1 + height2/2,
                           f'{y2[i]:.1f}', ha='center', va='center', fontsize=9, 
                           color='white', fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y' if not horizontal else 'x')
        
        # Add legend
        if show_legend:
            # Ajusta a posição da legenda para ter margem à esquerda
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), frameon=True, facecolor='white', edgecolor='gray')
            # Expande o background do eixo para a direita para acomodar a legenda
            box = ax.get_position()
            ax.set_position((box.x0, box.y0, box.width * 0.85, box.height))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def grid_plot(
        self,
        data: Union[Dict[str, Tuple], List[Tuple]],
        plot_type: str = 'scatter',
        grid_shape: Optional[Tuple[int, int]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: str = "Grid Plot",
        save_path: Optional[str] = None,
        share_x: bool = False,
        share_y: bool = False,
        **kwargs
    ) -> Tuple[Figure, np.ndarray]:
        """
        Create a grid of plots
        
        Args:
            data: Dictionary or list of plot data
                  Format: {'plot_name': (x, y, title), ...} or [(x, y, title), ...]
            plot_type: Type of plot ('scatter', 'line', 'bar')
            grid_shape: (rows, cols) for grid layout
            figsize: Figure size override
            title: Overall title
            save_path: Path to save the plot
            share_x, share_y: Whether to share axes
            **kwargs: Additional arguments for individual plots
        
        Returns:
            Tuple of (figure, axes_array)
        """
        # Convert data to list format if it's a dict
        if isinstance(data, dict):
            plot_data = list(data.values())
            plot_titles = list(data.keys())
        else:
            plot_data = data
            plot_titles = [f"Plot {i+1}" for i in range(len(data))]
        
        n_plots = len(plot_data)
        
        # Determine grid shape
        if grid_shape is None:
            cols = int(np.ceil(np.sqrt(n_plots)))
            rows = int(np.ceil(n_plots / cols))
        else:
            rows, cols = grid_shape
        
        fig_size = figsize or (self.figsize[0] * cols / 2, self.figsize[1] * rows / 2)
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, dpi=self.dpi,
                                sharex=share_x, sharey=share_y)
        
        # Handle single subplot case
        if n_plots == 1:
            axes = np.array([axes])
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        
        # Create plots
        for i, (plot_data_item, plot_title) in enumerate(zip(plot_data, plot_titles)):
            if i >= rows * cols:
                break
            
            # Get the correct axis
            if rows == 1 and cols == 1:
                ax = axes[0]
            elif rows == 1 or cols == 1:
                ax = axes[i]
            else:
                row, col = divmod(i, cols)
                ax = axes[row, col]
            
            # Extract data
            if len(plot_data_item) >= 3:
                x, y, subplot_title = plot_data_item[0], plot_data_item[1], plot_data_item[2]
            elif len(plot_data_item) >= 2:
                x, y = plot_data_item[0], plot_data_item[1]
                subplot_title = plot_title
            else:
                continue  # Skip invalid data
            
            # Create plot based on type
            if plot_type == 'scatter':
                ax.scatter(x, y, alpha=0.7, **kwargs)
            elif plot_type == 'line':
                ax.plot(x, y, **kwargs)
            elif plot_type == 'bar':
                ax.bar(x, y, **kwargs)
            
            ax.set_title(subplot_title, fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        total_subplots = rows * cols
        if n_plots < total_subplots:
            for i in range(n_plots, total_subplots):
                if rows == 1 or cols == 1:
                    axes[i].set_visible(False)
                else:
                    row, col = divmod(i, cols)
                    axes[row, col].set_visible(False)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, axes


class DotaPlotter3Col(DotaPlotter):
    """
    Extended plotting class with 3-column layout specialization
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_cols = 3
    
    def multi_scatter_2d(
        self,
        datasets: List[Dict[str, Any]],
        titles: Optional[List[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        overall_title: str = "Multiple 2D Scatter Plots"
    ) -> Tuple[Figure, np.ndarray]:
        """
        Create multiple 2D scatter plots in a 3-column layout
        
        Args:
            datasets: List of dictionaries containing plot data
                     Format: [{'x': x_data, 'y': y_data, 'c': colors, ...}, ...]
            titles: List of subplot titles
            figsize: Figure size override
            save_path: Path to save the plot
            overall_title: Overall figure title
        
        Returns:
            Tuple of (figure, axes_array)
        """
        n_plots = len(datasets)
        rows = int(np.ceil(n_plots / self.default_cols))
        cols = min(n_plots, self.default_cols)
        
        fig_size = figsize or (15, 5 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, dpi=self.dpi)
        
        # Handle single row case
        if rows == 1:
            axes = axes.reshape(1, -1) if n_plots > 1 else np.array([[axes]])
        
        plot_titles = titles or [f"Dataset {i+1}" for i in range(n_plots)]
        
        for i, (dataset, title) in enumerate(zip(datasets, plot_titles)):
            row, col = divmod(i, self.default_cols)
            ax = axes[row, col]
            
            # Extract data with defaults
            x = dataset['x']
            y = dataset['y']
            c = dataset.get('c', None)
            s = dataset.get('s', 50)
            alpha = dataset.get('alpha', 0.7)
            
            scatter = ax.scatter(x, y, c=c, s=s, alpha=alpha)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(dataset.get('xlabel', 'X axis'))
            ax.set_ylabel(dataset.get('ylabel', 'Y axis'))
            ax.grid(True, alpha=0.3)
            
            # Add colorbar if colors are provided
            if c is not None and dataset.get('colorbar', False):
                cbar = plt.colorbar(scatter, ax=ax)
                if 'colorbar_label' in dataset:
                    cbar.set_label(dataset['colorbar_label'])
        
        # Hide unused subplots
        total_subplots = rows * cols
        if n_plots < total_subplots:
            for i in range(n_plots, total_subplots):
                row, col = divmod(i, self.default_cols)
                axes[row, col].set_visible(False)
        
        fig.suptitle(overall_title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, axes
    
    def multi_bar_plot(
        self,
        datasets: List[Dict[str, Any]],
        titles: Optional[List[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        overall_title: str = "Multiple Bar Plots"
    ) -> Tuple[Figure, np.ndarray]:
        """
        Create multiple bar plots in a 3-column layout
        
        Args:
            datasets: List of dictionaries containing plot data
                     Format: [{'categories': cats, 'values': vals, ...}, ...]
            titles: List of subplot titles
            figsize: Figure size override
            save_path: Path to save the plot
            overall_title: Overall figure title
        
        Returns:
            Tuple of (figure, axes_array)
        """
        n_plots = len(datasets)
        rows = int(np.ceil(n_plots / self.default_cols))
        cols = min(n_plots, self.default_cols)
        
        fig_size = figsize or (15, 5 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, dpi=self.dpi)
        
        # Handle single row case
        if rows == 1:
            axes = axes.reshape(1, -1) if n_plots > 1 else np.array([[axes]])
        
        plot_titles = titles or [f"Dataset {i+1}" for i in range(n_plots)]
        
        for i, (dataset, title) in enumerate(zip(datasets, plot_titles)):
            row, col = divmod(i, self.default_cols)
            ax = axes[row, col]
            
            # Extract data
            categories = dataset['categories']
            values = dataset['values']
            colors = dataset.get('colors', None)
            horizontal = dataset.get('horizontal', False)
            
            if horizontal:
                bars = ax.barh(categories, values, color=colors)
                ax.set_xlabel(dataset.get('ylabel', 'Values'))
                ax.set_ylabel(dataset.get('xlabel', 'Categories'))
            else:
                bars = ax.bar(categories, values, color=colors)
                ax.set_xlabel(dataset.get('xlabel', 'Categories'))
                ax.set_ylabel(dataset.get('ylabel', 'Values'))
                
                # Rotate labels if specified
                rotation = dataset.get('rotation', 45)
                if rotation != 0:
                    ax.tick_params(axis='x', rotation=rotation)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y' if not horizontal else 'x')
            
            # Add value labels if specified
            if dataset.get('show_values', True):
                for bar in bars:
                    if horizontal:
                        width = bar.get_width()
                        ax.text(width + max(values) * 0.01, 
                               bar.get_y() + bar.get_height()/2,
                               f'{width:.1f}', ha='left', va='center', fontsize=9)
                    else:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, 
                               height + max(values) * 0.01,
                               f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Hide unused subplots
        total_subplots = rows * cols
        if n_plots < total_subplots:
            for i in range(n_plots, total_subplots):
                row, col = divmod(i, self.default_cols)
                axes[row, col].set_visible(False)
        
        fig.suptitle(overall_title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, axes
    
    def multi_stacked_bar_plot(
        self,
        datasets: List[Dict[str, Any]],
        titles: Optional[List[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        overall_title: str = "Multiple Stacked Bar Plots"
    ) -> Tuple[Figure, np.ndarray]:
        """
        Create multiple stacked bar plots in a 3-column layout
        
        Args:
            datasets: List of dictionaries containing plot data
                     Format: [{'categories': cats, 'y1': vals1, 'y2': vals2, ...}, ...]
            titles: List of subplot titles
            figsize: Figure size override
            save_path: Path to save the plot
            overall_title: Overall figure title
        
        Returns:
            Tuple of (figure, axes_array)
        """
        n_plots = len(datasets)
        rows = int(np.ceil(n_plots / self.default_cols))
        cols = min(n_plots, self.default_cols)
        
        fig_size = figsize or (15, 5 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, dpi=self.dpi)
        
        # Handle single row case
        if rows == 1:
            axes = axes.reshape(1, -1) if n_plots > 1 else np.array([[axes]])
        
        plot_titles = titles or [f"Dataset {i+1}" for i in range(n_plots)]
        
        for i, (dataset, title) in enumerate(zip(datasets, plot_titles)):
            row, col = divmod(i, self.default_cols)
            ax = axes[row, col]
            
            # Extract data
            categories = dataset['categories']
            y1 = dataset['y1']
            y2 = dataset['y2']
            colors = dataset.get('colors', ('#3498db', '#e74c3c'))
            labels = dataset.get('labels', ('Bottom', 'Top'))
            horizontal = dataset.get('horizontal', False)
            invert_colors = dataset.get('invert_colors', False)
            show_values = dataset.get('show_values', True)
            show_legend = dataset.get('show_legend', True)
            
            if invert_colors:
                colors = (colors[1], colors[0])
            
            if horizontal:
                # Horizontal stacked bars
                bars1 = ax.barh(categories, y1, color=colors[0], label=labels[0])
                bars2 = ax.barh(categories, y2, left=y1, color=colors[1], label=labels[1])
                ax.set_xlabel(dataset.get('ylabel', 'Values'))
                ax.set_ylabel(dataset.get('xlabel', 'Categories'))
                
                # Add value labels if specified
                if show_values:
                    for j, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                        # Label for y1 (bottom/left part)
                        width1 = bar1.get_width()
                        ax.text(width1/2, bar1.get_y() + bar1.get_height()/2,
                               f'{y1[j]:.1f}', ha='center', va='center', fontsize=8, 
                               color='white', fontweight='bold')
                        
                        # Label for y2 (top/right part)
                        width2 = bar2.get_width()
                        ax.text(width1 + width2/2, bar2.get_y() + bar2.get_height()/2,
                               f'{y2[j]:.1f}', ha='center', va='center', fontsize=8, 
                               color='white', fontweight='bold')
            else:
                # Vertical stacked bars
                bars1 = ax.bar(categories, y1, color=colors[0], label=labels[0])
                bars2 = ax.bar(categories, y2, bottom=y1, color=colors[1], label=labels[1])
                ax.set_xlabel(dataset.get('xlabel', 'Categories'))
                ax.set_ylabel(dataset.get('ylabel', 'Values'))
                
                # Rotate labels if specified
                rotation = dataset.get('rotation', 45)
                if rotation != 0:
                    ax.tick_params(axis='x', rotation=rotation)
                
                # Add value labels if specified
                if show_values:
                    for j, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                        # Label for y1 (bottom part)
                        height1 = bar1.get_height()
                        ax.text(bar1.get_x() + bar1.get_width()/2, height1/2,
                               f'{y1[j]:.1f}', ha='center', va='center', fontsize=8, 
                               color='white', fontweight='bold')
                        
                        # Label for y2 (top part)
                        height2 = bar2.get_height()
                        ax.text(bar2.get_x() + bar2.get_width()/2, height1 + height2/2,
                               f'{y2[j]:.1f}', ha='center', va='center', fontsize=8, 
                               color='white', fontweight='bold')
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y' if not horizontal else 'x')
            
            # Add legend (compact for subplots)
            if show_legend:
                ax.legend(loc='upper right', fontsize=8)
        
        # Hide unused subplots
        total_subplots = rows * cols
        if n_plots < total_subplots:
            for i in range(n_plots, total_subplots):
                row, col = divmod(i, self.default_cols)
                axes[row, col].set_visible(False)
        
        fig.suptitle(overall_title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, axes
    
    def comparison_plot(
        self,
        data_2d: List[Dict[str, Any]],
        data_3d: List[Dict[str, Any]],
        data_bar: List[Dict[str, Any]],
        titles: Optional[List[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        overall_title: str = "Comparison: 2D, 3D, and Bar Plots"
    ) -> Tuple[Figure, Any]:
        """
        Create a comparison plot with 2D scatter, 3D scatter, and bar plots
        
        Args:
            data_2d: List of 2D scatter plot data
            data_3d: List of 3D scatter plot data
            data_bar: List of bar plot data
            titles: List of subplot titles
            figsize: Figure size override
            save_path: Path to save the plot
            overall_title: Overall figure title
        
        Returns:
            Tuple of (figure, axes_array)
        """
        n_plots = len(data_2d) + len(data_3d) + len(data_bar)
        rows = int(np.ceil(n_plots / self.default_cols))
        
        fig_size = figsize or (15, 5 * rows)
        fig = plt.figure(figsize=fig_size, dpi=self.dpi)
        
        plot_titles = titles or []
        plot_idx = 0
        
        # 2D scatter plots
        for i, dataset in enumerate(data_2d):
            ax = fig.add_subplot(rows, self.default_cols, plot_idx + 1)
            
            x, y = dataset['x'], dataset['y']
            c = dataset.get('c', None)
            
            scatter = ax.scatter(x, y, c=c, alpha=0.7)
            
            title = plot_titles[plot_idx] if plot_idx < len(plot_titles) else f"2D Plot {i+1}"
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if c is not None:
                plt.colorbar(scatter, ax=ax)
            
            plot_idx += 1
        
        # 3D scatter plots
        for i, dataset in enumerate(data_3d):
            ax = fig.add_subplot(rows, self.default_cols, plot_idx + 1, projection='3d')
            
            x, y, z = dataset['x'], dataset['y'], dataset['z']
            c = dataset.get('c', None)
            
            scatter = ax.scatter(x, y, z, c=c, alpha=0.7)
            
            title = plot_titles[plot_idx] if plot_idx < len(plot_titles) else f"3D Plot {i+1}"
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            if c is not None:
                plt.colorbar(scatter, ax=ax, shrink=0.5)
            
            plot_idx += 1
        
        # Bar plots
        for i, dataset in enumerate(data_bar):
            ax = fig.add_subplot(rows, self.default_cols, plot_idx + 1)
            
            categories, values = dataset['categories'], dataset['values']
            colors = dataset.get('colors', None)
            
            ax.bar(categories, values, color=colors)
            
            title = plot_titles[plot_idx] if plot_idx < len(plot_titles) else f"Bar Plot {i+1}"
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            rotation = dataset.get('rotation', 45)
            if rotation != 0:
                ax.tick_params(axis='x', rotation=rotation)
            
            plot_idx += 1
        
        fig.suptitle(overall_title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, fig.get_axes()


# Convenience functions
def quick_scatter_2d(x, y, title="Quick 2D Scatter", **kwargs):
    """Quick 2D scatter plot"""
    plotter = DotaPlotter()
    return plotter.scatter_2d(x, y, title=title, **kwargs)


def quick_scatter_3d(x, y, z, title="Quick 3D Scatter", **kwargs):
    """Quick 3D scatter plot"""
    plotter = DotaPlotter()
    return plotter.scatter_3d(x, y, z, title=title, **kwargs)


def quick_bar(categories, values, title="Quick Bar Plot", **kwargs):
    """Quick bar plot"""
    plotter = DotaPlotter()
    return plotter.bar_plot(categories, values, title=title, **kwargs)


def quick_stacked_bar(categories, y1, y2, title="Quick Stacked Bar Plot", **kwargs):
    """Quick stacked bar plot"""
    plotter = DotaPlotter()
    return plotter.stacked_bar_plot(categories, y1, y2, title=title, **kwargs)


# Default plotter instances
plotter = DotaPlotter()
plotter_3col = DotaPlotter3Col()