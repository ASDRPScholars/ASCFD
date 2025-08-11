
import numpy as np
from skimage.segmentation import find_boundaries

def process_mask(mask):
    boundary = find_boundaries(mask, mode="inner").astype(int)
    inner = mask - boundary
    return inner, boundary

def get_grid_ids(X, Y):
    grid_ids = np.arange(Y * X).reshape(Y, X)
    return grid_ids

def get_selected_values(values, mask):
    assert values.shape == mask.shape
    nonzero_idx = np.nonzero(mask) # get mask 1
    return values[nonzero_idx]

def create_boundary_segments(region_mask, segment_definitions):
    """
    Create boundary segment mask from region mask and segment definitions.
    
    Args:
        region_mask: Boolean mask defining the computational domain
        segment_definitions: Dict mapping segment_id to boundary definition
            Each definition can be:
            - 'auto': automatically detect outer boundaries
            - {'side': 'left'/'right'/'top'/'bottom'}: for rectangular regions
            - {'x_range': (x0, x1), 'y_range': (y0, y1)}: for geometric regions
            - callable: function(x, y) -> bool for custom boundary detection
    
    Returns:
        boundary_segments: Integer mask with different values for each boundary segment
    """
    from skimage.segmentation import find_boundaries
    
    Y, X = region_mask.shape
    boundary_mask = find_boundaries(region_mask, mode="inner")
    boundary_segments = np.zeros_like(region_mask, dtype=int)
    
    # Get boundary coordinates for geometric definitions
    boundary_coords = np.where(boundary_mask)
    
    for segment_id, definition in segment_definitions.items():
        if definition == 'auto':
            # For single segment, mark all boundaries
            boundary_segments[boundary_mask] = segment_id
        elif isinstance(definition, dict):
            if 'side' in definition:
                # Handle rectangular side definitions
                side = definition['side']
                if side == 'left':
                    mask = (boundary_coords[1] == np.min(boundary_coords[1]))
                elif side == 'right':
                    mask = (boundary_coords[1] == np.max(boundary_coords[1]))
                elif side == 'top':
                    mask = (boundary_coords[0] == np.min(boundary_coords[0]))
                elif side == 'bottom':
                    mask = (boundary_coords[0] == np.max(boundary_coords[0]))
                else:
                    raise ValueError(f"Unknown side: {side}")
                boundary_segments[boundary_coords[0][mask], boundary_coords[1][mask]] = segment_id
                
            elif 'x_range' in definition or 'y_range' in definition:
                # Handle coordinate range definitions
                x_range = definition.get('x_range', (0, X))
                y_range = definition.get('y_range', (0, Y))
                
                mask = ((boundary_coords[1] >= x_range[0]) & (boundary_coords[1] <= x_range[1]) &
                       (boundary_coords[0] >= y_range[0]) & (boundary_coords[0] <= y_range[1]))
                boundary_segments[boundary_coords[0][mask], boundary_coords[1][mask]] = segment_id
                
        elif callable(definition):
            # Handle custom function definitions
            x_grid, y_grid = np.meshgrid(np.arange(X), np.arange(Y))
            custom_mask = definition(x_grid[boundary_mask], y_grid[boundary_mask])
            boundary_coords_custom = np.where(boundary_mask)
            selected_coords = (boundary_coords_custom[0][custom_mask], 
                             boundary_coords_custom[1][custom_mask])
            boundary_segments[selected_coords] = segment_id
    
    return boundary_segments

def create_hall_thruster_segments(region_mask, anode_x_range=None, cathode_x_range=None):
    """
    Convenience function to create boundary segments for hall thruster geometry.
    Assumes a rectangular domain with a rectangular channel cut-out on the left.
    
    Args:
        region_mask: Boolean mask defining the hall thruster domain
        anode_x_range: (x_min, x_max) range for anode boundary (default: leftmost boundary)
        cathode_x_range: (x_min, x_max) range for cathode boundary (default: rightmost boundary)
    
    Returns:
        boundary_segments: Integer mask with segments labeled as:
            1: anode (high voltage, typically 300V Dirichlet)
            2: cathode (ground, 0V Dirichlet) 
            3-8: walls (Neumann outflow/no-flux conditions)
    """
    from skimage.segmentation import find_boundaries
    
    Y, X = region_mask.shape
    boundary_mask = find_boundaries(region_mask, mode="inner")
    boundary_segments = np.zeros_like(region_mask, dtype=int)
    
    # Get boundary coordinates
    boundary_y, boundary_x = np.where(boundary_mask)
    
    # Auto-detect ranges if not provided
    if anode_x_range is None:
        anode_x_range = (np.min(boundary_x), np.min(boundary_x) + 5)  # Assume leftmost
    if cathode_x_range is None:
        cathode_x_range = (np.max(boundary_x) - 5, np.max(boundary_x))  # Assume rightmost
    
    # Segment 1: Anode (left boundary in typical hall thruster)
    anode_mask = ((boundary_x >= anode_x_range[0]) & (boundary_x <= anode_x_range[1]))
    boundary_segments[boundary_y[anode_mask], boundary_x[anode_mask]] = 1
    
    # Segment 2: Cathode (right boundary)  
    cathode_mask = ((boundary_x >= cathode_x_range[0]) & (boundary_x <= cathode_x_range[1]))
    boundary_segments[boundary_y[cathode_mask], boundary_x[cathode_mask]] = 2
    
    # Segments 3-8: Walls (remaining boundaries)
    wall_mask = ~(anode_mask | cathode_mask)
    boundary_segments[boundary_y[wall_mask], boundary_x[wall_mask]] = 3  # Can subdivide further if needed
    
    return boundary_segments

def visualize_boundary_segments(region_mask, boundary_segments, boundary_conditions=None):
    """
    Visualize boundary segments for debugging and verification.
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot region mask
    ax1.imshow(region_mask, cmap='gray', origin='lower')
    ax1.set_title('Region Mask')
    ax1.grid(True, alpha=0.3)
    
    # Plot boundary segments
    segments_plot = boundary_segments.copy().astype(float)
    segments_plot[~region_mask] = np.nan
    im = ax2.imshow(segments_plot, cmap='tab10', origin='lower')
    ax2.set_title('Boundary Segments')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, label='Segment ID')
    
    # Add boundary condition labels if provided
    if boundary_conditions:
        for segment_id, (value, mode) in boundary_conditions.items():
            ax2.text(0.02, 0.98 - 0.05*segment_id, f'Seg {segment_id}: {mode} = {value}', 
                    transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig