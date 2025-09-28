import numpy as np

def tessellation_protocol(dim: int) -> tuple[np.ndarray, int]:
    """Tessellation protocol.

    This function determines the tessellation protocol and the number of
    tessellations involved in that protocol. Each of these are sensitive
    to the physical dimensionality of the network.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
    
    Returns:
        tuple[np.ndarray, int]: 2D np.ndarray with (3**dim, dim) int
        entries representing the tessellation protocol and the number of
        tessellations involved in that protocol, respectively.
    
    """
    base_tsslltn = np.asarray([-1, 0, 1], dtype=int)
    tsslltn = np.asarray(np.meshgrid(*(base_tsslltn,)*dim)).T.reshape(-1, dim)
    return tsslltn, np.shape(tsslltn)[0]

def tessellation(
        coords: np.ndarray,
        tsslltn: np.ndarray,
        L: np.ndarray) -> np.ndarray:
    """Tessellation.
    
    This function fully tessellates (or translates) an arbitrary
    coordinate in each cardinal direction in the spatial plane via a
    scaling L.

    Args:
        coords (np.ndarray): 2D np.ndarray with (en, dim) float entries of the coordinates to be tessellated.
        tsslltn (np.ndarray): 2D np.ndarray with (3**dim, dim) int entries representing the tessellation protocol.
        L (np.ndarray): 1D np.ndarray with dim float entries of the tessellation scaling.
    
    Returns:
        np.ndarray: 2D np.ndarray with (en*3**dim, dim) float entries of
        the tessellated coordinates.
    
    """
    return coords + tsslltn * L

def core_clnkr_tessellation(
        dim: int,
        core_clnkrs: np.ndarray,
        core_coords: np.ndarray,
        L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Core cross-linker tessellation.
    
    This function fully tessellates (or translates) an arbitrary set of
    coordinates in each cardinal direction in the spatial plane via a
    scaling L. The initially provided coordinates are associated with
    the core cross-linkers. The tessellated coordinates are stored after
    the initially provided coordinates of the core cross-linkers.
    Additionally, an np.ndarray is created that identifies which core
    cross-linker it represents in the tessellated configuration.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        core_clnkrs (np.ndarray): 1D np.ndarray with en int entries of the core cross-linkers.
        core_coords (np.ndarray): 2D np.ndarray with (en, dim) float entries of the core cross-linker coordinates to be tessellated.
        L (np.ndarray): 1D np.ndarray with dim float entries of the tessellation scaling.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: 2D np.ndarray with
        (en*3**dim, dim) float entries of the tessellated cross-linker
        coordinates and 1D np.ndarray with en*3**dim int entries that
        returns the core cross-linker that corresponds to each core and
        periodic boundary cross-linker, i.e.,
        pb_to_core_clnkrs[core_pb_clnkr] = core_clnkr.
    
    """
    if (np.shape(core_clnkrs)[0] != np.shape(core_coords)[0]) or (dim != np.shape(core_coords)[1]):
        error_str = (
            "Either the number of core cross-linkers does not match "
            + "the number of core cross-linker coordinates or the "
            + "specified network dimension does not match the "
            + "dimensionality of the core cross-linker coordinates. "
            + "This calculation will only proceed if both of those "
            + "conditions are satisfied. Please modify accordingly."
        )
        raise ValueError(error_str)
    else:
        # Tessellation protocol
        tsslltn, tsslltn_num = tessellation_protocol(dim)

        # Copy the coordinates as the first n entries in the tessellated
        # coordinate np.ndarray
        tsslltd_core_coords = core_coords.copy()

        # Tessellate the core cross-linkers
        for tsslltn_actn in range(tsslltn_num):
            # Skip the zero tessellation call because the core
            # cross-linkers are being tessellated about themselves
            if np.array_equal(tsslltn[tsslltn_actn], np.zeros(dim)) == True:
                continue
            else:
                tsslltd_core_coords = np.vstack(
                    (tsslltd_core_coords, tessellation(core_coords, tsslltn[tsslltn_actn], L)))

        # Construct the pb_to_core_clnkrs np.ndarray such that
        # pb_to_core_clnkrs[core_pb_clnkr] = core_clnkr
        pb_to_core_clnkrs = np.tile(core_clnkrs, tsslltn_num)

        return tsslltd_core_coords, pb_to_core_clnkrs

def core_to_pb_clnkrs_func(
        core_clnkrs: np.ndarray,
        pb_to_core_clnkrs: np.ndarray) -> list[np.ndarray]:
    """List of np.ndarrays corresponding to the periodic boundary
    cross-linkers associated with a particular core cross-linker.

    This function creates a list of np.ndarrays corresponding to the
    periodic boundary cross-linkers associated with a particular core
    cross-linker such that core_to_pb_clnkrs[core_clnkr] = pb_clnkrs.

    Args:
        core_clnkrs (np.ndarray): 1D np.ndarray with en int entries of the core cross-linkers.
        pb_to_core_clnkrs (np.ndarray): 1D np.ndarray with en*3**dim int entries that returns the core cross-linker that corresponds to each core and periodic boundary cross-linker, i.e., pb_to_core_clnkrs[core_pb_clnkr] = core_clnkr.
    
    Returns:
        list[np.ndarray]: List of 1D np.ndarrays with int entries
        corresponding to the periodic boundary cross-linkers associated
        with a particular core cross-linker.

    """
    core_to_pb_clnkrs = []
    for core_clnkr in np.nditer(core_clnkrs):
        # Isolate core and periodic boundary cross-linkers associated
        # with the core cross-linker, and delete the core cross-linker
        pb_clnkrs = np.where(pb_to_core_clnkrs == int(core_clnkr))[0][1:]
        core_to_pb_clnkrs.append(pb_clnkrs)
    return core_to_pb_clnkrs

def box_neighborhood_id(
        dim: int,
        coords: np.ndarray,
        coord: np.ndarray,
        l: float,
        inclusive: bool) -> tuple[np.ndarray, int]:
    """Box neighborhood identification.
    
    This function return the coordinates or the indices of the
    coordinates that lie within a box neighborhood that is \\pm
    half-side length l about a given coordinate.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        coords (np.ndarray): 2D np.ndarray of (en, dim) float entries of coordinates that may or may not reside in the box neighborhood.
        coord (np.ndarray): 1D np.ndarray of dim float entries of a given coordinate which the box neighborhood is defined about.
        l (float): Half-side length defining the box neighborhood about the given coordinate.
        inclusive (bool): Boolean indicating if the box neighborhood is inclusive or exclusive of its boundary.
    
    Returns:
        tuple[np.ndarray, int]: 1D np.ndarray of en int entries of box
        neighbor indices and the number of box neighbors.

    """
    # Extract the x- and y-coordinates of the box center point and the
    # candidate points
    x_coord = coord[0]
    y_coord = coord[1]
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # Define the boundary of the box neighborhood
    box_nghbr_x_lb = x_coord - l
    box_nghbr_x_ub = x_coord + l
    box_nghbr_y_lb = y_coord - l
    box_nghbr_y_ub = y_coord + l

    # Determine which candidate points are box neighbors
    if inclusive:
        box_nghbrs = np.logical_and(
            np.logical_and(x_coords>=box_nghbr_x_lb, x_coords<=box_nghbr_x_ub),
            np.logical_and(y_coords>=box_nghbr_y_lb, y_coords<=box_nghbr_y_ub))
    else:
        box_nghbrs = np.logical_and(
            np.logical_and(x_coords>box_nghbr_x_lb, x_coords<box_nghbr_x_ub),
            np.logical_and(y_coords>box_nghbr_y_lb, y_coords<box_nghbr_y_ub))

    if dim == 3:
        # Extract the z-coordinates of the box center point and the
        # candidate points
        z_coord = coord[2]
        z_coords = coords[:, 2]

        # Define the boundary of the box neighborhood
        box_nghbr_z_lb = z_coord - l
        box_nghbr_z_ub = z_coord + l

        # Determine which candidate points are box neighbors
        if inclusive:
            box_nghbrs = np.logical_and(
                box_nghbrs,
                np.logical_and(z_coords>=box_nghbr_z_lb, z_coords<=box_nghbr_z_ub))
        else:
            box_nghbrs = np.logical_and(
                box_nghbrs,
                np.logical_and(z_coords>box_nghbr_z_lb, z_coords<box_nghbr_z_ub))
    
    # Determine the indices of the box neighbors, and calculate the
    # number of box neighbors
    box_nghbr_indcs = np.where(box_nghbrs)[0]
    box_nghbr_num = np.shape(box_nghbr_indcs)[0]

    # Box neighborhood is empty
    if box_nghbr_num == 0: return np.asarray([], dtype=int), 0
    # Box neighborhood has at least one neighbor in it
    else: return box_nghbr_indcs, box_nghbr_num

def orb_neighborhood_id(
        dim: int,
        coords: np.ndarray,
        coord: np.ndarray,
        r: float,
        inclusive: bool) -> tuple[np.ndarray, int]:
    """Orb neighborhood identification.

    This function identifies which coordinates lie within an orb
    neighborhood defined by radius r about a given coordinate.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        coords (np.ndarray): 2D np.ndarray of (en, dim) float entries of coordinates that may or may not reside in the orb neighborhood.
        coord (np.ndarray): 1D np.ndarray of dim float entries of a given coordinate which the orb neighborhood is defined about.
        r (float): Radius defining the orb neighborhood about the given coordinate.
        inclusive (bool): Boolean indicating if the orb neighborhood is inclusive or exclusive of its boundary.
    
    Returns:
        tuple[np.ndarray, int]: 1D np.ndarray of en int entries of orb
        neighbor indices, and the number of orb neighbors.

    """
    # Gather the corresponding box neighborhood about the given
    # coordinate
    box_nghbr_indcs, box_nghbr_num = box_neighborhood_id(
        dim, coords, coord, r, inclusive)
    
    # Corresponding box neighborhood is empty, which implies that the
    # orb neighborhood is also empty
    if box_nghbr_num == 0: return np.asarray([], dtype=int), 0
    # Corresponding box neighborhood has at least one neighbor in it
    elif box_nghbr_num > 0:
        # Extract box neighbor coordinates
        box_nghbr_coords = coords[box_nghbr_indcs]

        # Calculate the distance between the given coordinate and its
        # box neighbors
        dist = np.asarray(
            [
                np.linalg.norm(coord-box_nghbr_coords[box_nghbr_indx])
                for box_nghbr_indx in range(box_nghbr_num)
            ])
        
        # Determine the naive indices of the orb neighbors, and
        # calculate the number of orb neighbors
        if inclusive: orb_nghbr_indcs = np.where(dist <= r)[0]
        else: orb_nghbr_indcs = np.where(dist < r)[0]
        orb_nghbr_num = np.shape(orb_nghbr_indcs)[0]

        # Orb neighborhood is empty
        if orb_nghbr_num == 0: return np.asarray([], dtype=int), 0
        # Orb neighborhood has at least one neighbor in it
        else: 
            # Determine the indices of the orb neighbors
            orb_nghbr_indcs = box_nghbr_indcs[orb_nghbr_indcs]
            return orb_nghbr_indcs, orb_nghbr_num