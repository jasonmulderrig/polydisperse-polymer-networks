import numpy as np
import numpy.typing as npt
from src.helpers.network import a_or_v_func

def A_or_V_arg_L_func(L: npt.NDArray[np.float64]) -> float:
    """Simulation box area or volume.

    This function calculates the simulation box area or volume given the
    simulation box side lengths.

    Args:
        L (npt.NDArray[np.float64]): 1D array with dim float entries of the simulation box side lengths.

    Returns:
        float: Simulation box area or volume.
    
    """
    return np.prod(L)

def A_or_V_arg_rho_func(en: float, rho: float) -> float:
    """Simulation box area or volume.

    This function calculates the simulation box area or volume given the
    number of particles and the particle number density.

    Args:
        en (float): Number of particles.
        rho (float): Particle number density.

    Returns:
        float: Simulation box area or volume.
    
    """
    return en / rho

def A_or_V_arg_eta_func(dim: int, b: float, en: float, eta: float) -> float:
    """Simulation box area or volume.
    
    This function calculates the simulation box area or volume given the
    number of particles and the particle packing density.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Particle diameter.
        en (float): Number of particles.
        eta (float): Particle packing density.

    Returns:
        float: Simulation box area or volume.
    
    """
    return a_or_v_func(dim, b) * en / eta

def L_arg_A_or_V_func(dim: int, A_or_V: float) -> npt.NDArray[np.float64]:
    """Simulation box side lengths.
    
    This function calculates the simulation box side lengths given the
    simulation box area or volume. This function assumes that the
    simulation box is either a square or a cube (for two-dimensional or
    three-dimensional networks, respectively).

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        A_or_V (float): Simulation box area or volume.

    Returns:
        npt.NDArray[np.float64]: 1D array with dim float entries of the
        simulation box side lengths.
    
    """
    return np.repeat(np.power(A_or_V, np.reciprocal(1.0*dim)), dim)

def L_arg_rho_func(dim: int, en: float, rho: float) -> npt.NDArray[np.float64]:
    """Simulation box side lengths.
    
    This function calculates the simulation box side lengths given the
    number of particles and the particle number density. This function
    assumes that the simulation box is either a square or a cube (for
    two-dimensional or three-dimensional networks, respectively).

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        en (float): Number of particles.
        rho (float): Particle number density.

    Returns:
        npt.NDArray[np.float64]: 1D array with dim float entries of the
        simulation box side lengths.
    
    """
    return L_arg_A_or_V_func(dim, A_or_V_arg_rho_func(en, rho))

def L_arg_eta_func(
        dim: int,
        b: float,
        en: float,
        eta: float) -> npt.NDArray[np.float64]:
    """Simulation box side lengths.
    
    This function calculates the simulation box side lengths given the
    number of particles and the particle packing density. This function
    assumes that the simulation box is either a square or a cube (for
    two-dimensional or three-dimensional networks, respectively).

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Particle diameter.
        en (float): Number of particles.
        eta (float): Particle packing density.

    Returns:
        npt.NDArray[np.float64]: 1D array with dim float entries of the
        simulation box side lengths.
    
    """
    return L_arg_A_or_V_func(dim, A_or_V_arg_eta_func(dim, b, en, eta))

def mic_func(
        coords: npt.NDArray[np.float64],
        L: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Minimum image criterion.
    
    This function modifies a provided coordinates array to satisfy the
    minimum image criterion within a simulation box defined by side
    lengths L.

    Args:
        coords (npt.NDArray[np.float64]): 2D array with (en, dim) float entries of coordinates.
        L (npt.NDArray[np.float64]): 1D array with dim float entries of the simulation box side lengths.

    Returns:
        npt.NDArray[np.float64]: Coordinates that satisfy the minimum
        image criterion.
    
    """
    while not np.all(np.logical_and(np.greater_equal(coords, np.zeros(3)), np.less(coords, L))):
        coords = np.where(coords<0, coords+L, coords)
        coords = np.where(coords>=L, coords-L, coords)
    return coords

def L_min_func(L: npt.NDArray[np.float64]) -> float:
    """Minimum simulation box length.
    
    This function calculates the minimum simulation box length.

    Args:
        L (npt.NDArray[np.float64]): 1D array with dim float entries of the simulation box side lengths.

    Returns:
        float: Minimum simulation box length.
    
    """
    return np.min(L)

def L_max_func(L: npt.NDArray[np.float64]) -> float:
    """Maximum simulation box length.
    
    This function calculates the maximum simulation box length.

    Args:
        L (npt.NDArray[np.float64]): 1D array with dim float entries of the simulation box side lengths.

    Returns:
        float: Maximum simulation box length.
    
    """
    return np.max(L)

def L_diag_max_func(L: npt.NDArray[np.float64]) -> float:
    """Maximum simulation box diagonal length.
    
    This function calculates the maximum simulation box diagonal length.

    Args:
        L (npt.NDArray[np.float64]): 1D array with dim float entries of the simulation box side lengths.

    Returns:
        float: Maximum simulation box diagonal length.
    
    """
    return np.sqrt(np.sum(L**2))

def tessellation_protocol(dim: int) -> tuple[npt.NDArray[np.int64], int]:
    """Tessellation protocol.

    This function determines the tessellation protocol and the number of
    tessellations involved in that protocol. Each of these are sensitive
    to the physical dimensionality of the network.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
    
    Returns:
        tuple[npt.NDArray[np.int64], int]: 2D array with (3**dim, dim)
        int entries representing the tessellation protocol and the
        number of tessellations involved in that protocol, respectively.
    
    """
    base_tsslltn = np.asarray([-1, 0, 1], dtype=int)
    tsslltn = (
        np.transpose(np.asarray(np.meshgrid(*(base_tsslltn,)*dim))).reshape(-1, dim)
    )
    return tsslltn, np.shape(tsslltn)[0]

def tessellation(
        coords: npt.NDArray[np.float64],
        tsslltn: npt.NDArray[np.int64],
        L: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Tessellation.
    
    This function fully tessellates (or translates) an arbitrary
    coordinate in each cardinal direction in the spatial plane via a
    scaling L.

    Args:
        coords (npt.NDArray[np.float64]): 2D array with (en, dim) float entries of the coordinates to be tessellated.
        tsslltn (npt.NDArray[np.int64]): 2Darray with (3**dim, dim) int entries representing the tessellation protocol.
        L (npt.NDArray[np.float64]): 1D array with dim float entries of the tessellation scaling.
    
    Returns:
        npt.NDArray[np.float64]: 2D array with (en*3**dim, dim) float
        entries of the tessellated coordinates.
    
    """
    return coords + tsslltn * L

def core_clnkr_tessellation(
        dim: int,
        core_clnkrs: npt.NDArray[np.int64],
        core_coords: npt.NDArray[np.float64],
        L: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
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
        core_clnkrs (npt.NDArray[np.int64]): 1D array with en int entries of the core cross-linkers.
        core_coords (npt.NDArray[np.float64]): 2D array with (en, dim) float entries of the core cross-linker coordinates to be tessellated.
        L (npt.NDArray[np.float64]): 1D array with dim float entries of the tessellation scaling.
    
    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: 2D
        array with (en*3**dim, dim) float entries of the tessellated
        cross-linker coordinates and 1D array with en*3**dim int entries
        that returns the core cross-linker that corresponds to each core
        and periodic boundary cross-linker, i.e.,
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
            if np.array_equal(tsslltn[tsslltn_actn], np.zeros(dim, dtype=int)) == True:
                continue
            else:
                tsslltd_core_coords = np.vstack(
                    (tsslltd_core_coords, tessellation(core_coords, tsslltn[tsslltn_actn], L)))

        # Construct the pb_to_core_clnkrs np.ndarray such that
        # pb_to_core_clnkrs[core_pb_clnkr] = core_clnkr
        pb_to_core_clnkrs = np.tile(core_clnkrs, tsslltn_num)

        return tsslltd_core_coords, pb_to_core_clnkrs

def core_to_pb_clnkrs_func(
        core_clnkrs: npt.NDArray[np.int64],
        pb_to_core_clnkrs: npt.NDArray[np.int64]) -> list[npt.NDArray[np.int64]]:
    """List of np.ndarrays corresponding to the periodic boundary
    cross-linkers associated with a particular core cross-linker.

    This function creates a list of np.ndarrays corresponding to the
    periodic boundary cross-linkers associated with a particular core
    cross-linker such that core_to_pb_clnkrs[core_clnkr] = pb_clnkrs.

    Args:
        core_clnkrs (npt.NDArray[np.int64]): 1D array with en int entries of the core cross-linkers.
        pb_to_core_clnkrs (npt.NDArray[np.int64]): 1D array with en*3**dim int entries that returns the core cross-linker that corresponds to each core and periodic boundary cross-linker, i.e., pb_to_core_clnkrs[core_pb_clnkr] = core_clnkr.
    
    Returns:
        list[npt.NDArray[np.int64]]: List of 1D arrays with int
        entries corresponding to the periodic boundary cross-linkers
        associated with a particular core cross-linker.

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
        coords: npt.NDArray[np.float64],
        coord: npt.NDArray[np.float64],
        l: float,
        inclusive: bool) -> tuple[npt.NDArray[np.int64], int]:
    """Box neighborhood identification.
    
    This function return the coordinates or the indices of the
    coordinates that lie within a box neighborhood that is \\pm
    half-side length l about a given coordinate.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        coords (npt.NDArray[np.float64]): 2D array of (en, dim) float entries of coordinates that may or may not reside in the box neighborhood.
        coord (npt.NDArray[np.float64]): 1D array of dim float entries of a given coordinate which the box neighborhood is defined about.
        l (float): Half-side length defining the box neighborhood about the given coordinate.
        inclusive (bool): Boolean indicating if the box neighborhood is inclusive or exclusive of its boundary.
    
    Returns:
        tuple[npt.NDArray[np.int64], int]: 1D array of en int entries
        of box neighbor indices and the number of box neighbors.

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
        coords: npt.NDArray[np.float64],
        coord: npt.NDArray[np.float64],
        r: float,
        inclusive: bool) -> tuple[npt.NDArray[np.int64], int]:
    """Orb neighborhood identification.

    This function identifies which coordinates lie within an orb
    neighborhood defined by radius r about a given coordinate.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        coords (npt.NDArray[np.float64]): 2D array of (en, dim) float entries of coordinates that may or may not reside in the orb neighborhood.
        coord (npt.NDArray[np.float64]): 1D array of dim float entries of a given coordinate which the orb neighborhood is defined about.
        r (float): Radius defining the orb neighborhood about the given coordinate.
        inclusive (bool): Boolean indicating if the orb neighborhood is inclusive or exclusive of its boundary.
    
    Returns:
        tuple[npt.NDArray[np.int64], int]: 1D array of en int entries
        of orb neighbor indices, and the number of orb neighbors.

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