import numpy as np
from src.helpers.network_topology_initialization_utils import (
    tessellation_protocol,
    tessellation,
    orb_neighborhood_id
)

def initial_random_clnkr_placement(
        rng: np.random.Generator,
        dim: int,
        L: np.ndarray,
        en: int) -> np.ndarray:
    """Initial random cross-linker placement procedure.

    This function randomly places/seeds cross-linkers within an empty
    simulation box.

    Args:
        rng (np.random.Generator): np.random.Generator object.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        L (np.ndarray): 1D np.ndarray with dim float entries of the simulation box side lengths.
        en (int): Number of cross-linkers.
    
    Returns:
        np.ndarray: 2D np.ndarray with (en, dim) float entries of the
        coordinates of the randomly placed/seeded cross-linkers.
    
    """
    # Random cross-linker placement
    return L*rng.random((en, dim))

def periodic_random_hard_disk_clnkr_placement(
        rng: np.random.Generator,
        dim: int,
        L: np.ndarray,
        b: float,
        tsslltn: np.ndarray,
        coords: np.ndarray,
        tsslltd_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Periodic random hard disk cross-linker placement procedure.

    This function randomly places/seeds cross-linkers within a
    simulation box where each cross-linker is treated as the center of a
    hard disk, and the simulation box is periodic.

    Args:
        rng (np.random.Generator): np.random.Generator object.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        L (np.ndarray): 1D np.ndarray with dim float entries of the simulation box side lengths.
        b (float): Hard disk diameter.
        tsslltn (np.ndarray): 2D np.ndarray with (3**dim, dim) int entries representing the tessellation protocol.
        coords (np.ndarray): 2D np.ndarray with (en, dim) float entries of the cross-linker coordinates.
        tsslltd_coords (np.ndarray): 2D np.ndarray with (en*3**dim, dim) float entries of the tessellated cross-linker coordinates.
    
    Return:
        tuple[np.ndarray, np.ndarray]: 2D np.ndarray with (en, dim)
        float entries of the cross-linker coordinates, and the 2D
        np.ndarray with (en*3**dim, dim) float entries of the
        tessellated cross-linker coordinates.
    
    """
    # Begin periodic random hard disk cross-linker placement procedure
    num_try = 0
    
    while num_try < 1000:
        # Generate randomly placed cross-linker candidate
        seed_cnddt = L * rng.random(size=dim)

        # Downselect the previously-accepted tessellated cross-linkers
        # to those that reside in a local orb neighborhood with radius
        # b about the cross-linker candidate
        _, orb_nghbr_num = orb_neighborhood_id(
            dim, tsslltd_coords, seed_cnddt, b, inclusive=False)
        
        # Try again if the local orb neighborhood has at least one
        # neighbor in it
        if orb_nghbr_num > 0:
            num_try += 1
            continue
        
        # Accept and tessellate the cross-linker candidate if no local
        # orb neighborhood of tessellated cross-linkers exists about the
        # cross-linker candidate
        coords = np.vstack((coords, seed_cnddt))
        tsslltd_coords = np.vstack(
            (tsslltd_coords, tessellation(seed_cnddt, tsslltn, L)))
        break

    return coords, tsslltd_coords

def initial_periodic_random_hard_disk_clnkr_placement(
        rng: np.random.Generator,
        dim: int,
        L: np.ndarray,
        b: float,
        en: int) -> np.ndarray:
    """Initial periodic random hard disk cross-linker placement
    procedure.

    This function randomly places/seeds cross-linkers within an empty
    simulation box where each cross-linker is treated as the center of a
    hard disk, and the simulation box is periodic.

    Args:
        rng (np.random.Generator): np.random.Generator object.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        L (np.ndarray): 1D np.ndarray with dim float entries of the simulation box side lengths.
        b (float): Hard disk diameter.
        en (int): Intended number of cross-linkers.
    
    Returns:
        np.ndarray: 2D np.ndarray with (en, dim) float entries of the
        coordinates of the periodic random hard disk placed/seeded
        cross-linkers.
    
    """
    # Tessellation protocol
    tsslltn, _ = tessellation_protocol(dim)

    # Periodic random hard disk cross-linker placement procedure
    for seed_attmpt in range(en):
        # Accept and tessellate the first cross-linker
        if seed_attmpt == 0:
            # Accept the first cross-linker
            seed = L * rng.random(size=dim)
            coords = seed.copy()

            # Tessellate the first cross-linker
            seed_tsslltn = tessellation(seed, tsslltn, L)
            tsslltd_coords = seed_tsslltn.copy()
        else:
            # Begin periodic random hard disk cross-linker placement
            # procedure
            coords, tsslltd_coords = periodic_random_hard_disk_clnkr_placement(
                rng, dim, L, b, tsslltn, coords, tsslltd_coords)

    # Return the seeded cross-linkers if the number of seeded
    # cross-linkers equals the the intended number of cross-linkers. If
    # this is not true, then the number of seeded cross-linkers is less
    # than the intended number of cross-linkers, and an error is raised.
    prhd_en = np.shape(coords)[0]
    if prhd_en == en: return coords
    else:
        error_str = (
            "The actual number of seeded-cross-linkers, " + str(prhd_en)
            + ", is less than the intended number of cross-linkers, "
            + str(en) + ". "
        )
        raise ValueError(error_str)

def initial_clnkr_seeding(
        rng: np.random.Generator,
        scheme: str,
        dim: int,
        L: np.ndarray,
        b: float,
        en: int) -> np.ndarray:
    """Initial cross-linker placement procedure.

    This function calls upon a corresponding helper function to
    calculate the initial cross-linker positions.

    Args:
        rng (np.random.Generator): np.random.Generator object.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the cross-linkers; either "random" or "prhd" (corresponding to the random cross-linker placement procedure ("random") or periodic random hard disk cross-linker placement procedure ("prhd")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        L (np.ndarray): 1D np.ndarray with dim float entries of the simulation box side lengths.
        b (float): Cross-linker diameter.
        en (int): Number of cross-linkers.
    
    Returns:
        np.ndarray: 2D np.ndarray with (en, dim) float entries of the
        cross-linker coordinates.
    
    """
    # Call appropriate initial cross-linker placement helper function
    if scheme == "random":
        return initial_random_clnkr_placement(rng, dim, L, en)
    elif scheme == "prhd":
        return initial_periodic_random_hard_disk_clnkr_placement(
            rng, dim, L, b, en)