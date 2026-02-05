import numpy as np
import numpy.typing as npt

def n_init_func(
        n_init: str,
        n: tuple[float] | tuple[int],
        n_type: float | int) -> tuple[npt.NDArray[np.floating | np.integer], int]:
    """Salient chain segment numbers.

    This function initializes the salient chain segment numbers.

    Args:
        n_init (str): Short-hand description for the salient chain segment number initialization protocol; either "explicit" or "linspace".
        n (tuple[float] | tuple[int]): Salient chain segment numbers, or information needed to properly initialize the salient chain segment numbers.
        n_type (float | int): Array type of the salient chain segment numbers; either float or int.
    
    Returns:
        tuple[npt.NDArray[np.floating | np.integer], int]: Salient chain
        segment numbers and the number of salient chain segment numbers.
    
    """
    # Define the salient chain segment number array
    if n_type != float and n_type != int:
        error_str = (
            "The chain segment numbers need to be provided as a tuple "
            + "of ints or a tuple of floats!"
        )
        raise ValueError(error_str)
    else:
        if n_init == "explicit": n = np.asarray(n, dtype=n_type)
        elif n_init == "linspace": n = np.linspace(n[0], n[1], n[2], dtype=n_type)

    # Number of salient chain segment numbers
    N = np.shape(n)[0]

    return n, N