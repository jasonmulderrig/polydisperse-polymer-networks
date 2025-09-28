import numpy as np

def n_init_func(n_init: str, n: tuple[int]) -> tuple[np.ndarray, int]:
    """Salient chain segment numbers.

    This function initializes the salient chain segment numbers.

    Args:
        n_init (str): Short-hand description for the salient chain segment number initialization protocol; either "explicit" or "linspace".
        n (tuple[int]): Salient chain segment numbers, or information needed to properly initialize the salient chain segment numbers.
    
    Returns:
        tuple[np.ndarray, int]: Salient chain segment numbers (sorted
        from least to greatest), and the number of salient chain segment
        numbers.
    
    """
    # Define the salient chain segment number array
    if n_init == "explicit": n = np.asarray(n, dtype=int)
    elif n_init == "linspace": n = np.linspace(n[0], n[1], n[2], dtype=int)

    # Number of salient chain segment numbers
    N = np.shape(n)[0]

    return n, N