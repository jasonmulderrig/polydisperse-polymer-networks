import os
import pathlib

def data_filepath_str() -> str:
    # For MacOS
    data_filepath = f"/Users/jasonmulderrig/research/projects/polydisperse-polymer-networks/src/data/"
    # For Windows OS
    # data_filepath = f"C:\\Users\\mulderjp\\projects\\polydisperse-polymer-networks\\src\\data\\"
    # For Linux
    # data_filepath = f"/p/home/jpm2225/projects/polydisperse-polymer-networks/src/data/"
    if os.path.isdir(data_filepath) == False:
        pathlib.Path(data_filepath).mkdir(parents=True, exist_ok=True)
    return data_filepath

def spherical_quadrature_filepath_str() -> str:
    # For MacOS
    sph_quad_filepath = f"/Users/jasonmulderrig/research/projects/polydisperse-polymer-networks/src/spherical_quadrature/"
    # For Windows OS
    # sph_quad_filepath = f"C:\\Users\\mulderjp\\projects\\polydisperse-polymer-networks\\src\\spherical_quadrature\\"
    # For Linux
    # sph_quad_filepath = f"/p/home/jpm2225/projects/polydisperse-polymer-networks/src/spherical_quadrature/"
    if os.path.isdir(sph_quad_filepath) == False:
        pathlib.Path(sph_quad_filepath).mkdir(parents=True, exist_ok=True)
    return sph_quad_filepath

def root_filepath_str(workdir: str) -> str:
    """Root filepath generator.

    This function returns the root baseline filepath. The filepath must
    match the directory structure of the local computer. For Windows
    machines, the backslash must be represented as a double backslash.
    For Linux/Mac machines, the forwardslash can be directly represented
    as a forwardslash.

    Args:
        workdir (str): Work directory name.
    
    Returns:
        str: The root baseline filepath.
    
    """
    # For MacOS
    root_filepath = f"/Users/jasonmulderrig/research/projects/polydisperse-polymer-networks/{workdir}/"
    # For Windows OS
    # root_filepath = f"C:\\Users\\mulderjp\\projects\\polydisperse-polymer-networks\\{workdir}\\"
    # For Linux
    # root_filepath = f"/p/home/jpm2225/projects/polydisperse-polymer-networks/{workdir}/"
    if os.path.isdir(root_filepath) == False:
        pathlib.Path(root_filepath).mkdir(parents=True, exist_ok=True)
    return root_filepath

def filepath_str(workdir: str) -> str:
    """Filepath generator.

    This function returns the baseline filepath.

    Args:
        workdir (str): Work directory name.
    
    Returns:
        str: The baseline filepath.
    
    """
    # For MacOS
    filepath = f"/Users/jasonmulderrig/research/projects/polydisperse-polymer-networks/{workdir}/raw/"
    # For Windows OS
    # filepath = f"C:\\Users\\mulderjp\\projects\\polydisperse-polymer-networks\\{workdir}\\raw\\"
    # For Linux
    # filepath = f"/p/home/jpm2225/projects/polydisperse-polymer-networks/{workdir}/raw/"
    if os.path.isdir(filepath) == False:
        pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
    return filepath

def _filename_str(date: str, batch: str, sample: int) -> str:
    """Baseline filename string generator.

    This function returns the baseline filename string.

    Args:
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
    
    Returns:
        str: The baseline filename string.
    
    """
    return f"{date}{batch}{sample:d}"

def filename_str(
        workdir: str,
        date: str,
        batch: str,
        sample: int) -> str:
    """Baseline filename generator.

    This function returns the baseline filename. The baseline filename
    is explicitly prefixed with the filepath to the directory that the
    files ought to be saved to (and loaded from for future use). This
    filepath is set by the user, and must match the directory structure
    of the local computer. The baseline filename is then appended to the
    filepath. It is incumbent on the user to save a data file that
    records the network parameter values that correspond to each network
    sample in the batch (i.e., a "lookup table").

    Args:
        workdir (str): Work directory name.
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
    
    Returns:
        str: The baseline filename.
    
    """
    return filepath_str(workdir) + _filename_str(date, batch, sample)

def L_filename_str(
        workdir: str,
        date: str,
        batch: str,
        sample: int) -> str:
    """Filename for simulation box side lengths.

    This function returns the filename for the simulation box side
    lengths.

    Args:
        workdir (str): Work directory name.
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
    
    Returns:
        str: The simulation box side lengths filename.
    
    """
    return filename_str(workdir, date, batch, sample) + "-L" + ".dat"

def _config_filename_str(
        date: str,
        batch: str,
        sample: int,
        config: int) -> str:
    """Configuration filename string.

    This function returns the configuration filename string.

    Args:
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
    
    Returns:
        str: The configuration filename string.
    
    """
    return _filename_str(date, batch, sample) + f"C{config:d}"

def config_filename_str(
        workdir: str,
        date: str,
        batch: str,
        sample: int,
        config: int) -> str:
    """Configuration filename prefix.

    This function returns the configuration filename prefix.

    Args:
        workdir (str): Work directory name.
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
    
    Returns:
        str: The configuration filename prefix.
    
    """
    return (
        filepath_str(workdir)
        + _config_filename_str(date, batch, sample, config)
    )