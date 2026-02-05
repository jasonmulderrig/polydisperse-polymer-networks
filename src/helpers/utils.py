def arccos_arg_cnstrnt_func(arccos_arg: float) -> float:
    """Constraint for the argument of the arccos function.

    This function constrains the argument of the arccos function.

    Args:
        x (float): Argument of the arccos function.
    
    Returns:
        float: Constrained argument of the arccos function.
    
    """
    if arccos_arg >= 1. - 1.e-14: arccos_arg = 1. - 1.e-14
    elif arccos_arg < -1. + 1.e-14: arccos_arg = -1. + 1.e-14
    return arccos_arg