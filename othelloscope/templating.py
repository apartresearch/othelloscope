def generate_from_template(template: str, *args) -> str:
    """Generate a file from a template.

    Parameters
    ----------
    template : str
        Path to the template file.
    **kwargs
        Keyword arguments to be used in the template.

    Returns
    -------
    str
        The generated file.
    """
    with open(template, "r") as f:
        template = f.read()
    return template.format(*args)
