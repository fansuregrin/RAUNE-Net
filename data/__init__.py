import os


__image_extions = ('.JPG', '.JPEG', '.PNG', '.TIFF', '.BMP')


def is_image_file(filepath: str) -> bool:
    """Test whether a path is image file.
    
    Args:
        filepath: a path to be tested.

    Returns:
        A bool value indicates whether a path is an image file.
    """
    if not os.path.isfile(filepath):
        return False
    _, ext = os.path.splitext(filepath)
    if ext: ext = ext.upper()
    else: return False
    if ext in __image_extions:
        return True
    else:
        return False