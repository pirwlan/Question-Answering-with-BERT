import os


def create_folder(path):
    """
    savely creates folder if not already present

    Args:
        path: str - path of folder to create
    """

    try:
        os.mkdir(path)
    except FileExistsError:
        pass