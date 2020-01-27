"""
In this file
All the custom exceptions, errors
"""



class EmptyFolderError(Exception):
    """
    Raised when the folder does not contain any data of the right format
    In dataloader.py, they should be .npy for the images and .png for the masks
    """
    pass

class WrongArgumentError(Exception):
    """
    Raised when a wrong choice is made in the parser
    """
    pass



