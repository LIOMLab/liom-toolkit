import os


def fix_even(number):
    """
    Fix even numbers by adding 1

    :param number: The number to fix
    :return: The fixed number
    """
    if number % 2 == 0:
        number += 1
    return number


def clean_dir(directory):
    """
    Remove default files in a directory

    :param directory: The directory to clean
    """
    if os.path.exists(directory + '.DS_Store'):
        os.remove(directory + '.DS_Store')
