import os
import datetime

def get_folders_files(path, wp=True):
    """
    Returns lists of files and folders from directory
    path - folder address
    wp - BOOL, list with path
    """
    files = []
    folders = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        folders.extend(dirnames)
        break
    if wp:
        files = [path + os.sep + i for i in files]
        folders = [path + os.sep + i for i in folders]

    return (folders, files)


def get_files(path, prefix, extension, wp=True):
    """
    Returns lists of files within folder with specific
    prefix, extension and both.
    path - folder address
    wp - BOOL, list with path
    prefix - string, beginning of the filename
    extension - string, extension w/o dot
    """
    ext = []
    prfx = []
    intersect = []
    files = get_folders_files(path, wp=False)[1]
    for i in files:
        if i.startswith(prefix):
            prfx.append(i)
        if i.endswith(extension):
            ext.append(i)
        if i.endswith(extension) and i.startswith(prefix):
            intersect.append(i)
    if wp:
        ext = [path + os.sep + i for i in ext]
        prfx = [path + os.sep + i for i in prfx]
        intersect = [path + os.sep + i for i in intersect]

    return (ext, prfx, intersect)


def get_folders(path, prefix, extension, wp=True):
    """
    Returns lists of folders within folder with specific
    prefix, extension and both.
    path - folder address
    wp - BOOL, list with path
    prefix - string, beginning of the filename
    extension - string, extension w/o dot
    """
    ext = []
    prfx = []
    intersect = []
    folders = get_folders_files(path, wp=False)[0]
    for i in folders:
        if i.startswith(prefix):
            prfx.append(i)
        if i.endswith(extension):
            ext.append(i)
        if i.endswith(extension) and i.startswith(prefix):
            intersect.append(i)
    if wp:
        ext = [path + os.sep + i for i in ext]
        prfx = [path + os.sep + i for i in prfx]
        intersect = [path + os.sep + i for i in intersect]

    return (ext, prfx, intersect)

def make_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)