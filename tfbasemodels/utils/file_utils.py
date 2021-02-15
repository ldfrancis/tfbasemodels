import requests
import re
import os
import hashlib
from pathlib import Path
from .progressbar_utils import obtain_progressbar
from math import ceil


def obtain_base_dir():
    """Obtains the base directory to use
    Returns:
        A Path object
    """
    path = Path(os.path.expanduser("~")+"/tfbasemodels")
    path.mkdir(exist_ok=True)
    return path


def download_file(url, file_dir="", name_to_use=None):
    """Downloads file from url into file_dir

    Args:
        url: str. the url of the file to be downloaded
        file_dir: str. directory to save the downloaded file.

    Returns:
        None
    """
    # create file folder
    path = obtain_base_dir()/(file_dir)
    path.mkdir(exist_ok=True)

    # make requestt
    r = requests.get(url, allow_redirects=True, stream=True)

    # obtain filename and set path
    filename = _obtain_filename_from_response(response) \
        if name_to_use is None else name_to_use
    filepath = path/(""+filename)

    total_size_in_bytes= int(r.headers.get('content-length', 0))
    divisor_MB = 1024*1024
    total_size_in_MB = ceil(total_size_in_bytes/divisor_MB)
    print("Downloading", str(total_size_in_MB)+"MB")
    print("Url:", url)
    print("filename:", filename)
    print("filepath:", filepath)
    running_sum_of_size = 0
    block_size = 1024
    progress_bar = obtain_progressbar(max_value=total_size_in_MB)
    with open(filepath, 'wb') as file:
        for data in r.iter_content(block_size):
            running_sum_of_size += len(data)/divisor_MB
            value = round(running_sum_of_size, 2)
            value = min(value, total_size_in_MB)
            progress_bar.update(value)
            file.write(data)
    progress_bar.update(total_size_in_MB)
    progress_bar.finish()

    
def _obtain_filename_from_response(response):
    """Obtains the filename from a requests response object
    Args:
        response: a response object

    Returns:
        filename: a string
    """
    content_disposition = response.headers.get("content-disposition")
    filename = re.findall('filename=(.+)', content_disposition) \
        if content_disposition else None
    filename = (None if len(filename)==0 else filename[0]) \
        if filename else None

    return filename


def validate_file(path, filehash, hash_algo="md5"):
    """Validates a file given by the path using the supplied
    file hash
    """
    obtained_file_hash = obtain_file_hash(path, hash_algo)
    return obtained_file_hash==filehash
    

def obtain_file_hash(path, hash_algo="md5"):
    """Obtains the hash of a file using the specified hash algorithm
    """
    hash_algo = hashlib.sha256() if hash_algo=="sha256" else hashlib.md5()

    block_size = 65535

    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size),b''):
            hash_algo.update(chunk)

    return hash_algo.hexdigest()

