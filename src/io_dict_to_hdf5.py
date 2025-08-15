# copied from https://codereview.stackexchange.com/a/121308 (and slightly modified for updated h5py, Elliott Abe)
import numpy as np
import h5py
import jax.numpy as jnp
#import os

def save(filename, dic, compression='gzip', compression_opts=9):
    """
    saves a python dictionary or list, with items that are themselves either
    dictionaries or lists or (in the case of tree-leaves) numpy arrays
    or basic scalar types (int/float/str/bytes) in a recursive
    manner to an hdf5 file, with an intact hierarchy.
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 file to save
    dic : dict/list
        Data structure to save
    compression : str, optional
        Compression algorithm ('gzip', 'lzf', 'szip', None). Default is 'gzip'
    compression_opts : int, optional
        Compression level (0-9 for gzip). Default is 9 (maximum compression)
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic, compression, compression_opts)

def recursively_save_dict_contents_to_group(h5file, path, dic, compression='gzip', compression_opts=9):
    """
    Recursively save dictionary contents to HDF5 group with compression support.
    
    Parameters:
    -----------
    h5file : h5py.File
        The HDF5 file object
    path : str
        Current path in the HDF5 hierarchy
    dic : dict/list/object
        Data structure to save
    compression : str, optional
        Compression algorithm ('gzip', 'lzf', 'szip', None). Default is 'gzip'
    compression_opts : int, optional
        Compression level (0-9 for gzip). Default is 9 (maximum compression)
    """
    if isinstance(dic,dict):
        iterator = dic.items()
    elif isinstance(dic,list):
        iterator = enumerate(dic)
    elif isinstance(dic,object):
        iterator = dic.__dict__.items()
    else:
        ValueError('Cannot save %s type' % type(dic))

    for key, item in iterator: #dic.items():
        if isinstance(dic,(list, tuple)):
            key = str(key)
        if isinstance(item, (jnp.ndarray, np.ndarray, np.int64, np.float64, int, float, str, bytes)):
            # Use create_dataset with compression for arrays and numeric data
            if isinstance(item, (jnp.ndarray, np.ndarray)) and item.size > 1:
                # Apply compression for arrays with more than one element
                h5file.create_dataset(path + key, data=item, compression=compression, compression_opts=compression_opts)
            else:
                # For scalars and single-element arrays, compression may not be beneficial
                h5file[path + key] = item
        elif isinstance(item, (dict,list,object)): # or isinstance(item,list) or isinstance(item,tuple):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item, compression, compression_opts)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load(filename, ASLIST=False, enable_jax=False):
    """
    Default: load a hdf5 file (saved with io_dict_to_hdf5.save function above) as a hierarchical
    python dictionary (as described in the doc_string of io_dict_to_hdf5.save).
    if ASLIST is True: then it loads as a list (on in the first layer) and gives error if key's are not convertible
    to integers. Unlike io_dict_to_hdf5.save, a mixed dictionary/list hierarchical version is not implemented currently
    for .load
    """
    with h5py.File(filename, 'r') as h5file:
        out = recursively_load_dict_contents_from_group(h5file, '/', enable_jax=enable_jax)
        if ASLIST:
            outl = [None for l in range(len(out.keys()))]
            for key, item in out.items():
                outl[int(key)] = item
            out = outl
        return out


def recursively_load_dict_contents_from_group(h5file, path, enable_jax=False):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            if enable_jax and isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = jnp.asarray(item[()])
            else:
                ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/', enable_jax=enable_jax)
    return ans
