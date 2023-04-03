def _blosc_opts(complevel=1, complib='blosc:zstd', shuffle='byte'):
    shuffle = 2 if shuffle == 'bit' else 1 if shuffle == 'byte' else 0
    compressors = ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']
    complib = ['blosc:' + c for c in compressors].index(complib)
    args = {
        'compression': 32001,
        'compression_opts': (0, 0, 0, 0, complevel, shuffle, complib),
    }
    if shuffle > 0:
        # Do not use h5py shuffle if blosc shuffle is enabled.
        args['shuffle'] = False
    return args
