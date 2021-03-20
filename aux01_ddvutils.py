def chunk4d(ds, maxsize_4d=1000**2, sample_var="u", round_func=round, 
            time_var="time", **kwargs):
    """ Chunk `ds` in time while keeping each chunk's size roughly 
    around `maxsize_4d`. The default `maxsize_4d=1000**2` comes from
    xarray's rule of thumb for chunking:
    http://xarray.pydata.org/en/stable/dask.html#chunking-and-performance
    """
    chunk_number = ds[sample_var].size / maxsize_4d
    chunk_size = int(round_func(len(ds[sample_var][time_var]) / chunk_number))
    return ds.chunk({time_var : chunk_size})
