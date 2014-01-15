from .calibrate_map_scans import load_data_file
from .makecube import selectsource,velo_iterator
import numpy as np

def make_off(fitsfile, scanrange=[], sourcename=None, feednum=1, sampler=0,
             dataarr=None, obsmode=None, exclude_velo=(), interp_polyorder=5,
             extension=1,
             percentile=50, interp_vrange=(), linefreq=None, return_uninterp=False):
    """
    Create an 'off' spectrum from a large collection of data by taking
    the median across time (or fitting across time?) and interpolating across certain
    spectral channels

    Parameters
    ----------
    fitsfile : str or pyfits.HDUList
    scanrange : 2-tuple
        *DATA SELECTION PARAMETER* Range of scans to include when creating off positions
    sourcename : str or None
        *DATA SELECTION PARAMETER* Name of source to include
    feednum : int
        *DATA SELECTION PARAMETER* Feed number to use (1 for single-feed systems)
    sampler : str
        *DATA SELECTION PARAMETER* Sampler to create the off for (e.g., 'A9')
    obsmode : str
        *DATA SELECTION PARAMETER* Observation Mode to include (e.g., DecLatMap)
    dataarr : None or np.ndarray
        OPTIONAL input of data array.  If it has already been read, this param saves time
    exclude_velo : 2n-tuple
        velocities to exclude / interpolate over when making the 'off'
    interp_polyorder : int
        Order of the polynomial to fit when interpolating across
    interp_vrange : 2-tuple
        Range of velocities to interpolate over (don't use whole spectrum -
        leads to bad fits)
    linefreq : float
        Line frequency reference for velocity
    percentile : float
        The percentile of the data to use for the reference.  Normally, you
        would use 50 to get the median of the data, but if there is emission at
        all positions, you might choose, e.g., 25, or absorption, 75.

    Returns
    -------
    off_template (interpolated) : np.ndarray
        a NORMALIZED off spectrum
    off_template_in : np.ndarray [OPTIONAL]
        if return_uninterp is set, the "average" off position (not
        interpolated) will be returned
    
        
    """

    data, dataarr, namelist, filepyfits = load_data_file(fitsfile, extension=extension, dataarr=dataarr)

    # deals with possible pyfits bug?
    #if dataarr.sum() == 0 or dataarr[-1,:].sum() == 0:
    #    print "Reading file using pfits because pyfits didn't read any values!"
    #    import pfits
    #    if datapfits is not None:
    #        data = datapfits
    #    else:
    #        data = pfits.FITS(fitsfile).get_hdus()[1].get_data()

    #    dataarr = np.reshape(data['DATA'],data['DATA'].shape[::-1])

    #    namelist = data.keys()

    OK, OKsource = selectsource(data, sampler, feednum=feednum,
                                sourcename=sourcename, scanrange=scanrange)

    nspec = OKsource.sum()
    if nspec == 0:
        raise ValueError("No matches found for source %s in scan range %i:%i" % (sourcename,scanrange[0],scanrange[1]))

    print "Beginning scan selection and calibration for sampler %s and feed %s with %i spectra" % (sampler,feednum,nspec)

    CalOff = (data['CAL']=='F')
    CalOn  = (data['CAL']=='T')

    if CalOff.sum() == 0:
        raise ValueError("No cal-off found: you're probably working with reduced instead of raw data")
    if CalOn.sum() == 0:
        raise ValueError("No cal-on found")

    speclen = dataarr.shape[1]

    scan_means_on = dataarr[OKsource*CalOn].mean(axis=1)
    scan_means_off = dataarr[OKsource*CalOff].mean(axis=1)
    medon = np.percentile(dataarr[OKsource*CalOn].T / scan_means_on, percentile, axis=1)
    medoff = np.percentile(dataarr[OKsource*CalOff].T / scan_means_off, percentile, axis=1)
    off_template = np.mean([medon,medoff],axis=0)

    velo = velo_iterator(data,linefreq=linefreq).next()
    OKvelo = (velo > interp_vrange[0]) * (velo < interp_vrange[1]) 
    nOKvelo = np.zeros(velo.size,dtype='bool')
    for low,high in zip(*[iter(exclude_velo)]*2):
        OKvelo[(velo > low) * (velo < high)] = False
        nOKvelo[(velo > low) * (velo < high)] = True

    polypars = np.polyfit( np.arange(velo.size)[OKvelo], off_template[OKvelo],
            interp_polyorder)

    if return_uninterp:
        off_template_in = np.copy(off_template)
    # replace the "not OK" regions with the interpolated values
    off_template[nOKvelo] = np.polyval(polypars, np.arange(velo.size)[nOKvelo]).astype(off_template.dtype)

    if np.any(np.isnan(off_template)):
        raise ValueError("Invalid off: contains nans.")

    if return_uninterp:
        return off_template,off_template_in
    else:
        return off_template