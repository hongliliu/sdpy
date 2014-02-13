from .calibrate_map_scans import load_data_file
from .makecube import selectsource,velo_iterator
import numpy as np

from astropy.io import fits
from astropy import units as u

type_to_ctype = {'length':'WAVE',
                 'frequency':'FREQ',
                 'speed':'VELO',
                }

def generate_1d_header_fromdisparray(arr, cdelt_tolerance=1e-8, reference=None,
                                     unit=None):
    """
    Parameters
    ----------
    cdelt_tolerance : float
        Tolerance in the difference between pixels that determines
        how near to linear the dispersion axis must be
    """
    header = fits.Header()

    # convert array to numpy array with no units
    if hasattr(arr,'unit'):
        if unit is None:
            unit = arr.unit
        arr = arr.value

    # convert string to unit
    if not hasattr(unit,'to_string'):
        unit = u.Unit(unit)
    
    # determine CDELT
    dxarr = np.diff(arr)
    if abs(dxarr.max()-dxarr.min())/abs(dxarr.min()) < cdelt_tolerance:
        cdelt = dxarr.mean().flat[0]
    else:
        raise ValueError("Dispersion array is not linear.")

    # determine CRVAL, CRPIX
    crval = arr.min()
    crpix = 1

    if reference is not None:
        restfrq = reference.to(u.Hz, u.spectral())
        header['RESTFRQ'] = restfrq.value

    header['CRVAL1'] = crval
    header['CDELT1'] = cdelt
    header['CRPIX1'] = crpix
    header['CUNIT1'] = unit.to_string()
    header['CTYPE1'] = type_to_ctype[unit.physical_type]

    return header

def make_off(fitsfile, scanrange=[], sourcename=None, feednum=1, sampler=0,
             exclude_spectral_ends=False,
             savefile=None,
             dataarr=None, obsmode=None, exclude_velo=(), interp_polyorder=5,
             return_poly=False,
             extension=1,
             percentile=50, interp_vrange=(), linefreq=None, return_uninterp=False,
             clobber=False):
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
    exclude_spectral_ends: bool or float
        If a float, indicates the percent of the start & end of the spectra to
        exclude when computing the mean (i.e., the continuum level).  Note that
        this is a PERCENT, so it should be in the range [0,100]
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
    savefile : None or str
        Optional save file name *prefix* to which the string
        "_offspectra.fits" will be appended.  This file will contain
        two spectra, one with interpolation and one without
    clobber : bool
        Overwrite savefile?

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

    # compute mean of each spectrum (this is a continuum time-series)
    # Start by optionally excluding the ends of the spectrum, where signal is usually forced to zero
    if exclude_spectral_ends:
        endsslice = slice(speclen*exclude_spectral_ends/100.,
                          -speclen*exclude_spectral_ends/100.)
    else:
        endsslice = slice(None)

    scan_means_on = dataarr[OKsource*CalOn][:,endsslice].mean(axis=1)
    scan_means_off = dataarr[OKsource*CalOff][:,endsslice].mean(axis=1)

    # Divide by the continuum before taking the median across time
    # (note the transpose .T, which is why axis=1 works here too)
    # Without normalizing by the continuum, the median is meaningless
    medon = np.percentile(dataarr[OKsource*CalOn].T / scan_means_on, percentile, axis=1)
    medoff = np.percentile(dataarr[OKsource*CalOff].T / scan_means_off, percentile, axis=1)
    # the off template is then the mean of the caloff + calon scans
    # (it will be normalized, so this just increases the S/N)
    off_template = np.mean([medon,medoff],axis=0)

    if return_uninterp:
        off_template_in = np.copy(off_template)

    if interp_vrange:
        # interpolate across the excluded regions
        velo = velo_iterator(data,linefreq=linefreq).next()
        OKvelo = (velo > interp_vrange[0]) * (velo < interp_vrange[1]) 
    if exclude_velo:
        nOKvelo = np.zeros(velo.size,dtype='bool')
        for low,high in zip(*[iter(exclude_velo)]*2):
            OKvelo[(velo > low) * (velo < high)] = False
            nOKvelo[(velo > low) * (velo < high)] = True

        if OKvelo.sum() > interp_polyorder:
            polypars = np.polyfit(np.arange(velo.size)[OKvelo],
                                  off_template[OKvelo],
                                  interp_polyorder)
        else:
            raise ValueError("Polynomial order to be fitted is greater than the number of points to be fitted.")

        # replace the "not OK" regions with the interpolated values
        off_template[nOKvelo] = np.polyval(polypars, np.arange(velo.size)[nOKvelo]).astype(off_template.dtype)
    elif return_poly and interp_vrange:
        polypars = np.polyfit(np.arange(velo.size)[OKvelo],
                              off_template[OKvelo],
                              interp_polyorder)
    elif return_poly:
        polypars = np.polyfit(np.arange(velo.size),
                              off_template,
                              interp_polyorder)

    if np.any(np.isnan(off_template)):
        raise ValueError("Invalid off: contains nans.")


    return_vals = off_template,
    if return_uninterp:
        return_vals = return_vals + (off_template_in,)
    if return_poly:
        return_vals = return_vals + (np.polyval(polypars, np.arange(velo.size)).astype(off_template.dtype),)


    if savefile:
        if not 'velo' in locals():
            velo = velo_iterator(data,linefreq=linefreq).next()
        header = generate_1d_header_fromdisparray(velo*u.km/u.s,
                                                  reference=linefreq*u.Hz if linefreq is not None else None)
        outf = fits.PrimaryHDU(data=np.array(return_vals),
                               header=header)
        outf.writeto(savefile+"_offspectra.fits", clobber=clobber)


    if len(return_vals) == 1:
        return return_vals[0]
    else:
        return return_vals
