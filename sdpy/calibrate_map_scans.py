from __future__ import print_function
import astropy.io.fits as pyfits
import numpy as np
import warnings
from astropy import log
from astropy.utils.console import ProgressBar

from .timer import print_timing


def load_data_file(filename, extension=1, dataarr=None, filepyfits=None,
                   datapfits=None):
    """
    Load the series of spectra from a raw SDFITS file
    """

    if filepyfits is not None:
        datapyfits = filepyfits[extension].data
    else:
        try:
            print("Treating file as an open FITS HDU... ", end=' ')
            datapyfits = filename[extension].data
        except AttributeError:
            print("File is not an HDU. Reading file from disk using pyfits...",
                  end=' ')
            if isinstance(filename, str):
                filepyfits = pyfits.open(filename, memmap=True)
                datapyfits = filepyfits[extension].data
            else:
                print("Assuming file is a FITS BinaryTableHDU")
                datapyfits = filename
    if dataarr is None:
        dataarr = datapyfits['DATA']
    print("Data successfully read")
    namelist = datapyfits.names
    data = datapyfits

    return data, dataarr, namelist, filepyfits


def compute_gains_highfreq(data, feednum=1, sampler=0, tcold=50.):
    """
    Compute all gains as a function of time for a feed / sampler
    """

    OBSMODE = np.core.defchararray.rstrip(data['OBSMODE'])

    calseqs = ((OBSMODE == 'CALSEQ:NONE:TPNOCAL') &
               (data['FEED'] == feednum) &
               (data['SAMPLER'] == sampler)
               )

    if np.count_nonzero(calseqs) == 0:
        raise ValueError("No matching feeds or observation types for feed {0}"
                         " and sampler {1}"
                         .format(feednum, sampler))

    twarm = data['TWARM'][calseqs]

    scans = list(set(data['SCAN'][calseqs]))

    firstscans = [sn for sn in scans
                  if sn+1 in scans and sn+2 in scans]

    results = {}

    for scan in firstscans:
        sky = data['DATA'][data['SCAN'] == scan].mean(axis=0)
        tem1 = data['DATA'][data['SCAN'] == scan+1].mean(axis=0)
        tem2 = data['DATA'][data['SCAN'] == scan+2].mean(axis=0)
        if feednum == 2:
            warm_meas, cold_meas = tem1, tem2
        elif feednum == 1:
            warm_meas, cold_meas = tem2, tem1
        else:
            raise ValueError("Feed must be 1 or 2")

        warmload = np.median(data['TWARM'][data['SCAN'] == scan])
        coldload = tcold

        gain = (warmload-coldload)/np.median(warm_meas-cold_meas)
        tsys = np.median(gain*sky)

        time = data['LST'][data['SCAN'] == scan][0]
        source = data['OBJECT'][data['SCAN'] == scan][0]
        results[time] = (gain, tsys, source)

    return results


@print_timing
def calibrate_cube_data(filename, outfilename, scanrange=[],
                        sourcename=None, feednum=1, sampler=0,
                        return_data=False, datapfits=None, dataarr=None,
                        clobber=True, tau=0.0, obsmode=None, refscans=None,
                        tauz=0.0,
                        off_template=None, filepyfits=None,
                        refscan1=None, refscan2=None,
                        exclude_spectral_ends=10., extension=1,
                        min_scale_reference=False,
                        verbose=1,
                        tsysmethod='perscan',
                        tatm=273.0,
                        trec=None,
                        airmass_method='maddalena',
                        scale_airmass=True,
                        tsys=None,
                        gain=None,
                        highfreq=False,
                        ):
    """
    The calibration process in pseudocode:

    # Create a "reference spectrum" on blank sky
    refspec = mean(refspec_calon,refspec_caloff)
    # Determine TSYS, the total atmospheric + astrophysical + receiver noise
    # temperature
    tsys = tcal * mean_continuum(cal_off) / (mean_continuum(cal_on - cal_off)) + tcal/2.0

    # for "TOTAL POWER" mode
    # Remove the atmospheric and receiver contribution
    tsource = tsys - (np.exp(tau*airmass)-1)*tatm - trec
    tsource_star = tsource * np.exp(tau*airmass)

    # for SIG-REF mode (signal + reference; on/off nod)
    calSpec = (spec-specRef)/specRef * tsys


    How does GBTIDL do it?

     * in dcmeantsys.pro_, the +/-10% edge channels are excluded, and
        mean_tsys = mean(cal_off) / (mean(cal_on-cal_off)) * tcal + tcal/2.0
     * in dototalpower.pro_, the data is set to tp_data = (cal_on + cal_off) / 2.0
     * in dofullsigref.pro_, tsys is corrected with airmass=1/sin(elev)
     * in dosigref.pro_, sigrefdata = (tpdata - refdata)/refdata * tsys


    .. _dofullsigref.pro: http://www.gb.nrao.edu/GBT/DA/gbtidl/release/user/toolbox/dofullsigref.html
    .. _dcmeantsys.pro: http://www.gb.nrao.edu/GBT/DA/gbtidl/release/user/toolbox/dcmeantsys.html
    .. _dototalpower.pro: http://www.gb.nrao.edu/GBT/DA/gbtidl/release/user/toolbox/dototalpower.html
    .. _dosigref.pro: http://www.gb.nrao.edu/GBT/DA/gbtidl/release/user/toolbox/dosigref.html

    Parameters
    ----------
    filename : str
        input file name
    outfilename : str
        output file name
    scanrange : 2-tuple
        *DATA SELECTION PARAMETER* Range of scans to include when creating off
        positions
    sourcename : str or None
        *DATA SELECTION PARAMETER* Name of source to include
    feednum : int
        *DATA SELECTION PARAMETER* Feed number to use (1 for single-feed
        systems)
    sampler : str
        *DATA SELECTION PARAMETER* Sampler to create the off for (e.g., 'A9')
    obsmode : str
        *DATA SELECTION PARAMETER* Observation Mode to include (e.g.,
        DecLatMap)
    dataarr : None or np.ndarray
        OPTIONAL input of data array.  If it has already been read, this param
        saves time
    tau : float
        Zenith optical depth
        [Deprecated, use tauz]
    tauz : float
        Zenith optical depth
    off_template : np.ndarray
        A spectrum representing the 'off' position generated by make_off
        (normalized!)
    refscans : list
        Reference scans at beginning & end of observation (probably).  These
        are used as a spectroscopic off template unless off_template is
        specified, but they are always used to determine the continuum
        "zero-point"
    exclude_spectral_ends : float
        PERCENT (range [0,100]) of the spectrum to exclude at either end when
        computing TSYS etc.
    min_scale_reference : False or float
        EXPERIMENTAL: rescale the "reference" to be the scan of lowest TSYS,
        then use the value of min_scale_reference as a percentile to determine
        the integration to use from that scan.  Try 10.
        WARNING: Can cause major problems if obserations occur at highly
        variable airmass!
    tsysmethod : 'perscan' or 'perint'
        Compute tsys for each scan or for each integration?
    verbose: int
        Level of verbosity.  0 is none, 1 is some, 2 is very, 3 is very lots
    tatm : float
        The atmospheric temperature.  Will be subtracted from TSYS using the
        provided optical depth and assuming a plane-parallel atmosphere,
        which is good down to 8 degrees
        (http://www.gb.nrao.edu/~rmaddale/GBT/HighPrecisionCalibrationFromSingleDishTelescopes.pdf)
    trec : float
        Not presently used
    airmass_method : float
        Method of airmass determination, either csc(elev) or Ron Maddalena's
        version
    scale_airmass : bool
        If the min_scale_reference method is used, try to re-scale the off
        position mean by the airmass
    tsys : None or float
        Manually specified system temperature.  Defaults to None, which means
        it will be computed.  This is intended for high-frequency (W-band)
        observing.
    gain : None or float
        Manually specified gain.  Defaults to None.  This is intended for
        high-frequency (W-band) observing.
    highfreq : bool
        Is the data high freqency?  If True, there should be no noise
        calibrator, so calibration requires gain to be specified.

    """

    if tau != 0:
        if tauz != 0:
            raise ValueError("Only use tauz, not both tau and tauz")
        else:
            tauz = tau
            warnings.warn("Use tauz instead of tau",
                          DeprecationWarning)

    if refscan1 is not None or refscan2 is not None:
        warnings.warn("Use refscans (a list of scans) rather than ref1,ref2",
                      DeprecationWarning)
        if (type(refscans) == list and not
            (len(refscans) ==2 and
             refscans[0] == refscan1 and refscans[1] == refscan2)):
            raise ValueError('refscans does not match refscan1,2')
        elif refscans is None:
            refscans = refscan1,refscan2

    if highfreq:
        assert gain is not None

    data, dataarr, namelist, filepyfits = load_data_file(filename,
                                                         extension=extension,
                                                         dataarr=dataarr,
                                                         datapfits=datapfits,
                                                         filepyfits=filepyfits)

    newdatadict = dict([(n,[]) for n in namelist])
    formatdict = dict([(t.name,t.format) for t in filepyfits[extension].columns])

    samplers = np.unique(data['SAMPLER'])
    if isinstance(sampler,int):
        sampler = samplers[sampler]

    OK = data['SAMPLER'] == sampler
    if np.count_nonzero(OK) == 0:
        raise ValueError("No matches to sampler {0}".format(sampler))
    OK *= data['FEED'] == feednum
    if np.count_nonzero(OK) == 0:
        raise ValueError("No matches to sampler {0} and feed {1}"
                         .format(sampler, feednum))
    OK *= np.isfinite(data['DATA'].sum(axis=1))
    if np.count_nonzero(OK) == 0:
        raise ValueError("There is no finite data.")
    OKsource = OK.copy()
    if sourcename is not None:
        OKsource &= (data['OBJECT'] == sourcename)
        if np.count_nonzero(OKsource) == 0:
            raise ValueError("Object {0} not in data".format(sourcename))
    if scanrange is not []:
        OKsource &= (scanrange[0] < data['SCAN'])&(data['SCAN'] < scanrange[1])
        if np.count_nonzero(OKsource) == 0:
            raise ValueError("No scans in range {0}-{1}".format(scanrange[0],
                                                                scanrange[1]))
    if obsmode is not None:
        OBSMODE = np.core.defchararray.rstrip(data.OBSMODE)
        OKsource &= ((obsmode == OBSMODE) |
                     ((obsmode+":NONE:TPWCAL") == OBSMODE) |
                     ((obsmode+":NONE:TPNOCAL") == OBSMODE)
                    )
        if np.count_nonzero(OKsource) == 0:
            raise ValueError("No matches to OBSMODE={0}."
                             "  Valid modes include {1}".format(obsmode,
                                                                set(OBSMODE)))
    if sourcename is None and scanrange is None:
        raise IndexError("Must specify a source name and/or a scan range")

    if verbose:
        log.info("Beginning scan selection and calibration for "
                 "sampler {0} and feed {1}.  Found {2} matching"
                 "scans and {3} with the source {4} in it."
                 .format(sampler,feednum, OK.sum(), OKsource.sum(),
                         sourcename))

    CalOff = (data['CAL']=='F')
    CalOn  = (data['CAL']=='T')

    speclen = dataarr.shape[1]

    # Fraction of ends to exclude.  exslice = "Exclusion Slice"
    exfrac = exclude_spectral_ends/100.
    exslice = slice(speclen*exfrac,-speclen*exfrac)

    # reference scans define the "background continuum"
    if type(refscans) == list:
        if not highfreq:
            # split into two steps for readability
            temp_ref = get_reference(data, refscans, CalOn=CalOn, CalOff=CalOff,
                                     exslice=exslice, OK=OK)
            LSTrefs, refarray, ref_cntstoK, tsysref = temp_ref
        else:
            LSTrefs, refarray = get_reference_highfreq(data, refscans, OK=OK)
    else:
        raise TypeError("Must specify reference scans as a list of scan numbers.")

    if highfreq:
        nscansok = np.count_nonzero(OKsource)
    else:
        nscansok = np.count_nonzero(OKsource*CalOn)

    if verbose:
        log.info("Beginning calibration of %i scans." % nscansok)

    if nscansok == 0:
        import pdb; pdb.set_trace()
        raise ValueError("There are no locations where the source was observed"
                         " with the calibration diode on.  That can't be right.")

    if tsys is None:
        compute_tsys(data, tsysmethod=tsysmethod, OKsource=OKsource, CalOn=CalOn,
                     CalOff=CalOff, exslice=exslice, verbose=verbose)
    else:
        data['TSYS'] = tsys

    # experimental: try to rescale the "reference" scan to be the minimum
    if min_scale_reference:
        ref_scale,ref_airmass = get_min_scale_reference(data,
                                                        min_scale_reference,
                                                        OKsource=OKsource,
                                                        CalOn=CalOn,
                                                        CalOff=CalOff,
                                                        exslice=exslice,
                                                        airmass_method=airmass_method)
        if verbose:
            log.info("EXPERIMENTAL: min_scale_reference = {0}".format(ref_scale))

    if highfreq:
        newdatadict = cal_loop_highfreq(data, dataarr, newdatadict, OKsource,  speclen,
                                        airmass_method, LSTrefs, exslice,
                                        refscans, namelist, refarray, off_template, gain)
    else:
        newdatadict = cal_loop_lowfreq(data, dataarr, newdatadict, OKsource, CalOn,
                                       CalOff, speclen, airmass_method,
                                       LSTrefs, min_scale_reference, exslice,
                                       tatm, tauz, refscans, namelist,
                                       refarray, off_template)

    # how do I get the "Format" for the column definitions?

    # Make Table
    cols = [pyfits.Column(name=key,format=formatdict[key],array=value)
            for key,value in newdatadict.items()]
    colsP = pyfits.ColDefs(cols)
    #tablehdu = copy.copy(filepyfits[extension])
    #tablehdu.data = colsP
    # this lies and claims corrupted
    tablehdu = pyfits.new_table(colsP, header=filepyfits[extension].header)
    phdu = pyfits.PrimaryHDU(header=filepyfits[0].header)
    hdulist = pyfits.HDUList([phdu,tablehdu])
    hdulist.writeto(outfilename,clobber=clobber)

    #tablehdu.writeto(outfilename,clobber=clobber)

    if return_data:
        return filepyfits,data,colsP


def compute_tsys(data, tsysmethod='perscan', OKsource=None, CalOn=None,
                 CalOff=None, verbose=False, exslice=slice(None)):
    """
    Calculate the TSYS vector for a set of scans

    from GBTIDL's dcmeantsys.py
    ;  mean_tsys = tcal * mean(nocal) / (mean(withcal-nocal)) + tcal/2.0

    """
    if CalOn is None:
        CalOn  = (data['CAL']=='T')
    if CalOff is None:
        CalOff = (data['CAL']=='F')

    dataarr = data['DATA']

    if OKsource is None:
        OKsource = np.ones(dataarr.shape[0], dtype='bool')

    if tsysmethod == 'perscan':
        # compute TSYS on a scan-by-scan basis to avoid problems with saturated
        # TSYS.
        scannumbers = np.unique(data['SCAN'][OKsource])
        for scanid in scannumbers:
            whscan = data['SCAN'] == scanid

            on_data = dataarr[whscan & CalOn,exslice]
            off_data = dataarr[whscan & CalOff,exslice]
            tcal = np.median(data['TCAL'][whscan])

            offmean = np.median(off_data,axis=0).mean()
            onmean  = np.median(on_data,axis=0).mean()
            diffmean = onmean-offmean

            tsys = (offmean / diffmean * tcal + tcal/2.0)
            if verbose > 1:
                print("Scan %4i:  TSYS=%12.3f" % (scanid,tsys))
            data['TSYS'][whscan] = tsys
    elif tsysmethod == 'perint':
        on_data = dataarr[CalOn & OKsource,exslice]
        off_data = dataarr[CalOff & OKsource,exslice]
        tcal = data['TCAL'][CalOn & OKsource]

        offmean = np.mean(off_data,axis=1)
        onmean  = np.mean(on_data,axis=1)
        diffmean = onmean-offmean

        # K / count = tcal / diffmean
        tsys = (offmean / diffmean * tcal + tcal/2.0)
        data['TSYS'][CalOn & OKsource] = tsys
        data['TSYS'][CalOff & OKsource] = tsys

    return data['TSYS']

def elev_to_airmass(elev, method='maddalena'):
    """
    Calculate the airmass with np.csc(elev) or Ron Maddalena's improved method
    for low elevations

    http://www.gb.nrao.edu/~rmaddale/GBT/HighPrecisionCalibrationFromSingleDishTelescopes.pdf
    """
    if method != 'maddalena':
        return 1/np.sin(elev/180*np.pi)
    else:
        # http://www.gb.nrao.edu/~rmaddale/GBT/HighPrecisionCalibrationFromSingleDishTelescopes.pdf
        return -0.0234+1.014/np.sin((elev+5.18/(elev+3.35))*np.pi/180.)

def get_min_scale_reference(data, min_scale_reference, OKsource=None,
                            CalOn=None, CalOff=None, verbose=False,
                            exslice=slice(None), airmass_method='maddalena'):

    if CalOn is None:
        CalOn  = (data['CAL']=='T')
    if CalOff is None:
        CalOff = (data['CAL']=='F')

    min_tsys = np.argmin(data['TSYS'][OKsource])
    whmin = data['SCAN'][OKsource][min_tsys]
    whscan = data['SCAN'] == whmin
    dataarr = data['DATA']
    r1 = np.percentile(dataarr[whscan*OKsource*CalOn,exslice],
                       min_scale_reference, axis=0)
    r2 = np.percentile(dataarr[whscan*OKsource*CalOff,exslice],
                       min_scale_reference, axis=0)
    ref_scale = np.median((r1+r2)/2.0)
    ref_airmass = elev_to_airmass(data['ELEVATIO'][OKsource][min_tsys],
                                  method=airmass_method)

    return ref_scale,ref_airmass

def get_reference_highfreq(data, refscans, OK=None):
    """
    Extract the reference scans from the data, but don't try to calibrate them
    with the noise calibrator (because we're doing high frequency)
    """
    dataarr = data['DATA']
    speclen = dataarr.shape[1]

    refarray = np.zeros([len(refscans),speclen])
    LSTrefs  = np.zeros([len(refscans)])
    for II,refscan in enumerate(refscans):
        OKref = OK & (refscan == data['SCAN'])
        # use "where" in case that reduces amount of stuff read in...
        CalOnRef = np.nonzero(OKref)[0]

        specrefon  = np.median(dataarr[CalOnRef,:],axis=0)

        refarray[II] = specrefon
        LSTrefs[II]  = np.mean(data['LST'][OKref])
        if specrefon.sum() == 0:
            raise ValueError("All values in reference scan %i are zero" % refscan)
        elif np.isnan(specrefon).sum() > 0:
            raise ValueError("Reference scan %i contains a NAN" % refscan)

    return LSTrefs, refarray

def get_reference(data, refscans, CalOn=None, CalOff=None,
                  exslice=slice(None), OK=None):
    """
    Extract the reference scans from the data.

    Parameters
    ----------
    data: FITS table
        Table of the data and associated metadata
    refscans: list
        List of scan numbers
    CalOn/CalOff: boolean arrays
        Optional; if not specified they will be recomputed.  Boolean arrays
        identified the on/off regions of the data
    exslice: slice
        Slice along the spectral axis for computing means
    OK: boolean array
        Mandatory.  All valid spectra for the specified feed and sampler
    """
    if CalOn is None:
        CalOn  = (data['CAL']=='T')
    if CalOff is None:
        CalOff = (data['CAL']=='F')

    dataarr = data['DATA']
    speclen = dataarr.shape[1]

    refarray = np.zeros([len(refscans),speclen])
    LSTrefs  = np.zeros([len(refscans)])
    for II,refscan in enumerate(refscans):
        OKref = OK & (refscan == data['SCAN'])
        if np.count_nonzero(OKref) == 0:
            raise ValueError("No 'OK' data for scan {0}".format(refscan))
        # use "where" in case that reduces amount of stuff read in...
        CalOnRef = np.nonzero(OKref & CalOn)[0]
        CalOffRef = np.nonzero(OKref & CalOff)[0]

        specrefon  = np.median(dataarr[CalOnRef,:],axis=0)
        specrefoff = np.median(dataarr[CalOffRef,:],axis=0)
        tcalref    = np.median(data['TCAL'][OKref])
        ref_cntstoK = tcalref/np.mean((specrefon-specrefoff)[exslice])
        #tsysref    = ( np.mean(specrefoff[exslice]) /
        #        (np.mean((specrefon-specrefoff)[exslice])) *
        #        tcalref + tcalref/2.0 )
        tsysref = np.mean(specrefoff[exslice]) * ref_cntstoK + tcalref/2.0
        refarray[II] = (specrefon + specrefoff)/2.0
        LSTrefs[II]  = np.mean(data['LST'][OKref])
        if specrefon.sum() == 0 or specrefoff.sum() == 0:
            raise ValueError("All values in reference scan %i are zero" % refscan)
        elif np.isnan(specrefon).sum() > 0 or np.isnan(specrefoff).sum() > 0:
            raise ValueError("Reference scan %i contains a NAN" % refscan)

    return LSTrefs, refarray, ref_cntstoK, tsysref

def cal_loop_lowfreq(data, dataarr, newdatadict, OKsource, CalOn, CalOff,
                     speclen, airmass_method, LSTrefs, min_scale_reference,
                     exslice, tatm, tauz, refscans, namelist, refarray,
                     off_template):

    for specindOn,specindOff in zip(np.where(OKsource*CalOn)[0],
                                    np.where(OKsource*CalOff)[0]):

        for K in namelist:
            if K != 'DATA':
                newdatadict[K].append(data[K][specindOn])
            else:
                # should this be speclen or 4096?  Changing to speclen...
                newdatadict['DATA'].append(np.zeros(speclen))

        # http://www.gb.nrao.edu/~rmaddale/Weather/
        elev = data['ELEVATIO'][specindOn]
        airmass = elev_to_airmass(elev,
                                  method=airmass_method)

        specOn = dataarr[specindOn,:]
        specOff = dataarr[specindOff,:]
        spec = (specOn + specOff)/2.0
        LSTspec = data['LST'][specindOn]

        # this "if" test is no longer necessary
        if refscans is not None:
            # find the reference scan closest to the current scan
            # (LSTspec is a number, LSTrefs is an array, probably length 2)
            refscannumber = np.argmin(np.abs(LSTspec-LSTrefs))
            # if the closest reference scan is the last or it is after the spectrum...
            # the earlier reference scan has index self-1
            if refscannumber == len(refscans) - 1 or LSTrefs[refscannumber] > LSTspec:
                r1 = refscannumber - 1
                r2 = refscannumber
            elif LSTrefs[refscannumber] < LSTspec:
                r1 = refscannumber
                r2 = refscannumber + 1
            LSTref1 = LSTrefs[r1]
            LSTref2 = LSTrefs[r2]
            specref1 = refarray[r1,:]
            specref2 = refarray[r2,:]
            LSTspread = LSTref2-LSTref1

        # LINEAR interpolation between the reference scans
        specRef = (specref2-specref1)/LSTspread*(LSTspec-LSTref1) + specref1
        # EXPERIMENTAL
        if min_scale_reference:
            if verbose > 2:
                log.info("Rescaling specRef from {0} to {1}"
                         .format(specRef[exslice].mean(),ref_scale))
            specRef = specRef/specRef[exslice].mean() * ref_scale
            if scale_airmass:
                specRef += tatm/ref_cntstoK*(np.exp(-tauz*ref_airmass)-np.exp(-tauz*airmass))


        # use a templated OFF spectrum
        # (e.g., one that has had spectral lines interpolated over)
        if off_template is not None:
            if off_template.shape != specRef.shape:
                raise ValueError("Off template shape does not match spectral shape")
            # exclude spectral ends when ratio-ing
            specRef = off_template * specRef[exslice].mean() / off_template[exslice].mean()

        tsys = data['TSYS'][specindOn]

        # I don't think this is right... the correct way is to make sure
        # specRef moves with Spec
        #tsys_eff = tsys * np.exp(tau*airmass) - (np.exp(tau*airmass)-1)*tatm
        tsys_eff = tsys * np.exp(tauz*airmass)

        calSpec = (spec-specRef)/specRef * tsys_eff
        if calSpec.sum() == 0:
            raise ValueError("All values in calibrated spectrum are zero")

        newdatadict['TSYS'][-1] = tsys
        newdatadict['DATA'][-1] = calSpec

        return newdatadict

def cal_loop_highfreq(data, dataarr, newdatadict, OKsource,  speclen,
                      airmass_method, LSTrefs, exslice, refscans, namelist,
                      refarray, off_template, gain):

    inds = np.where(OKsource)[0]
    for specindOn in ProgressBar(inds):

        for K in namelist:
            if K != 'DATA':
                newdatadict[K].append(data[K][specindOn])
            else:
                # should this be speclen or 4096?  Changing to speclen...
                newdatadict['DATA'].append(np.zeros(speclen))

        # http://www.gb.nrao.edu/~rmaddale/Weather/
        elev = data['ELEVATIO'][specindOn]
        airmass = elev_to_airmass(elev,
                                  method=airmass_method)

        spec = specOn = dataarr[specindOn,:]
        LSTspec = data['LST'][specindOn]

        # this "if" test is no longer necessary
        if refscans is not None:
            # find the reference scan closest to the current scan
            # (LSTspec is a number, LSTrefs is an array, probably length 2)
            refscannumber = np.argmin(np.abs(LSTspec-LSTrefs))
            # if the closest reference scan is the last or it is after the spectrum...
            # the earlier reference scan has index self-1
            if refscannumber == len(refscans) - 1 or LSTrefs[refscannumber] > LSTspec:
                r1 = refscannumber - 1
                r2 = refscannumber
            elif LSTrefs[refscannumber] < LSTspec:
                r1 = refscannumber
                r2 = refscannumber + 1
            LSTref1 = LSTrefs[r1]
            LSTref2 = LSTrefs[r2]
            specref1 = refarray[r1,:]
            specref2 = refarray[r2,:]
            LSTspread = LSTref2-LSTref1

        # LINEAR interpolation between the reference scans
        specRef = (specref2-specref1)/LSTspread*(LSTspec-LSTref1) + specref1

        # use a templated OFF spectrum
        # (e.g., one that has had spectral lines interpolated over)
        if off_template is not None:
            if off_template.shape != specRef.shape:
                raise ValueError("Off template shape does not match spectral shape")
            # exclude spectral ends when ratio-ing
            specRef = off_template * specRef[exslice].mean() / off_template[exslice].mean()

        tsys = data['TSYS'][specindOn]

        if isinstance(gain, dict):
            gaintimes = np.array(list(gain.keys()))
            gains = np.array([v[0] for v in list(gain.values())])
            nearest_gain = np.argmin(np.abs(gaintimes-LSTspec))
            if (nearest_gain == 0 or (gaintimes[nearest_gain] < LSTspec and
                                      nearest_gain < len(gaintimes)-1)):
                prev_gain, next_gain = gains[nearest_gain], gains[nearest_gain+1]
                prev_lst, next_lst = gaintimes[nearest_gain], gaintimes[nearest_gain+1]
            elif nearest_gain == len(gaintimes)-1 or gaintimes[nearest_gain] > LSTspec:
                prev_gain, next_gain = gains[nearest_gain-1], gains[nearest_gain]
                prev_lst, next_lst = gaintimes[nearest_gain-1], gaintimes[nearest_gain]
            local_gain = (next_gain-prev_gain)/(next_lst-prev_lst)*(LSTspec-prev_lst) + prev_gain
            calSpec = (spec-specRef)*local_gain
        else:
            calSpec = (spec-specRef)*gain

        if calSpec.sum() == 0:
            raise ValueError("All values in calibrated spectrum are zero")

        newdatadict['TSYS'][-1] = tsys
        newdatadict['DATA'][-1] = calSpec

    return newdatadict
