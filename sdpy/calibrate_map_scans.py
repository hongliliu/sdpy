import astropy.io.fits as pyfits
import numpy as np
import warnings

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
            print "Treating file as an open FITS HDU",
            datapyfits = filename[extension].data
        except AttributeError:
            print "File is not an HDU.  Reading file from disk using pyfits...",
            if isinstance(filename,str):
                filepyfits = pyfits.open(filename,memmap=True)
                datapyfits = filepyfits[extension].data
            else:
                print "Assuming file is a FITS BinaryTableHDU"
                datapyfits = filename
    if dataarr is None:
        dataarr = datapyfits['DATA']
    print "Data successfully read"
    namelist = datapyfits.names
    data = datapyfits

    if dataarr.sum() == 0 or dataarr[-1,:].sum() == 0:
        print "Reading file using pfits because pyfits didn't read any values!"
        import pfits
        if datapfits is not None:
            data = datapfits
        else:
            data = pfits.FITS(filename).get_hdus()[1].get_data()

        dataarr = np.reshape(data['DATA'],data['DATA'].shape[::-1])

        namelist = data.keys()

    return data, dataarr, namelist, filepyfits

@print_timing
def calibrate_cube_data(filename, outfilename, scanrange=[], 
                        sourcename=None, feednum=1, sampler=0,
                        return_data=False, datapfits=None, dataarr=None,
                        clobber=True, tau=0.0, obsmode=None, refscans=None,
                        off_template=None, filepyfits=None,
                        refscan1=None, refscan2=None,
                        exclude_spectral_ends=10., extension=1):
    """

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
    off_template : np.ndarray
        A spectrum representing the 'off' position generated by make_off
        (normalized!)
    refscans: list
        Reference scans at beginning & end of observation (probably).  These
        are used as a spectroscopic off template unless off_template is
        specified, but they are always used to determine the continuum
        "zero-point"
    exclude_spectral_ends: float
        PERCENT (range [0,100]) of the spectrum to exclude at either end when
        computing TSYS etc.
    min_scale_reference: False or float
        EXPERIMENTAL: rescale the "reference" to be the scan of lowest TSYS,
        then use the value of min_scale_reference as a percentile to determine
        the integration to use from that scan.  Try 10.
    """

    if refscan1 is not None or refscan2 is not None:
        warnings.warn("Use refscans (a list of scans) rather than ref1,ref2",
                      warnings.DeprecationWarning)
        if (type(refscans) == list and not 
            (len(refscans) ==2 and 
             refscans[0] == refscan1 and refscans[1] == refscan2)):
            raise ValueError('refscans does not match refscan1,2')
        elif refscans is None:
            refscans = refscan1,refscan2

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
    OK *= data['FEED'] == feednum
    OK *= np.isfinite(data['DATA'].sum(axis=1))
    OKsource = OK.copy()
    if sourcename is not None:
        OKsource *= (data['OBJECT'] == sourcename)
    if scanrange is not []:
        OKsource *= (scanrange[0] < data['SCAN'])*(data['SCAN'] < scanrange[1])
    if obsmode is not None:
        OKsource *= ((obsmode == data.OBSMODE) + ((obsmode+":NONE:TPWCAL") == data.OBSMODE))
    if sourcename is None and scanrange is None:
        raise IndexError("Must specify a source name and/or a scan range")

    print "Beginning scan selection and calibration for sampler %s and feed %s" % (sampler,feednum)

    CalOff = (data['CAL']=='F')
    CalOn  = (data['CAL']=='T')

    speclen = dataarr.shape[1]

    # Fraction of ends to exclude
    exfrac = exclude_spectral_ends/100.
    exslice = slice(speclen*exfrac,-speclen*exfrac)

    # reference scans define the "background continuum"
    if type(refscans) == list:
        refarray = np.zeros([len(refscans),speclen])
        LSTrefs  = np.zeros([len(refscans)])
        for II,refscan in enumerate(refscans):
            OKref = OK * (refscan == data['SCAN'])  

            specrefon  = np.median(dataarr[OKref*CalOn,:],axis=0) 
            specrefoff = np.median(dataarr[OKref*CalOff,:],axis=0)
            tcalref    = np.median(data['TCAL'][OKref])
            tsysref    = ( np.mean(specrefoff[exslice]) / 
                    (np.mean((specrefon-specrefoff)[exslice])) * 
                    tcalref + tcalref/2.0 )
            refarray[II] = (specrefon + specrefoff)/2.0
            LSTrefs[II]  = np.mean(data['LST'][OKref])
            if specrefon.sum() == 0 or specrefoff.sum() == 0:
                raise ValueError("All values in reference scan %i are zero" % refscan)
            elif np.isnan(specrefon).sum() > 0 or np.isnan(specrefoff).sum() > 0:
                raise ValueError("Reference scan %i contains a NAN" % refscan)

    else:
        raise TypeError("Must specify reference scans as a list of scan numbers.")

    print "Beginning calibration of %i scans." % ((OKsource*CalOn).sum())

    if ((OKsource*CalOn).sum()) == 0:
        import pdb; pdb.set_trace()

    # compute TSYS on a scan-by-scan basis to avoid problems with saturated TSYS.
    scannumbers = np.unique(data['SCAN'][OKsource])
    for scanid in scannumbers:
        whscan = data['SCAN'] == scanid

        on_data = dataarr[whscan*CalOn,exslice]
        off_data = dataarr[whscan*CalOff,exslice]
        tcal = np.median(data['TCAL'][whscan])

        offmean = np.median(off_data,axis=0).mean()
        onmean  = np.median(on_data,axis=0).mean()
        diffmean = onmean-offmean

        tsys = ( offmean / diffmean * tcal + tcal/2.0 )
        print "Scan %4i:  TSYS=%12.3f" % (scanid,tsys)
        data['TSYS'][whscan] = tsys

    # experimental: try to rescale the "reference" scan to be the minimum
    if min_scale_reference:
        min_tsys = np.argmin(data['TSYS'])
        whmin = data['SCAN'][min_tsys]
        whscan = data['SCAN'] == whmin
        r1 = np.percentile(dataarr[whscan*OKsource*CalOn,exslice], min_scale_reference, axis=0)
        r2 = np.percentile(dataarr[whscan*OKsource*CalOff,exslice], min_scale_reference, axis=0)
        ref_scale = np.median((r1+r2)/2.0)
        print "EXPERIMENTAL: min_scale_reference = ",ref_scale
    
    for specindOn,specindOff in zip(np.where(OKsource*CalOn)[0],np.where(OKsource*CalOff)[0]):

        for K in namelist:
            if K != 'DATA':
                newdatadict[K].append(data[K][specindOn])
            else:
                # should this be speclen or 4096?  Changing to speclen...
                newdatadict['DATA'].append(np.zeros(speclen))

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
            print "Rescaling specRef from ",specRef[exslice].mean()," to ",ref_scale
            specRef = specRef/specRef[exslice].mean() * ref_scale

        # use a templated OFF spectrum
        # (e.g., one that has had spectral lines interpolated over)
        if off_template is not None:
            if off_template.shape != specRef.shape:
                raise ValueError("Off template shape does not match spectral shape")
            # exclude spectral ends when ratio-ing
            specRef = off_template * specRef[exslice].mean() / off_template[exslice].mean()

        tsys = data['TSYS'][specindOn]

        calSpec = (spec-specRef)/specRef * tsys * np.exp(tau)
        if calSpec.sum() == 0:
            raise ValueError("All values in calibrated spectrum are zero")

        newdatadict['TSYS'][-1] = tsys
        newdatadict['DATA'][-1] = calSpec

    # how do I get the "Format" for the column definitions?

    # Make Table
    cols = [pyfits.Column(name=key,format=formatdict[key],array=value)
        for key,value in newdatadict.iteritems()]
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


