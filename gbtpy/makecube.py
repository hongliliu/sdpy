try:
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
except ImportError:
    import pyfits
    import pywcs
import coords
import numpy as np
import matplotlib
import pylab
try:
    import aplpy
except ImportError:
    pass
import os
import copy
try:
    from progressbar import ProgressBar
except ImportError:
    pass

import time
from functools import wraps

def print_timing(func):
    @wraps(func)
    def wrapper(*arg,**kwargs):
        t1 = time.time()
        res = func(*arg,**kwargs)
        t2 = time.time()
        print '%s took %0.5g s' % (func.func_name, (t2-t1))
        return res
    return wrapper


def generate_header(centerx, centery, naxis1=64, naxis2=64, naxis3=4096,
        coordsys='galactic', ctype3='VELO-LSR', bmaj=0.138888, bmin=0.138888,
        pixsize=24, cunit3='km/s', output_flatheader='header.txt',
        output_cubeheader='cubeheader.txt', cd3=1.0, crval3=0.0,
        clobber=False, bunit="K", restfreq=None):
    header = pyfits.Header()
    header.update('NAXIS1',naxis1)
    header.update('NAXIS2',naxis2)
    header.update('NAXIS3',naxis3)
    header.update('CD1_1',-1*pixsize/3600.0)
    header.update('CD2_2',pixsize/3600.0)
    header.update('EQUINOX',2000.0)
    header.update('SPECSYS','LSRK')
    header.update('VELREF','257') # CASA convention:
    # VELREF  =                  259 /1 LSR, 2 HEL, 3 OBS, +256 Radio
    # COMMENT casacore non-standard usage: 4 LSD, 5 GEO, 6 SOU, 7 GAL
    if restfreq:
        header.update('RESTFRQ',restfreq)
    if coordsys == 'galactic':
        header.update('CTYPE1','GLON-CAR')
        header.update('CTYPE2','GLAT-CAR')
        header.update('CRVAL1',centerx)
        header.update('CRVAL2',0)
        header.update('CRPIX1',naxis1/2)
        header.update('CRPIX2',naxis2/2-centery/header['CD2_2'])
    if coordsys in ('celestial','radec'):
        header.update('CTYPE1','RA---TAN')
        header.update('CTYPE2','DEC--TAN')
        header.update('CRPIX1',naxis1/2-1)
        header.update('CRPIX2',naxis2/2-1)
        header.update('CRVAL1',centerx)
        header.update('CRVAL2',centery)
    header.update('BMAJ',bmaj)
    header.update('BMIN',bmin)
    header.update('CRPIX3',naxis3/2-1)
    header.update('CRVAL3',crval3)
    header.update('CD3_3',cd3)
    header.update('CTYPE3',ctype3)
    header.update('CUNIT3',cunit3)
    header.update('BUNIT',bunit)
    header.totextfile(output_cubeheader,clobber=clobber)
    del header['NAXIS3']
    del header['CRPIX3']
    del header['CRVAL3']
    del header['CD3_3' ]
    del header['CTYPE3']
    del header['CUNIT3']
    header.totextfile(output_flatheader,clobber=clobber)
    return header

def make_blank_images(cubeprefix, flatheader='header.txt',
        cubeheader='cubeheader.txt', clobber=False):

    flathead = pyfits.Header.fromtextfile(flatheader)
    header = pyfits.Header.fromtextfile(cubeheader)
    naxis1,naxis2,naxis3 = header.get('NAXIS1'),header.get('NAXIS2'),header.get('NAXIS3')
    blankcube = np.zeros([naxis3,naxis2,naxis1])
    blanknhits = np.zeros([naxis2,naxis1])
    print "Blank image size: ",naxis1,naxis2,naxis3,".  Blankcube shape: ",blankcube.shape
    file1 = pyfits.PrimaryHDU(header=header,data=blankcube)
    file1.writeto(cubeprefix+".fits",clobber=clobber)
    file2 = pyfits.PrimaryHDU(header=flathead,data=blanknhits)
    file2.writeto(cubeprefix+"_nhits.fits",clobber=clobber)

def data_iterator(data,continuum=False,fsw=False):
    if hasattr(data,'SPECTRA'):
        shape0 = data.SPECTRA.shape[0]
        shape1 = data.SPECTRA.shape[1]
        for ii in xrange(shape0):
            if continuum:
                yield data.SPECTRA[ii,shape1*0.1:shape1*0.9].mean()
            else:
                if fsw:
                    sign = -1 if data['SIG'][ii] == 'F' else 1
                else:
                    sign = 1
                yield sign * data.SPECTRA[ii,:]
    elif hasattr(data,'DATA'):
        shape0 = data.DATA.shape[0]
        shape1 = data.DATA.shape[1]
        for ii in xrange(shape0):
            if continuum:
                yield data.DATA[ii,shape1*0.1:shape1*0.9].mean()
            else:
                if fsw:
                    sign = -1 if data['SIG'][ii] == 'F' else 1
                else:
                    sign = 1
                yield sign * data.DATA[ii,:]
    else:
        raise Exception("Structure does not have DATA or SPECTRA tags.  Can't use it.  Write your own iterator.")

def coord_iterator(data,coordsys_out='galactic'):
    if hasattr(data,'GLON') and hasattr(data,'GLAT'):
        for ii in xrange(data.GLON.shape[0]):
            if coordsys_out == 'galactic':
                yield data.GLON[ii],data.GLAT[ii]
            elif coordsys_out in ('celestial','radec'):
                pos = coords.Position([data.GLON[ii],data.GLAT[ii]],system='galactic')
                ra,dec = pos.j2000()
                yield ra,dec
    elif hasattr(data,'CRVAL2') and hasattr(data,'CRVAL3'):
        if 'RA' in data.CTYPE2:
            coordsys_in='celestial'
        elif 'GLON' in data.CTYPE2:
            coordsys_in='galactic'
        else:
            raise Exception("CRVAL exists, but RA/GLON not in CTYPE")
        for ii in xrange(data.DATA.shape[0]):
            if coordsys_out == 'galactic' and coordsys_in == 'celestial':
                pos = coords.Position([data.CRVAL2[ii],data.CRVAL3[ii]])
                glon,glat = pos.galactic()
                yield glon,glat
            elif coordsys_out in ('celestial','radec') or coordsys_in==coordsys_out:
                yield data.CRVAL2[ii],data.CRVAL3[ii]
    else:
        raise Exception("No CRVAL or GLON struct in data.")

def velo_iterator(data,linefreq=None):
    for ii in xrange(data.CRPIX1.shape[0]):
        if hasattr(data,'SPECTRA'):
            npix = data.SPECTRA.shape[1]
            CRPIX = data.CRPIX1[ii]
            CRVAL = data.CRVAL1[ii]
            CDELT = data.CDELT1[ii]
            velo = (np.arange(npix)+1-CRPIX)*CDELT + CRVAL
        elif hasattr(data,'DATA'):
            npix = data.DATA.shape[1]
            restfreq = data.RESTFREQ[ii]
            obsfreq  = data.OBSFREQ[ii]
            deltaf   = data.CDELT1[ii]
            sourcevel = data.VELOCITY[ii]
            CRPIX = data.CRPIX1[ii]
            if linefreq is not None:
                # not the right frequency crvalfreq = data.CRVAL1[ii]
                # TEST change made 3/17/2013 - I think should be shifting
                # relative to the observed, rather than the rest freq, since we don't make
                # corrections for LSR.  I undid this, though, since it seemed to erase signal...
                # it may have misaligned data from different sessions.  Odd.
                #freqarr = (np.arange(npix)+1-CRPIX)*deltaf + restfreq # obsfreq #
                # trying again, since 2-2 clearly offset from 1-1
                freqarr = (np.arange(npix)+1-CRPIX)*deltaf + obsfreq
                velo = (linefreq-freqarr)/linefreq * 2.99792458e5
                #obsfreq = data.OBSFREQ[ii]
                #cenfreq = obsfreq + (linefreq-restfreq)
                #crfreq = (CRPIX-1)*deltaf + cenfreq
                #CRVAL = (crfreq - cenfreq)/cenfreq * 2.99792458e5
                #CDELT = -1*deltaf/cenfreq * 2.99792458e5
            else:
                CRVAL = sourcevel/1000.0
                CDELT = -1*deltaf/(obsfreq) * 2.99792458e5
                velo = (np.arange(npix)+1-CRPIX)*CDELT + CRVAL
        yield velo


def make_off(fitsfile, scanrange=[], sourcename=None, feednum=1, sampler=0,
        dataarr=None, obsmode=None, exclude_velo=(), interp_polyorder=5,
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
    try:
        print "Reading file using pyfits",
        filepyfits = pyfits.open(fitsfile,memmap=True)
        datapyfits = filepyfits[extension].data
    except (TypeError,ValueError):
        print "That failed, so trying to treat it as a file...",
        try:
            datapyfits = fitsfile[extension].data
        except AttributeError:
            datapyfits = fitsfile
    if dataarr is None:
        dataarr = datapyfits['DATA']
    print "Data successfully read"
    namelist = datapyfits.names
    data = datapyfits

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
    

def selectsource(data, sampler, sourcename=None, obsmode=None, scanrange=[],
                 feednum=1):

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

    return OK,OKsource

@print_timing
def calibrate_cube_data(filename, outfilename, scanrange=[], refscan1=0,
        refscan2=0, sourcename=None, feednum=1, sampler=0, return_data=False,
        filepyfits=None, datapfits=None, dataarr=None, clobber=True, tau=0.0,
        obsmode=None, refscans=None, off_template=None, flag_neg_tsys=True,
        replace_neg_tsys=False, extension=1):
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
    """

    try:
        print "Reading file using pyfits...",
        filepyfits = pyfits.open(filename,memmap=True)
        datapyfits = filepyfits[extension].data
    except (TypeError,ValueError):
        print "That failed, so trying to treat it as a file...",
        try:
            datapyfits = filename[extension].data
        except AttributeError:
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

    newdatadict = dict([(n,[]) for n in namelist])
    formatdict = dict([(t.name,t.format) for t in filepyfits[extension].columns])

    samplers = np.unique(data['SAMPLER'])
    if isinstance(sampler,int): sampler = samplers[sampler]

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

    if type(refscans) == list:
        refarray = np.zeros([len(refscans),speclen])
        LSTrefs  = np.zeros([len(refscans)])
        for II,refscan in enumerate(refscans):
            OKref = OK * (refscan == data['SCAN'])  

            specrefon  = np.median(dataarr[OKref*CalOn,:],axis=0) 
            specrefoff = np.median(dataarr[OKref*CalOff,:],axis=0)
            tcalref    = np.median(data['TCAL'][OKref])
            tsysref    = ( np.mean(specrefoff[speclen*0.1:speclen*0.9]) / 
                    (np.mean((specrefon-specrefoff)[speclen*0.1:speclen*0.9])) * 
                    tcalref + tcalref/2.0 )
            refarray[II] = (specrefon + specrefoff)/2.0
            LSTrefs[II]  = np.mean(data['LST'][OKref])
            if specrefon.sum() == 0 or specrefoff.sum() == 0:
                raise ValueError("All values in reference scan %i are zero" % refscan)
            elif np.isnan(specrefon).sum() > 0 or np.isnan(specrefoff).sum() > 0:
                raise ValueError("Reference scan %i contains a NAN" % refscan)

    elif refscan1 is not None and refscan2 is not None:
        OKref1 = OK * (refscan1 == data['SCAN'])  
        OKref2 = OK * (refscan2 == data['SCAN'])  
        
        specref1on  = np.median(dataarr[OKref1*CalOn,:],axis=0) 
        specref1off = np.median(dataarr[OKref1*CalOff,:],axis=0)
        tcalref1    = np.median(data['TCAL'][OKref1])
        tsysref1    = ( np.mean(specref1off[speclen*0.1:speclen*0.9]) / 
                (np.mean((specref1on-specref1off)[speclen*0.1:speclen*0.9])) * 
                tcalref1 + tcalref1/2.0 )
        specref1 = (specref1on + specref1off)/2.0
        LSTref1 = np.mean(data['LST'][OKref1])
        if specref1on.sum() == 0 or specref1off.sum() == 0:
            raise ValueError("All values in reference 1 are zero")
        elif np.isnan(specref1on).sum() > 0 or np.isnan(specref1off).sum() > 0:
            raise ValueError("Reference 1 contains a NAN")

        specref2on  = np.median(dataarr[OKref2*CalOn,:],axis=0) 
        specref2off = np.median(dataarr[OKref2*CalOff,:],axis=0)
        tcalref2    = np.median(data['TCAL'][OKref2])
        tsysref2    = ( np.mean(specref2off[speclen*0.1:speclen*0.9]) / 
                (np.mean((specref2on-specref2off)[speclen*0.1:speclen*0.9])) * 
                tcalref2 + tcalref2/2.0 )
        specref2 = (specref2on + specref2off)/2.0
        LSTref2 = np.mean(data['LST'][OKref2])
        LSTspread = LSTref2 - LSTref1
        if specref2on.sum() == 0 or specref2off.sum() == 0:
            raise ValueError("All values in reference 2 are zero")
        elif np.isnan(specref2on).sum() > 0 or np.isnan(specref2off).sum() > 0:
            raise ValueError("Reference 2 contains a NAN")

    print "Beginning calibration of %i scans." % ((OKsource*CalOn).sum())

    if ((OKsource*CalOn).sum()) == 0:
        import pdb; pdb.set_trace()

    # compute TSYS on a scan-by-scan basis to avoid problems with saturated TSYS.
    scannumbers = np.unique(data['SCAN'][OKsource])
    for scanid in scannumbers:
        whscan = data['SCAN'] == scanid

        on_data = dataarr[whscan*CalOn,speclen*0.1:speclen*0.9]
        off_data = dataarr[whscan*CalOff,speclen*0.1:speclen*0.9]
        tcal = np.median(data['TCAL'][whscan])

        offmean = np.median(off_data,axis=0).mean()
        onmean  = np.median(on_data,axis=0).mean()
        diffmean = onmean-offmean

        tsys = ( offmean / diffmean * tcal + tcal/2.0 )
        print "Scan %4i:  TSYS=%12.3f" % (scanid,tsys)
        data['TSYS'][whscan] = tsys
    
    for specindOn,specindOff in zip(np.where(OKsource*CalOn)[0],np.where(OKsource*CalOff)[0]):

        for K in namelist:
            if K != 'DATA':
                newdatadict[K].append(data[K][specindOn])
            else:
                newdatadict['DATA'].append(np.zeros(4096))

        specOn = dataarr[specindOn,:]
        specOff = dataarr[specindOff,:]
        spec = (specOn + specOff)/2.0
        LSTspec = data['LST'][specindOn]

        if refscans is not None:
            refscannumber = np.argmin(np.abs(LSTspec-LSTrefs))
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

        specRef = (specref2-specref1)/LSTspread*(LSTspec-LSTref1) + specref1 # LINEAR interpolation between refs

        # use a templated OFF spectrum
        # (e.g., one that has had spectral lines interpolated over)
        if off_template is not None and off_template.shape == specRef.shape:
            #import pdb; pdb.set_trace()
            specRef = off_template * specRef.mean() / off_template.mean()

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

def generate_continuum_map(filename, pixsize=24, **kwargs):
    data = pyfits.getdata(filename)
    fileheader = pyfits.getheader(filename)
    if hasattr(data,'CRVAL2') and hasattr(data,'CRVAL2'):
        minx,maxx = data.CRVAL2.min(),data.CRVAL2.max()
        miny,maxy = data.CRVAL3.min(),data.CRVAL3.max()
        if 'RA' in data.CTYPE2:
            coordsys_in='celestial'
            minx,miny = coords.Position([minx,miny],system='celestial').galactic()
            maxx,maxy = coords.Position([maxx,maxy],system='celestial').galactic()
        elif 'GLON' in data.CTYPE2:
            coordsys_in='galactic'
    elif hasattr(data,'GLON') and hasattr(data,'GLAT'):
        minx,maxx = data.GLON.min(),data.GLON.max()
        miny,maxy = data.GLAT.min(),data.GLAT.max()
        coordsys_in='galactic'

    centerx = (maxx + minx) / 2.0
    centery = (maxy + miny) / 2.0
    naxis1 = np.ceil( (maxx-minx)/(pixsize/3600.0) )
    naxis2 = np.ceil( (maxx-minx)/(pixsize/3600.0) )

    generate_header(centerx,centery,naxis1=naxis1, naxis2=naxis2, naxis3=1,
            pixsize=pixsize, 
            output_flatheader='continuum_flatheader.txt',
            output_cubeheader='continuum_cubeheader.txt',
            clobber=True)

    flathead = pyfits.Header.fromtextfile('continuum_flatheader.txt')
    wcs = pywcs.WCS(flathead)

    image = np.zeros([naxis2,naxis1])
    nhits = np.zeros([naxis2,naxis1])

    for datapoint,pos in zip(data_iterator(data,continuum=True),coord_iterator(data)):
        glon,glat = pos

        if glon != 0 and glat != 0:
            x,y = wcs.wcs_sky2pix(glon,glat,0)
            if 0 < int(np.round(x)) < naxis1 and 0 < int(np.round(y)) < naxis2:
                image[int(np.round(y)),int(np.round(x))]  += datapoint
                nhits[int(np.round(y)),int(np.round(x))]  += 1
            else:
                print "Skipped a data point at %f,%f in file %s because it's out of the grid" % (x,y,filename)

    imav = image/nhits

    outpre = os.path.splitext(filename)[0]

    HDU2 = pyfits.PrimaryHDU(data=imav,header=flathead)
    HDU2.writeto(outpre+"_continuum.fits",clobber=True)
    HDU2.data = nhits
    HDU2.writeto(outpre+"_nhits.fits",clobber=True)

def add_file_to_cube(filename, cubefilename, flatheader='header.txt',
        cubeheader='cubeheader.txt', nhits=None, wcstype='',
        smoothto=1, baselineorder=5,
        velocityrange=None, excludefitrange=None, noisecut=np.inf,
        do_runscript=False, linefreq=None, allow_smooth=True,
        data_iterator=data_iterator, coord_iterator=coord_iterator,
        velo_iterator=velo_iterator, debug=False,
        progressbar=False, coordsys='galactic',
        velocity_offset=0.0,
        negative_mean_cut=None,
        add_with_kernel=False,
        kernel_fwhm=None,
        fsw=False,
        diagnostic_plot_name=None,
        chmod=False,
        continuum_prefix=None):
    """ 
    Given a .fits file that contains a binary table of spectra (e.g., as
    you would get from the GBT mapping "pipeline" or the reduce_map.pro aoidl
    file provided by Adam Ginsburg), adds each spectrum into the cubefile.

    velocity_offset : 0.0
        Amount to add to the velocity vector before adding it to the cube
        (useful for FSW observations)
    """
    print "Loading file %s" % filename
    data = pyfits.getdata(filename)
    fileheader = pyfits.getheader(filename)

    if debug:
        print "Loaded ",filename,"...",

    if type(nhits) is str:
        if debug > 0:
            print "Loading nhits from %s" % nhits
        nhits = pyfits.getdata(nhits)
    elif type(nhits) is not np.ndarray:
        raise Exception( "nhits must be a .fits file or an ndarray, but it is ",type(nhits) )
    naxis2,naxis1 = nhits.shape

    contimage = np.zeros_like(nhits)
    nhits_once = np.zeros_like(nhits)

    # rescale image to weight by number of observations
    image = pyfits.getdata(cubefilename)*nhits
    if debug > 0:
        print "nhits statistics: mean, std, nzeros, size",nhits.mean(),nhits.std(),np.sum(nhits==0), nhits.size
        print "Image statistics: mean, std, nzeros, size",image.mean(),image.std(),np.sum(image==0), image.size, np.sum(np.isnan(image))
    # default is to set empty pixels to NAN; have to set them
    # back to zero
    image[image!=image] = 0.0
    header = pyfits.getheader(cubefilename)
    # debug print "Cube shape: ",image.shape," naxis3: ",header.get('NAXIS3')," nhits shape: ",nhits.shape

    if debug > 0:
        print "Image statistics: mean, std, nzeros, size",image.mean(),image.std(),np.sum(image==0), image.size

    # the input spectra are expected to have spectra on the 1st axis
    # cdelt = fileheader.get('CD1_1') if fileheader.get('CD1_1') else fileheader.get('CDELT1'+wcstype)
    # (not used - overwritten below)

    flathead = pyfits.Header.fromtextfile(flatheader)
    naxis3 = image.shape[0]
    wcs = pywcs.WCS(flathead)
    cd3 = header.get('CD3_3')
    cubevelo = (np.arange(naxis3)+1-header.get('CRPIX3'))*cd3 + header.get('CRVAL3')

    if add_with_kernel:
        cd = np.abs(wcs.wcs.cd[1,1])

    if velocityrange is not None:
        v1,v4 = velocityrange
        ind1 = np.argmin(np.abs(np.floor(v1-cubevelo)))
        ind2 = np.argmin(np.abs(np.ceil(v4-cubevelo)))+1
        # stupid hack.  REALLY stupid hack.  Don't crop.
        if np.abs(ind2-image.shape[0]) < 5: ind2 = image.shape[0]
        if np.abs(ind1) < 5: ind1 = 0
        #print "Velo match for v1,v4 = %f,%f: %f,%f" % (v1,v4,cubevelo[ind1],cubevelo[ind2])
        print "Updating CRPIX3 from %i to %i. Cropping to indices %i,%i" % (header.get('CRPIX3'),header.get('CRPIX3')-ind1,ind1,ind2)
        header.update('CRPIX3',header.get('CRPIX3')-ind1)
    else:
        ind1=0
        ind2 = image.shape[0]
        v1,v4 = min(cubevelo),max(cubevelo)

    # debug print "Cube has %i v-axis pixels from %f to %f.  Crop range is %f to %f" % (naxis3,cubevelo.min(),cubevelo.max(),v1,v4)

    #if abs(cdelt) < abs(cd3):
    #    print "Spectra have CD=%0.2f, cube has CD=%0.2f.  Will smooth & interpolate." % (cdelt,cd3)

    if progressbar and 'ProgressBar' in globals():
        pb = ProgressBar()
    else:
        pb = lambda x: x

    skipped = []

    for spectrum,pos,velo in pb(zip(data_iterator(data,fsw=fsw),
                                    coord_iterator(data,coordsys_out=coordsys),
                                    velo_iterator(data,linefreq=linefreq))):
        glon,glat = pos
        cdelt = velo[1]-velo[0]
        if cdelt < 0:
            # for interpolation, require increasing X axis
            spectrum = spectrum[::-1]
            velo     = velo[::-1]
            if debug > 2:
                print "Reversed spectral axis... ",

        velo += velocity_offset

        if glon != 0 and glat != 0:
            x,y = wcs.wcs_sky2pix(glon,glat,0)
            if debug > 2:
                print "At point ",x,y," ...",
            if abs(cdelt) < abs(cd3) and allow_smooth:
                # need to smooth before interpolating to preserve signal
                kernel = np.exp(-(np.linspace(-5,5,11)**2)/(2.0*abs(cd3/cdelt/2.35)**2))
                kernel /= kernel.sum()
                smspec = np.convolve(spectrum,kernel,mode='same')
                datavect = np.interp(cubevelo,velo,smspec)
            else:
                datavect = np.interp(cubevelo,velo,spectrum)
            OK = (datavect[ind1:ind2] == datavect[ind1:ind2])

            if excludefitrange is None:
                include = OK
            else:
                # Exclude certain regions (e.g., the spectral lines) when computing the noise
                include = OK.copy()

                # Convert velocities to indices
                exclude_inds = [np.argmin(np.abs(np.floor(v-cubevelo))) for v in excludefitrange]

                # Loop through exclude_inds pairwise
                for (i1,i2) in zip(exclude_inds[:-1:2],exclude_inds[1::2]):
                    # Do not include the excluded regions
                    include[i1:i2] = False

                if include.sum() == 0:
                    raise ValueError("All data excluded.")

            noiseestimate = datavect[ind1:ind2][include].std()
            contestimate = datavect[ind1:ind2][include].mean()

            if noiseestimate > noisecut:
                print "Skipped a data point at %f,%f in file %s because it had excessive noise %f" % (x,y,filename,noiseestimate)
                skipped.append(True)
                continue
            elif negative_mean_cut is not None and contestimate < negative_mean_cut:
                print "Skipped a data point at %f,%f in file %s because it had negative continuum %f" % (x,y,filename,contestimate)
                skipped.append(True)
                continue
            elif OK.sum() == 0:
                print "Skipped a data point at %f,%f in file %s because it had NANs" % (x,y,filename)
                skipped.append(True)
                continue
            elif OK.sum()/float(abs(ind2-ind1)) < 0.5:
                print "Skipped a data point at %f,%f in file %s because it had %i NANs" % (x,y,filename,np.isnan(datavect[ind1:ind2]).sum() )
                skipped.append(True)
                continue
            if debug > 2:
                print "did not skip...",
            if 0 < int(np.round(x)) < naxis1 and 0 < int(np.round(y)) < naxis2:
                if add_with_kernel:
                    kernel_size = kd = 5
                    kernel_middle = mid = (kd-1)/2.
                    xinds,yinds = (np.mgrid[:kd,:kd]-mid+np.array([np.round(x),np.round(y)])[:,None,None]).astype('int')
                    fwhm = np.sqrt(8*np.log(2))
                    kernel2d = np.exp(-((xinds-x)**2+(yinds-y))**2/(2*(kernel_fwhm/fwhm/cd)**2))

                    dim1 = datavect.shape[0]
                    vect_to_add = np.outer(datavect[ind1:ind2],kernel2d).reshape([dim1,kd,kd])
                    vect_to_add[True-OK] = 0

                    image[ind1:ind2,yinds,xinds] += vect_to_add
                    # NaN spectral bins are not appropriately downweighted... but they shouldn't exist anyway...
                    nhits[yinds,xinds] += kernel2d
                    contimage[yinds,xinds] += kernel2d * contestimate
                    nhits_once[yinds,xinds] += kernel2d

                else:
                    image[ind1:ind2,int(np.round(y)),int(np.round(x))][OK]  += datavect[ind1:ind2][OK]
                    nhits[int(np.round(y)),int(np.round(x))]     += 1
                    contimage[yinds,xinds] += contestimate
                    nhits_once[yinds,xinds] += 1

                if debug > 2:
                    print "Z-axis indices are ",ind1,ind2,"...",
                    print "Added a data point at ",int(np.round(x)),int(np.round(y)),"!"
                skipped.append(False)
            else:
                skipped.append(True)
                print "Skipped a data point at %f,%f in file %s because it's out of the grid" % (x,y,filename)
        #import pdb; pdb.set_trace()
        #raise Exception

    # this block redifining "include" is used for both diagnostics (optional)
    # and continuum below
    ind1a = np.argmin(np.abs(np.floor(v1-velo)))
    ind2a = np.argmin(np.abs(np.ceil(v4-velo)))+1
    dname = 'DATA' if 'DATA' in data.dtype.names else 'SPECTRA'
    OK = (data[dname][0,:]==data[dname][0,:])
    OK[:ind1a] = False
    OK[ind2a:] = False

    if excludefitrange is not None:
        include = OK

        # Convert velocities to indices
        exclude_inds = [np.argmin(np.abs(np.floor(v-velo))) for v in excludefitrange]

        # Loop through exclude_inds pairwise
        for (i1,i2) in zip(exclude_inds[:-1:2],exclude_inds[1::2]):
            # Do not include the excluded regions
            include[i1:i2] = False

        if include.sum() == 0:
            raise ValueError("All data excluded.")


    if diagnostic_plot_name:
        from mpl_plot_templates import imdiagnostics

        pylab.clf()

        dd = data[dname][:,include]
        imdiagnostics(dd,axis=pylab.gca())
        pylab.savefig(diagnostic_plot_name, bbox_inches='tight')

        # Save a copy with the bad stuff flagged out; this should tell whether flagging worked
        skipped = np.array(skipped,dtype='bool')
        dd[skipped,:] = -999
        maskdata = np.ma.masked_equal(dd,-999)
        pylab.clf()
        imdiagnostics(maskdata, axis=pylab.gca())
        dpn_pre,dpn_suf = os.path.splitext(diagnostic_plot_name)
        dpn_flagged = dpn_pre+"_flagged"+dpn_suf
        pylab.savefig(dpn_flagged, bbox_inches='tight')

        print "Saved diagnostic plot ",diagnostic_plot_name," and ",dpn_flagged

    if debug > 0:
        print "nhits statistics: mean, std, nzeros, size",nhits.mean(),nhits.std(),np.sum(nhits==0), nhits.size
        print "Image statistics: mean, std, nzeros, size",image.mean(),image.std(),np.sum(image==0), image.size
    
    imav = image/nhits

    if debug > 0:
        nnan = np.sum(np.isnan(imav))
        print "imav statistics: mean, std, nzeros, size, nnan, ngood:",imav.mean(),imav.std(),np.sum(imav==0), imav.size, nnan, imav.size-nnan
        print "imav shape: ",imav.shape

    subcube = imav[ind1:ind2,:,:]

    if debug > 0:
        nnan = np.sum(np.isnan(subcube))
        print "subcube statistics: mean, std, nzeros, size, nnan, ngood:",np.nansum(subcube)/subcube.size,np.std(subcube[subcube==subcube]),np.sum(subcube==0), subcube.size, nnan, subcube.size-nnan
        print "subcube shape: ",subcube.shape

    H = header.copy()
    for k,v in fileheader.iteritems():
        if 'RESTFRQ' in k or 'RESTFREQ' in k:
            header.update(k,v)
        #if k[0] == 'C' and '1' in k and k[-1] != '1':
        #    header.update(k.replace('1','3'), v)
    header.fromtextfile(cubeheader)
    for k,v in H.iteritems():
        header.update(k,v)
    HDU = pyfits.PrimaryHDU(data=subcube,header=header)
    HDU.writeto(cubefilename,clobber=True,output_verify='fix')

    outpre = cubefilename.replace(".fits","")

    #OKCube = (imav==imav)
    #contmap = np.nansum(imav[naxis3*0.1:naxis3*0.9,:,:],axis=0) / OKCube.sum(axis=0)
    contmap = np.nansum(imav[include,:,:]) / include.sum()
    HDU2 = pyfits.PrimaryHDU(data=contmap,header=flathead)
    HDU2.writeto(outpre+"_continuum.fits",clobber=True,output_verify='fix')
    HDU2.data = nhits
    HDU2.writeto(outpre+"_nhits.fits",clobber=True,output_verify='fix')

    if continuum_prefix is not None:
        # Solo continuum image (just this obs set)
        HDU2.data = contimage / nhits_once
        HDU2.writeto(continuum_prefix+"_continuum.fits",clobber=True,output_verify='fix')
        HDU2.data = nhits_once
        HDU2.writeto(continuum_prefix+"_nhits.fits",clobber=True,output_verify='fix')

    scriptfile = open(outpre+"_starlink.sh",'w')
    outpath,outfn = os.path.split(cubefilename)
    outpath,pre = os.path.split(outpre)
    print >>scriptfile,("#!/bin/bash")
    print >>scriptfile,('cd %s' % outpath)
    print >>scriptfile,('. /star/etc/profile')
    print >>scriptfile,('kappa > /dev/null')
    print >>scriptfile,('convert > /dev/null')
    print >>scriptfile,('fits2ndf %s %s' % (outfn,outfn.replace(".fits",".sdf")))
    if excludefitrange is not None:
        v2v3 = ""
        for v2,v3 in zip(excludefitrange[::2],excludefitrange[1::2]):
            v2v3 += "%0.2f %0.2f " % (v2,v3)
        print >>scriptfile,('mfittrend %s  ranges=\\\"%0.2f %s %0.2f\\\" order=%i axis=3 out=%s' % (outfn.replace(".fits",".sdf"),v1,v2v3,v4,baselineorder,outfn.replace(".fits","_baseline.sdf")))
    else:
        print >>scriptfile,('mfittrend %s  ranges=\\\"%0.2f %0.2f\\\" order=%i axis=3 out=%s' % (outfn.replace(".fits",".sdf"),v1,v4,baselineorder,outfn.replace(".fits","_baseline.sdf")))
    print >>scriptfile,('sub %s %s %s' % (outfn.replace(".fits",".sdf"),outfn.replace(".fits","_baseline.sdf"),outfn.replace(".fits","_sub.sdf")))
    print >>scriptfile,('sqorst %s_sub mode=pixelscale  axis=3 pixscale=%i out=%s_vrebin' % (pre,smoothto,pre))
    print >>scriptfile,('gausmooth %s_vrebin fwhm=1.0 axes=[1,2] out=%s_smooth' % (pre,pre))
    print >>scriptfile,('#collapse %s estimator=mean axis="RADI-LSR" low=-400 high=500 out=%s_continuum' % (pre,pre))
    print >>scriptfile,('rm %s_sub.fits' % (pre))
    print >>scriptfile,('ndf2fits %s_sub %s_sub.fits' % (pre,pre))
    print >>scriptfile,('rm %s_smooth.fits' % (pre))
    print >>scriptfile,('ndf2fits %s_smooth %s_smooth.fits' % (pre,pre))
    print >>scriptfile,("# Fix STARLINK's failure to respect header keywords.")
    print >>scriptfile,('sethead %s_smooth.fits RESTFRQ=`gethead RESTFRQ %s.fits`' % (pre,pre))
    scriptfile.close()

    if chmod:
        os.system("chmod +x "+outpre+"_starlink.sh")

    if do_runscript: runscript(outpre)

    _fix_ms_kms_file(outpre+"_sub.fits")
    _fix_ms_kms_file(outpre+"_smooth.fits")

def runscript(outpre):
    if outpre[0] != "/":
        os.system("./"+outpre+"_starlink.sh")
    else:
        os.system(outpre+"_starlink.sh")

def _fix_ms_kms_header(header):
    if header['CUNIT3'] == 'm/s':
        header['CUNIT3'] = 'km/s'
        header['CRVAL3'] /= 1e3
        if 'CD3_3' in header:
            header['CD3_3'] /= 1e3
        else:
            header['CDELT3'] /= 1e3
    return header

def _fix_ms_kms_file(filename):
    f = pyfits.open(filename)
    f[0].header = _fix_ms_kms_header(f[0].header)
    f.writeto(filename,clobber=True)

try:
    # requires agpy.  Might not work
    from agpy import cubes

    def make_flats(cubename,vrange=[0,10],noisevrange=[-100,-50],suffix='_sub.fits'):
        cubefile = pyfits.open(cubename+suffix)
        cubefile[0].header = _fix_ms_kms_header(cubefile[0].header)
        flathead = cubes.flatten_header(cubefile[0].header)
        integrated = cubes.integ(cubefile,vrange,zunits='wcs')[0]
        if integrated.shape != cubefile[0].data.shape[1:]:
            raise ValueError("Cube integrated to incorrect size.  Major error.  Badness.")
        flatimg = pyfits.PrimaryHDU(data=integrated,header=flathead)
        flatimg.writeto(cubename.replace("cube","integrated")+".fits",clobber=True)
        noise = cubes.integ(cubefile,noisevrange,average=np.std,zunits='wcs')[0]
        flatimg.data = noise
        flatimg.writeto(cubename.replace("cube","noise")+".fits",clobber=True)
        mincube = cubes.integ(cubefile,vrange,average=np.min,zunits='wcs')[0]
        flatimg.data = mincube
        flatimg.writeto(cubename.replace("cube","min")+".fits",clobber=True)

    def make_taucube(cubename,continuum=0.0,continuum_units='K',TCMB=2.7315, etamb=1.):
        cubefile = pyfits.open(cubename+"_sub.fits")
        cubefile[0].header = _fix_ms_kms_header(cubefile[0].header)
        if type(continuum) is str:
            continuum = pyfits.getdata(continuum)
        if cubefile[0].header.get('BUNIT') != continuum_units:
            raise ValueError("Unit mismatch.")
        tau = -np.log( (TCMB+cubefile[0].data/etamb+continuum/etamb) / (TCMB+continuum/etamb) )
        cubefile[0].data = tau
        cubefile[0].header['BUNIT']='tau'
        cubefile.writeto(cubename.replace("cube","taucube")+".fits",clobber=True)
        cubefile[0].data = tau.sum(axis=0)
        cubefile[0].header['BUNIT']='tau km/s'
        cubefile.writeto(cubename.replace("cube","taucube_integrated")+".fits",clobber=True)

except:
    pass
