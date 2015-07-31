try:
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
except ImportError:
    import pyfits
    import pywcs
#import coords
from astropy import coordinates, constants
from astropy import units as u
from astropy.convolution import convolve,Gaussian1DKernel,Gaussian2DKernel
import numpy as np
import pylab
try:
    import aplpy
except ImportError:
    pass
import os
import stat
import subprocess
try:
    from astropy.utils.console import ProgressBar
except ImportError:
    pass
from astropy import log
import time

# define speed of light for later use
ckms = constants.c.to(u.km/u.s).value

def generate_header(centerx, centery, naxis1=64, naxis2=64, naxis3=4096,
                    coordsys='galactic', ctype3='VRAD', bmaj=0.138888,
                    bmin=0.138888, pixsize=24, cunit3='km/s',
                    output_flatheader='header.txt',
                    output_cubeheader='cubeheader.txt', cd3=1.0, crval3=0.0,
                    crpix3=None, clobber=False, bunit="K", restfreq=None,
                    radio=True, author=None):
    """
    TODO: This should be automatically generated from appropriate data from
    GBT, APEX, etc.  Manually producing this leads to errors.
    """
    header = pyfits.Header()
    header.set('NAXIS1',naxis1)
    header.set('NAXIS2',naxis2)
    header.set('NAXIS3',naxis3)
    header.set('CDELT1',-1*pixsize/3600.0)
    header.set('CDELT2',pixsize/3600.0)
    header.set('EQUINOX',2000.0)
    header.set('SPECSYS','LSRK')
    if radio:
        header.set('VELREF',257) # CASA convention:
        # VELREF  =                  259 /1 LSR, 2 HEL, 3 OBS, +256 Radio
        # COMMENT casacore non-standard usage: 4 LSD, 5 GEO, 6 SOU, 7 GAL
    if restfreq:
        try:
            header.set('RESTFRQ',restfreq.to(u.Hz).value)
        except AttributeError:
            header.set('RESTFRQ',restfreq)
    if coordsys == 'galactic':
        header.set('CTYPE1','GLON-CAR')
        header.set('CTYPE2','GLAT-CAR')
        header.set('CRVAL1',centerx)
        header.set('CRVAL2',0)
        header.set('CRPIX1',(naxis1+1.)/2)
        header.set('CRPIX2',(naxis2+1.)/2-centery/header['CDELT2'])
    if coordsys in ('celestial','radec'):
        header.set('CTYPE1','RA---TAN')
        header.set('CTYPE2','DEC--TAN')
        header.set('CRPIX1',(naxis1+1.)/2)
        header.set('CRPIX2',(naxis2+1.)/2)
        header.set('CRVAL1',centerx)
        header.set('CRVAL2',centery)
    header.set('BMAJ',bmaj)
    header.set('BMIN',bmin)
    if crpix3 is None:
        header.set('CRPIX3',naxis3/2-1)
    else:
        header.set('CRPIX3',crpix3)
    header.set('CRVAL3',crval3)
    header.set('CDELT3',cd3)
    header.set('CTYPE3',ctype3)
    header.set('CUNIT3',cunit3)
    header.set('BUNIT',bunit)

    # Don't do this at install-time
    from .version import version
    header['SDPYVERS'] = (version,"sdpy code version")

    header.set('ORIGIN','sdpy')
    if author is not None:
        header.set('AUTHOR',author)
    header.totextfile(output_cubeheader,clobber=clobber)
    cubeheader = header.copy()
    del header['NAXIS3']
    del header['CRPIX3']
    del header['CRVAL3']
    del header['CDELT3' ]
    del header['CTYPE3']
    del header['CUNIT3']
    header.totextfile(output_flatheader,clobber=clobber)
    return cubeheader, header

def get_header(header):
    """
    Get header from a text file if its a string, or just return the header if
    it's already a header
    """
    if isinstance(header, str):
        return pyfits.Header.fromtextfile(header)
    elif isinstance(header, pyfits.Header):
        return header
    else:
        raise ValueError("Header is not of a valid type.")

def make_blank_images(cubeprefix, flatheader='header.txt',
                      cubeheader='cubeheader.txt', clobber=False,
                      dtype='float32'):

    flathead = get_header(flatheader)
    header = get_header(cubeheader)

    naxis1,naxis2,naxis3 = (int(np.ceil(header.get('NAXIS1'))),
                            int(np.ceil(header.get('NAXIS2'))),
                            int(np.ceil(header.get('NAXIS3'))))
    cubeshape = [naxis3,naxis2,naxis1]
    if np.product(cubeshape) > 2048**3:
        raise ValueError("Error: attempting to create cube with > 8 gigapixels")
    blankcube = np.zeros(cubeshape, dtype=dtype)
    blanknhits = np.zeros([naxis2,naxis1], dtype=dtype)
    log.info("".join(("Blank image size: {0},{1},{2}".format(naxis1,naxis2,naxis3),
                      ".  Blankcube shape: ",str(blankcube.shape))))
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
        raise Exception("Structure does not have DATA or SPECTRA tags. "
                        "Can't use it.  Write your own iterator.")

def coord_iterator(data,coordsys_out='galactic'):
    if hasattr(data,'GLON') and hasattr(data,'GLAT'):
        if coordsys_out == 'galactic':
            lon,lat = data.GLON,data.GLAT
        elif coordsys_out in ('celestial','radec'):
            pos = coordinates.SkyCoord(data.GLON,
                                       data.GLAT,
                                       unit=('deg','deg'),
                                       frame='galactic')
            lon,lat = pos.icrs.ra.deg,pos.icrs.dec.deg
    elif hasattr(data,'CRVAL2') and hasattr(data,'CRVAL3'):
        if 'RA' in data.CTYPE2:
            coordsys_in='celestial'
        elif 'GLON' in data.CTYPE2:
            coordsys_in='galactic'
        else:
            raise Exception("CRVAL exists, but RA/GLON not in CTYPE")
        if coordsys_out == 'galactic' and coordsys_in == 'celestial':
            #if hasattr(data, 'FEEDXOFF'):
            #    feedxoff = data.FEEDXOFF
            #elif hasattr(data, 'BEAMXOFF'):
            #    feedxoff = data.BEAMXOFF
            #else:
            #    feedxoff = 0
            pos = coordinates.SkyCoord(data.CRVAL2,
                                       data.CRVAL3,
                                       unit=('deg','deg'),
                                       frame='icrs')
            lon,lat = pos.galactic.l.deg, pos.galactic.b.deg
        elif (coordsys_out in ('celestial','radec') or
              coordsys_in==coordsys_out):
            lon,lat = data.CRVAL2,data.CRVAL3
    else:
        raise Exception("No CRVAL or GLON struct in data.")

    for ii in xrange(len(data)):
            yield lon[ii],lat[ii]

def velo_iterator(data,linefreq=None,useFreq=True, subvframe=True):
    for ii in xrange(data.CRPIX1.shape[0]):
        if hasattr(data,'SPECTRA'):
            npix = data.SPECTRA.shape[1]
            CRPIX = data.CRPIX1[ii]
            if useFreq:
                # 1e6 MHz to Hz
                CRVAL = data.CRVAL1F[ii] * u.MHz
                CDELT = data.CDELT1F[ii] * u.MHz
                vlsr_off = data.VLSR_OFF[ii] * u.km/u.s
                freq = (np.arange(npix)+1-CRPIX)*CDELT + CRVAL
                if linefreq is None:
                    linefreq = data.RESTFREQ[ii] * u.MHz
                else:
                    try:
                        linefreq = linefreq.to(u.Hz)
                    except AttributeError:
                        linefreq = linefreq * u.Hz
                # I still don't know if the sign of VLSR_OFF is right,
                # but it should be in km/s at least...
                # this is RADIO VELOCITY
                velo_u = (linefreq-freq)/linefreq * constants.c + vlsr_off
                velo = velo_u.to(u.km/u.s).value
            else:
                # this is the OPTICAL convention!!
                CRVAL = data.CRVAL1[ii]
                CDELT = data.CDELT1[ii]
                velo = (np.arange(npix)+1-CRPIX)*CDELT + CRVAL
        elif hasattr(data,'DATA'):
            npix = data.DATA.shape[1]
            #restfreq = data.RESTFREQ[ii]
            obsfreq = data.OBSFREQ[ii]
            deltaf = data.CDELT1[ii]
            sourcevel = data.VELOCITY[ii]
            if data.CTYPE1[ii] == 'FREQ-OBS' and subvframe:
                vframe = data.VFRAME[ii] / 1e3
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
                # RADIO VELOCITY
                velo = (linefreq-freqarr)/linefreq * ckms - vframe
                #obsfreq = data.OBSFREQ[ii]
                #cenfreq = obsfreq + (linefreq-restfreq)
                #crfreq = (CRPIX-1)*deltaf + cenfreq
                #CRVAL = (crfreq - cenfreq)/cenfreq * ckms
                #CDELT = -1*deltaf/cenfreq * ckms
            else:
                CRVAL = sourcevel/1000.0
                CDELT = -1*deltaf/(obsfreq) * ckms
                velo = (np.arange(npix)+1-CRPIX)*CDELT + CRVAL
        yield velo



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
        OKsource *= (scanrange[0] <= data['SCAN'])*(data['SCAN'] <= scanrange[1])
    if obsmode is not None:
        OKsource *= ((obsmode == data.OBSMODE) +
                     ((obsmode+":NONE:TPWCAL") == data.OBSMODE))
    if sourcename is None and scanrange is None:
        raise IndexError("Must specify a source name and/or a scan range")

    return OK,OKsource

def generate_continuum_map(filename, pixsize=24, **kwargs):
    data = pyfits.getdata(filename)
    #fileheader = pyfits.getheader(filename)
    if hasattr(data,'CRVAL2') and hasattr(data,'CRVAL2'):
        minx,maxx = data.CRVAL2.min(),data.CRVAL2.max()
        miny,maxy = data.CRVAL3.min(),data.CRVAL3.max()
        if 'RA' in data.CTYPE2:
            #coordsys_in='celestial'
            cmin = coordinates.SkyCoord(minx, miny, unit='deg,deg', frame='icrs')
            cmax = coordinates.SkyCoord(maxy, maxy, unit='deg,deg', frame='icrs')
            minx,miny = cmin.galactic.l.degree, cmin.galactic.b.degree
            maxx,maxy = cmax.galactic.l.degree, cmax.galactic.b.degree
            #minx,miny = coords.Position([minx,miny],system='celestial').galactic()
            #maxx,maxy = coords.Position([maxx,maxy],system='celestial').galactic()
        elif 'GLON' in data.CTYPE2:
            # this block exists because I thought I might have had to do
            # something extra...
            pass
            #coordsys_in='galactic'
    elif hasattr(data,'GLON') and hasattr(data,'GLAT'):
        minx,maxx = data.GLON.min(),data.GLON.max()
        miny,maxy = data.GLAT.min(),data.GLAT.max()
        #coordsys_in='galactic'

    centerx = (maxx + minx) / 2.0
    centery = (maxy + miny) / 2.0
    naxis1 = np.ceil((maxx-minx) / (pixsize/3600.0))
    naxis2 = np.ceil((maxx-minx) / (pixsize/3600.0))

    generate_header(centerx, centery, naxis1=naxis1, naxis2=naxis2, naxis3=1,
                    pixsize=pixsize,
                    output_flatheader='continuum_flatheader.txt',
                    output_cubeheader='continuum_cubeheader.txt', clobber=True)

    flathead = pyfits.Header.fromtextfile('continuum_flatheader.txt')
    wcs = pywcs.WCS(flathead)

    image = np.zeros([naxis2,naxis1])
    nhits = np.zeros([naxis2,naxis1])

    for datapoint,pos in zip(data_iterator(data,continuum=True),coord_iterator(data)):
        glon,glat = pos

        if glon != 0 and glat != 0:
            x,y = wcs.wcs_world2pix(glon,glat,0)
            if 0 < int(np.round(x)) < naxis1 and 0 < int(np.round(y)) < naxis2:
                image[int(np.round(y)),int(np.round(x))] += datapoint
                nhits[int(np.round(y)),int(np.round(x))] += 1
            else:
                log.info("Skipped a data point at %f,%f in file %s because it's out of the grid" % (x,y,filename))

    imav = image/nhits

    outpre = os.path.splitext(filename)[0]

    HDU2 = pyfits.PrimaryHDU(data=imav,header=flathead)
    HDU2.writeto(outpre+"_continuum.fits",clobber=True)
    HDU2.data = nhits
    HDU2.writeto(outpre+"_nhits.fits",clobber=True)


def add_file_to_cube(filename, cubefilename, **kwargs):
    log.info("Loading file %s" % filename)
    data = pyfits.getdata(filename)
    fileheader = pyfits.getheader(filename)

    log.debug("Loaded "+filename+"...")

    return add_data_to_cube(cubefilename, filename=filename, data=data,
                            fileheader=fileheader, **kwargs)

def add_data_to_cube(cubefilename, data=None, filename=None, fileheader=None,
                     flatheader='header.txt',
                     cubeheader='cubeheader.txt', nhits=None,
                     smoothto=1, baselineorder=5, velocityrange=None,
                     excludefitrange=None, noisecut=np.inf, do_runscript=False,
                     linefreq=None, allow_smooth=True,
                     data_iterator=data_iterator,
                     coord_iterator=coord_iterator,
                     velo_iterator=velo_iterator,
                     progressbar=False, coordsys='galactic',
                     datalength=None,
                     velocity_offset=0.0, negative_mean_cut=None,
                     add_with_kernel=False, kernel_fwhm=None, fsw=False,
                     kernel_function=Gaussian2DKernel,
                     diagnostic_plot_name=None, chmod=False,
                     continuum_prefix=None,
                     debug_breakpoint=False,
                     default_unit=u.km/u.s,
                     make_continuum=True,
                     weightspec=None,
                     varweight=False):
    """
    Given a .fits file that contains a binary table of spectra (e.g., as
    you would get from the GBT mapping "pipeline" or the reduce_map.pro aoidl
    file provided by Adam Ginsburg), adds each spectrum into the cubefile.

    velocity_offset : 0.0
        Amount to add to the velocity vector before adding it to the cube
        (useful for FSW observations)
    weightspec : np.ndarray
        A spectrum with the same size as the input arrays but containing the relative
        weights of the data
    """

    #if not default_unit.is_equivalent(u.km/u.s):
    #    raise TypeError("Default unit is not a velocity equivalent.")

    if type(nhits) is str:
        log.debug("Loading nhits from %s" % nhits)
        nhits = pyfits.getdata(nhits)
    elif type(nhits) is not np.ndarray:
        raise TypeError("nhits must be a .fits file or an ndarray, but it is ",type(nhits))
    naxis2,naxis1 = nhits.shape

    if velocity_offset and not fsw:
        raise ValueError("Using a velocity offset, but obs type is not "
                         "frequency switched; this is almost certainly wrong, "
                         "but if there's a case for it I'll remove this.")
    if not hasattr(velocity_offset,'unit'):
        velocity_offset = velocity_offset*default_unit


    contimage = np.zeros_like(nhits)
    nhits_once = np.zeros_like(nhits)

    log.debug("Loading data cube {0}".format(cubefilename))
    t0 = time.time()
    # rescale image to weight by number of observations
    image = pyfits.getdata(cubefilename)*nhits
    log.debug(" ".join(("nhits statistics: mean, std, nzeros, size",str(nhits.mean()),str(nhits.std()),str(np.sum(nhits==0)), str(nhits.size))))
    log.debug(" ".join(("Image statistics: mean, std, nzeros, size",str(image.mean()),str(image.std()),str(np.sum(image==0)), str(image.size), str(np.sum(np.isnan(image))))))
    log.debug(" ".join(("nhits shape: ",str(nhits.shape))))
    # default is to set empty pixels to NAN; have to set them
    # back to zero
    image[image!=image] = 0.0
    header = pyfits.getheader(cubefilename)
    # debug print "Cube shape: ",image.shape," naxis3: ",header.get('NAXIS3')," nhits shape: ",nhits.shape

    log.debug("".join(("Image statistics: mean, std, nzeros, size",str(image.mean()),str(image.std()),str(np.sum(image==0)), str(image.size))))

    flathead = get_header(flatheader)
    naxis3 = image.shape[0]
    wcs = pywcs.WCS(flathead)
    cwcs = pywcs.WCS(header)
    vwcs = cwcs.sub([pywcs.WCSSUB_SPECTRAL])
    vunit = u.Unit(vwcs.wcs.cunit[vwcs.wcs.spec])
    cubevelo = vwcs.wcs_pix2world(np.arange(naxis3),0)[0] * vunit
    cd3 = vwcs.wcs.cdelt[vwcs.wcs.spec] * vunit

    if not vunit.is_equivalent(default_unit):
        raise ValueError("The units of the cube and the velocity axis are "
                         "possibly not equivalent.  Change default_unit to "
                         "the appropriate unit (probably {0})".format(vunit))

    if add_with_kernel:
        if wcs.wcs.has_cd():
            cd = np.abs(wcs.wcs.cd[1,1])
        else:
            cd = np.abs(wcs.wcs.cdelt[1])
        # Alternative implementation; may not work for .cd?
        #cd = np.abs(np.prod((wcs.wcs.get_cdelt() * wcs.wcs.get_pc().diagonal())))**0.5

    if velocityrange is not None:
        v1,v4 = velocityrange * default_unit
        ind1 = np.argmin(np.abs(np.floor(v1-cubevelo)))
        ind2 = np.argmin(np.abs(np.ceil(v4-cubevelo)))+1

        # stupid hack.  REALLY stupid hack.  Don't crop.
        if np.abs(ind2-image.shape[0]) < 5:
            ind2 = image.shape[0]
        if np.abs(ind1) < 5:
            ind1 = 0

        #print "Velo match for v1,v4 = %f,%f: %f,%f" % (v1,v4,cubevelo[ind1],cubevelo[ind2])
        # print "Updating CRPIX3 from %i to %i. Cropping to indices %i,%i" % (header.get('CRPIX3'),header.get('CRPIX3')-ind1,ind1,ind2)
        # I think this could be disastrous: cubevelo is already set, but now we're changing how it's set in the header!
        # I don't think there's any reason to have this in the first place
        # header.set('CRPIX3',header.get('CRPIX3')-ind1)

        # reset v1,v4 to the points we just selected
        v1 = cubevelo[ind1]
        v4 = cubevelo[ind2-1]
    else:
        ind1=0
        ind2 = image.shape[0]
        v1,v4 = min(cubevelo),max(cubevelo)

    # debug print "Cube has %i v-axis pixels from %f to %f.  Crop range is %f to %f" % (naxis3,cubevelo.min(),cubevelo.max(),v1,v4)

    #if abs(cdelt) < abs(cd3):
    #    print "Spectra have CD=%0.2f, cube has CD=%0.2f.  Will smooth & interpolate." % (cdelt,cd3)

    # Disable progressbar if debug-logging is enabled (they clash)
    if progressbar and 'ProgressBar' in globals() and log.level > 10:
        if datalength is None:
            pb = ProgressBar(len(data))
        else:
            pb = ProgressBar(datalength)
    else:
        progressbar = False

    skipped = []

    for spectrum,pos,velo in zip(data_iterator(data,fsw=fsw),
                                 coord_iterator(data,coordsys_out=coordsys),
                                 velo_iterator(data,linefreq=linefreq)):

        if log.level <= 10:
            t1 = time.time()

        if not hasattr(velo,'unit'):
            velo = velo * default_unit

        glon,glat = pos
        cdelt = velo[1]-velo[0]
        if cdelt < 0:
            # for interpolation, require increasing X axis
            spectrum = spectrum[::-1]
            velo = velo[::-1]
            if log.level < 5:
                log.debug("Reversed spectral axis... ")

        if (velo.max() < cubevelo.min() or velo.min() > cubevelo.max()):
            raise ValueError("Data out of range.")

        if progressbar and log.level > 10:
            pb.update()

        velo += velocity_offset

        if glon != 0 and glat != 0:
            x,y = wcs.wcs_world2pix(glon,glat,0)
            if np.isnan(x) or np.isnan(y):
                log.warn("".join(("Skipping NaN point {0}, {1} ...".format(glon,glat))))
                continue
            if log.level < 10:
                log.debug("".join(("At point {0},{1} ...".format(glon,glat),)))
            if abs(cdelt) < abs(cd3) and allow_smooth:
                # need to smooth before interpolating to preserve signal
                kernwidth = abs(cd3/cdelt/2.35).decompose().value
                if kernwidth > 2 and kernwidth < 10:
                    xr = kernwidth*5
                    npx = np.ceil(xr*2 + 1)
                elif kernwidth > 10:
                    raise ValueError('Too much smoothing')
                else:
                    xr = 5
                    npx = 11
                #kernel = np.exp(-(np.linspace(-xr,xr,npx)**2)/(2.0*kernwidth**2))
                #kernel /= kernel.sum()
                kernel = Gaussian1DKernel(stddev=kernwidth, x_size=npx)
                smspec = np.convolve(spectrum,kernel,mode='same')
                datavect = np.interp(cubevelo.to(default_unit).value,
                                     velo.to(default_unit).value,
                                     smspec)
            else:
                datavect = np.interp(cubevelo.to(default_unit).value,
                                     velo.to(default_unit).value,
                                     spectrum)
            OK = (datavect[ind1:ind2] == datavect[ind1:ind2])

            if excludefitrange is None:
                include = OK
            else:
                # Exclude certain regions (e.g., the spectral lines) when computing the noise
                include = OK.copy()

                if not hasattr(excludefitrange,'unit'):
                    excludefitrange = excludefitrange * default_unit

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
                log.info("Skipped a data point at %f,%f in file %s because it had excessive noise %f" % (x,y,filename,noiseestimate))
                skipped.append(True)
                continue
            elif negative_mean_cut is not None and contestimate < negative_mean_cut:
                log.info("Skipped a data point at %f,%f in file %s because it had negative continuum %f" % (x,y,filename,contestimate))
                skipped.append(True)
                continue
            elif OK.sum() == 0:
                log.info("Skipped a data point at %f,%f in file %s because it had NANs" % (x,y,filename))
                skipped.append(True)
                continue
            elif OK.sum()/float(abs(ind2-ind1)) < 0.5:
                log.info("Skipped a data point at %f,%f in file %s because it had %i NANs" % (x,y,filename,np.isnan(datavect[ind1:ind2]).sum()))
                skipped.append(True)
                continue
            if log.level < 10:
                log.debug("did not skip...")

            if varweight:
                weight = 1./noiseestimate**2
            else:
                weight = 1.

            if weightspec is None:
                wspec = weight
            else:
                wspec = weight * weightspec


            if 0 < int(np.round(x)) < naxis1 and 0 < int(np.round(y)) < naxis2:
                if add_with_kernel:
                    fwhm = np.sqrt(8*np.log(2))
                    kernel_size = kd = int(np.ceil(kernel_fwhm/fwhm/cd * 5))
                    if kernel_size < 5:
                        kernel_size = kd = 5
                    if kernel_size % 2 == 0:
                        kernel_size = kd = kernel_size+1
                    if kernel_size > 100:
                        raise ValueError("Huge kernel - are you sure?")
                    kernel_middle = mid = (kd-1)/2.
                    xinds,yinds = (np.mgrid[:kd,:kd]-mid+np.array([np.round(x),np.round(y)])[:,None,None]).astype('int')
                    # This kernel is NOT centered, and that's the bloody point.
                    # (I made a very stupid error and used Gaussian2DKernel,
                    # which is strictly centered, in a previous version)
                    kernel2d = np.exp(-((xinds-x)**2+(yinds-y)**2)/(2*(kernel_fwhm/fwhm/cd)**2))

                    dim1 = ind2-ind1
                    vect_to_add = np.outer(datavect[ind1:ind2],kernel2d).reshape([dim1,kd,kd])
                    vect_to_add[True-OK] = 0

                    # need to slice out edges
                    if yinds.max() >= naxis2 or yinds.min() < 0:
                        yok = (yinds[0,:] < naxis2) & (yinds[0,:] >= 0)
                        xinds,yinds = xinds[:,yok],yinds[:,yok]
                        vect_to_add = vect_to_add[:,:,yok]
                        kernel2d = kernel2d[:,yok]
                    if xinds.max() >= naxis1 or xinds.min() < 0:
                        xok = (xinds[:,0] < naxis1) & (xinds[:,0] >= 0)
                        xinds,yinds = xinds[xok,:],yinds[xok,:]
                        vect_to_add = vect_to_add[:,xok,:]
                        kernel2d = kernel2d[xok,:]

                    image[ind1:ind2,yinds,xinds] += vect_to_add*wspec
                    # NaN spectral bins are not appropriately downweighted... but they shouldn't exist anyway...
                    nhits[yinds,xinds] += kernel2d*weight
                    contimage[yinds,xinds] += kernel2d * contestimate*weight
                    nhits_once[yinds,xinds] += kernel2d*weight

                else:
                    image[ind1:ind2,int(np.round(y)),int(np.round(x))][OK] += datavect[ind1:ind2][OK]*weight
                    nhits[int(np.round(y)),int(np.round(x))] += weight
                    contimage[int(np.round(y)),int(np.round(x))] += contestimate*weight
                    nhits_once[int(np.round(y)),int(np.round(x))] += weight

                if log.level < 10:
                    log.debug("Z-axis indices are %i,%i..." % (ind1,ind2,))
                    log.debug("Added a data point at %i,%i" % (int(np.round(x)),int(np.round(y))))
                skipped.append(False)
            else:
                skipped.append(True)
                log.info("Skipped a data point at x,y=%f,%f "
                         "lon,lat=%f,%f in file %s because "
                         "it's out of the grid" % (x,y,glon,glat,filename))

            if debug_breakpoint:
                import ipdb
                ipdb.set_trace()

        if log.level <= 10:
            dt = time.time() - t1
            log.debug("Completed x,y={x:4.0f},{y:4.0f}"
                      " ({x:6.2f},{y:6.2f}) in {dt:6.2g}s".format(x=float(x),
                                                                  y=float(y),
                                                                  dt=dt))

    log.info("Completed 'add_data' loop for"
             " {0} in {1}s".format(cubefilename, time.time()-t0))

    if excludefitrange is not None:
        # this block redefining "include" is used for diagnostics (optional)
        ind1a = np.argmin(np.abs(np.floor(v1-velo)))
        ind2a = np.argmin(np.abs(np.ceil(v4-velo)))+1
        dname = 'DATA' if 'DATA' in data.dtype.names else 'SPECTRA'
        OK = (data[dname][0,:]==data[dname][0,:])
        OK[:ind1a] = False
        OK[ind2a:] = False

        include = OK

        # Convert velocities to indices
        exclude_inds = [np.argmin(np.abs(np.floor(v-velo))) for v in excludefitrange]

        # Loop through exclude_inds pairwise
        for (i1,i2) in zip(exclude_inds[:-1:2],exclude_inds[1::2]):
            # Do not include the excluded regions
            include[i1:i2] = False

        if include.sum() == 0:
            raise ValueError("All data excluded.")
    else:
        dname = 'DATA' if 'DATA' in data.dtype.names else 'SPECTRA'
        include = slice(None)


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

        log.info("Saved diagnostic plot %s and %s" % (diagnostic_plot_name,dpn_flagged))

    log.debug("nhits statistics: mean, std, nzeros, size {0} {1} {2} {3}".format(nhits.mean(),nhits.std(),np.sum(nhits==0), nhits.size))
    log.debug("Image statistics: mean, std, nzeros, size {0} {1} {2} {3}".format(image.mean(),image.std(),np.sum(image==0), image.size))
    
    imav = image/nhits

    if log.level <= 10:
        nnan = np.count_nonzero(np.isnan(imav))
        log.debug("imav statistics: mean, std, nzeros, size, nnan, ngood: {0} {1} {2} {3} {4} {5}".format(imav.mean(),imav.std(),np.sum(imav==0), imav.size, nnan, imav.size-nnan))
        log.debug("imav shape: {0}".format(imav.shape))

    subcube = imav[ind1:ind2,:,:]

    if log.level <= 10:
        nnan = np.sum(np.isnan(subcube))
        print "subcube statistics: mean, std, nzeros, size, nnan, ngood:",np.nansum(subcube)/subcube.size,np.std(subcube[subcube==subcube]),np.sum(subcube==0), subcube.size, nnan, subcube.size-nnan
        print "subcube shape: ",subcube.shape

    H = header.copy()
    if fileheader is not None:
        for k,v in fileheader.iteritems():
            if 'RESTFRQ' in k or 'RESTFREQ' in k:
                header.set(k,v)
            #if k[0] == 'C' and '1' in k and k[-1] != '1':
            #    header.set(k.replace('1','3'), v)
    moreH = get_header(cubeheader)
    for k,v in H.iteritems():
        header.set(k,v)
    for k,v in moreH.iteritems():
        header.set(k,v)
    HDU = pyfits.PrimaryHDU(data=subcube,header=header)
    HDU.writeto(cubefilename,clobber=True,output_verify='fix')

    outpre = cubefilename.replace(".fits","")

    include = np.ones(imav.shape[0],dtype='bool')

    if excludefitrange is not None:
        # this block redifining "include" is used for continuum
        ind1a = np.argmin(np.abs(np.floor(v1-cubevelo)))
        ind2a = np.argmin(np.abs(np.ceil(v4-cubevelo)))+1

        # Convert velocities to indices
        exclude_inds = [np.argmin(np.abs(np.floor(v-cubevelo))) for v in excludefitrange]

        # Loop through exclude_inds pairwise
        for (i1,i2) in zip(exclude_inds[:-1:2],exclude_inds[1::2]):
            # Do not include the excluded regions
            include[i1:i2] = False

        if include.sum() == 0:
            raise ValueError("All data excluded.")

    HDU2 = pyfits.PrimaryHDU(data=nhits,header=flathead)
    HDU2.writeto(outpre+"_nhits.fits",clobber=True,output_verify='fix')

    #OKCube = (imav==imav)
    #contmap = np.nansum(imav[naxis3*0.1:naxis3*0.9,:,:],axis=0) / OKCube.sum(axis=0)
    if make_continuum:
        contmap = np.nansum(imav[include,:,:],axis=0) / include.sum()
        HDU2 = pyfits.PrimaryHDU(data=contmap,header=flathead)
        HDU2.writeto(outpre+"_continuum.fits",clobber=True,output_verify='fix')

        if continuum_prefix is not None:
            # Solo continuum image (just this obs set)
            HDU2.data = contimage / nhits_once
            HDU2.writeto(continuum_prefix+"_continuum.fits",clobber=True,output_verify='fix')
            HDU2.data = nhits_once
            HDU2.writeto(continuum_prefix+"_nhits.fits",clobber=True,output_verify='fix')

    log.info("Writing script file {0}".format(outpre+"_starlink.sh"))
    scriptfile = open(outpre+"_starlink.sh",'w')
    outpath,outfn = os.path.split(cubefilename)
    outpath,pre = os.path.split(outpre)
    print >>scriptfile,("#!/bin/bash")
    if outpath != '':
        print >>scriptfile,('cd %s' % outpath)
    print >>scriptfile,('. /star/etc/profile')
    print >>scriptfile,('kappa > /dev/null')
    print >>scriptfile,('convert > /dev/null')
    print >>scriptfile,('fits2ndf %s %s' % (outfn,outfn.replace(".fits",".sdf")))
    if excludefitrange is not None:
        v2v3 = ""
        for v2,v3 in zip(excludefitrange[::2],excludefitrange[1::2]):
            v2v3 += "%0.2f %0.2f " % (v2.to(default_unit).value,v3.to(default_unit).value)
        print >>scriptfile,('mfittrend %s  ranges=\\\"%0.2f %s %0.2f\\\" order=%i axis=3 out=%s' % (outfn.replace(".fits",".sdf"),v1.to(default_unit).value,v2v3,v4.to(default_unit).value,baselineorder,outfn.replace(".fits","_baseline.sdf")))
    else:
        print >>scriptfile,('mfittrend %s  ranges=\\\"%0.2f %0.2f\\\" order=%i axis=3 out=%s' % (outfn.replace(".fits",".sdf"),v1.to(default_unit).value,v4.to(default_unit).value,baselineorder,outfn.replace(".fits","_baseline.sdf")))
    print >>scriptfile,('sub %s %s %s' % (outfn.replace(".fits",".sdf"),outfn.replace(".fits","_baseline.sdf"),outfn.replace(".fits","_sub.sdf")))
    print >>scriptfile,('sqorst %s_sub mode=pixelscale  axis=3 pixscale=%i out=%s_vrebin' % (pre,smoothto,pre))
    print >>scriptfile,('gausmooth %s_vrebin fwhm=1.0 axes=[1,2] out=%s_smooth' % (pre,pre))
    print >>scriptfile,('#collapse %s estimator=mean axis="VRAD" low=-400 high=500 out=%s_continuum' % (pre,pre))
    print >>scriptfile,('rm %s_sub.fits' % (pre))
    print >>scriptfile,('ndf2fits %s_sub %s_sub.fits' % (pre,pre))
    print >>scriptfile,('rm %s_smooth.fits' % (pre))
    print >>scriptfile,('ndf2fits %s_smooth %s_smooth.fits' % (pre,pre))
    print >>scriptfile,("# Fix STARLINK's failure to respect header keywords.")
    print >>scriptfile,('sethead %s_smooth.fits RESTFRQ=`gethead RESTFRQ %s.fits`' % (pre,pre))
    print >>scriptfile,('rm %s_baseline.sdf' % (pre))
    print >>scriptfile,('rm %s_smooth.sdf' % (pre))
    print >>scriptfile,('rm %s_sub.sdf' % (pre))
    print >>scriptfile,('rm %s_vrebin.sdf' % (pre))
    print >>scriptfile,('rm %s.sdf' % (pre))
    scriptfile.close()

    if chmod:
        scriptfilename = (outpre+"_starlink.sh").replace(" ","")
        #subprocess.call("chmod +x {0}".format(scriptfilename), shell=True)
        st = os.stat(scriptfilename)
        os.chmod(scriptfilename, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH | stat.S_IXUSR)

    if do_runscript:
        runscript(outpre)

    _fix_ms_kms_file(outpre+"_sub.fits")
    _fix_ms_kms_file(outpre+"_smooth.fits")

    if log.level <= 20:
        log.info("Completed {0} in {1}s".format(pre, time.time()-t0))

def runscript(outpre):
    scriptfilename = (outpre+"_starlink.sh").replace(" ","")
    if scriptfilename[0] != "/":
        return subprocess.call("./"+scriptfilename, shell=True)
    else:
        return subprocess.call(scriptfilename, shell=True)

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
    if os.path.exists(filename):
        f = pyfits.open(filename)
        f[0].header = _fix_ms_kms_header(f[0].header)
        f.writeto(filename,clobber=True)
    else:
        print "{0} does not exist".format(filename)

try:
    from pyspeckit import cubes
    from FITS_tools import strip_headers

    def make_flats(cubename,vrange=[0,10],noisevrange=[-100,-50],suffix='_sub.fits',
                   out_suffix=""):
        cubefile = pyfits.open(cubename+suffix)
        if not os.path.exists(cubename+suffix):
            raise IOError("Missing file %s.  This may be caused by a lack of starlink." % (cubename+suffix))
        cubefile[0].header = _fix_ms_kms_header(cubefile[0].header)
        flathead = strip_headers.flatten_header(cubefile[0].header)
        integrated = cubes.integ(cubefile,vrange,zunits='wcs',dvmult=True)[0]
        if integrated.shape != cubefile[0].data.shape[1:]:
            raise ValueError("Cube integrated to incorrect size.  Major error.  Badness.")
        flatimg = pyfits.PrimaryHDU(data=integrated,header=flathead)
        flatimg.writeto(cubename.replace("cube","integrated")+out_suffix+".fits",
                        clobber=True)
        noise = cubes.integ(cubefile,noisevrange,average=np.std,zunits='wcs')[0]
        flatimg.data = noise
        flatimg.writeto(cubename.replace("cube","noise")+out_suffix+".fits",
                        clobber=True)
        mincube = cubes.integ(cubefile,vrange,average=np.min,zunits='wcs')[0]
        flatimg.data = mincube
        flatimg.writeto(cubename.replace("cube","min")+out_suffix+".fits",
                        clobber=True)


except:
    def make_flats(*args, **kwargs):
        print "Make flats did not import"

def make_taucube(cubename,continuum=0.0,continuum_units='K',TCMB=2.7315,
                 etamb=1., suffix="_sub.fits", outsuffix='.fits',
                 linefreq=None, tex=None):
    cubefile = pyfits.open(cubename+suffix)
    cubefile[0].header = _fix_ms_kms_header(cubefile[0].header)
    if type(continuum) is str:
        continuum = pyfits.getdata(continuum)
    if cubefile[0].header.get('BUNIT') != continuum_units:
        raise ValueError("Unit mismatch.")
    if linefreq is not None and tex is not None:
        from astropy import units as u
        from astropy import constants
        if not hasattr(linefreq,'unit'):
            linefreq = linefreq * u.GHz
        if not hasattr(tex,'unit'):
            tex = tex * u.K
        T0 = (constants.h * linefreq / constants.k_B).to(u.K)
        # TB is the "beam temperature"
        TB = (cubefile[0].data/etamb + continuum/etamb + TCMB)*u.K
        TBG = (continuum/etamb + TCMB)*u.K

        tau = (-np.log((TB-tex) / (TBG-tex))).value
    else:
        # Works in low-tex regime
        TBG = continuum/etamb + TCMB
        TB = (cubefile[0].data/etamb + TBG)
        tau = -np.log(TB / TBG)

    cubefile[0].data = tau
    cubefile[0].header['BUNIT']='tau'
    cubefile.writeto(cubename.replace("cube","taucube")+outsuffix,clobber=True)
    cubefile[0].data = tau.sum(axis=0)
    cubefile[0].header['BUNIT']='tau km/s'
    cubefile.writeto(cubename.replace("cube","taucube_integrated")+outsuffix,clobber=True)
