try:
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
except ImportError:
    import pyfits
    import pywcs
#import coords
from astropy import coordinates
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
                pos = coordinates.Galactic(data.GLON[ii],data.GLAT[ii],unit=('deg','deg'))
                ra,dec = pos.icrs.ra.deg,pos.icrs.dec.deg
                #pos = coords.Position([data.GLON[ii],data.GLAT[ii]],system='galactic')
                #ra,dec = pos.j2000()
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
                pos = coordinates.ICRS(data.CRVAL2[ii],data.CRVAL3[ii],unit=('deg','deg'))
                glon,glat = pos.galactic.l.deg, pos.galactic.b.deg
                #pos = coords.Position([data.CRVAL2[ii],data.CRVAL3[ii]])
                #glon,glat = pos.galactic()
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
            #restfreq = data.RESTFREQ[ii]
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
        OKsource *= ((obsmode == data.OBSMODE) + ((obsmode+":NONE:TPWCAL") == data.OBSMODE))
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
                     smoothto=1, baselineorder=5, velocityrange=None,
                     excludefitrange=None, noisecut=np.inf, do_runscript=False,
                     linefreq=None, allow_smooth=True,
                     data_iterator=data_iterator,
                     coord_iterator=coord_iterator,
                     velo_iterator=velo_iterator, debug=False,
                     progressbar=False, coordsys='galactic',
                     velocity_offset=0.0, negative_mean_cut=None,
                     add_with_kernel=False, kernel_fwhm=None, fsw=False,
                     diagnostic_plot_name=None, chmod=False,
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

    if debug > 0:
        print "Loading data cube ",cubefilename
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
                    fwhm = np.sqrt(8*np.log(2))
                    kernel_size = kd = int(np.ceil(kernel_fwhm/fwhm/cd * 5))
                    if kernel_size < 5:
                        kernel_size = kd = 5
                    if kernel_size % 2 == 0:
                        kernel_size = kd = kernel_size+1
                    kernel_middle = mid = (kd-1)/2.
                    xinds,yinds = (np.mgrid[:kd,:kd]-mid+np.array([np.round(x),np.round(y)])[:,None,None]).astype('int')
                    kernel2d = np.exp(-((xinds-x)**2+(yinds-y)**2)/(2*(kernel_fwhm/fwhm/cd)**2))

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
                    contimage[int(np.round(y)),int(np.round(x))] += contestimate
                    nhits_once[int(np.round(y)),int(np.round(x))] += 1

                if debug > 2:
                    print "Z-axis indices are ",ind1,ind2,"...",
                    print "Added a data point at ",int(np.round(x)),int(np.round(y)),"!"
                skipped.append(False)
            else:
                skipped.append(True)
                print "Skipped a data point at %f,%f in file %s because it's out of the grid" % (x,y,filename)
        #import pdb; pdb.set_trace()
        #raise Exception

    if excludefitrange is not None:
        # this block redifining "include" is used for diagnostics (optional)
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

    #OKCube = (imav==imav)
    #contmap = np.nansum(imav[naxis3*0.1:naxis3*0.9,:,:],axis=0) / OKCube.sum(axis=0)
    contmap = np.nansum(imav[include,:,:],axis=0) / include.sum()
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
    if outpath != '':
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
    if os.path.exists(filename):
        f = pyfits.open(filename)
        f[0].header = _fix_ms_kms_header(f[0].header)
        f.writeto(filename,clobber=True)
    else:
        print "{0} does not exist".format(filename)

try:
    from pyspeckit import cubes
    from FITS_tools import strip_headers

    def make_flats(cubename,vrange=[0,10],noisevrange=[-100,-50],suffix='_sub.fits'):
        cubefile = pyfits.open(cubename+suffix)
        if not os.path.exists(cubename+suffix):
            raise IOError("Missing file %s.  This may be caused by a lack of starlink." % (cubename+suffix))
        cubefile[0].header = _fix_ms_kms_header(cubefile[0].header)
        flathead = strip_headers.flatten_header(cubefile[0].header)
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
        # TB is the "beam temperature" background-subtracted (from Rohlfs & Wilson)
        TB = (cubefile[0].data/etamb + continuum/etamb)*u.K
        tau = -np.log(1-(TB/T0)*( (np.exp(T0/tex)-1)**-1  - (np.exp(T0/TCMB)-1)**-1 )**-1)
    else:
        # Works in low-tau regime
        tau = -np.log( (TCMB+cubefile[0].data/etamb+continuum/etamb) / (TCMB+continuum/etamb) )
    cubefile[0].data = tau
    cubefile[0].header['BUNIT']='tau'
    cubefile.writeto(cubename.replace("cube","taucube")+outsuffix,clobber=True)
    cubefile[0].data = tau.sum(axis=0)
    cubefile[0].header['BUNIT']='tau km/s'
    cubefile.writeto(cubename.replace("cube","taucube_integrated")+outsuffix,clobber=True)
