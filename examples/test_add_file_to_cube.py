from sdpy import makecube
import numpy as np

filename = '/Users/adam/observations/gbt/W51map/Session17_111to186_D37_F2.fits'

cubename_lores = 'test'
velocityrange=[-50,150]
cd3 = 1.0
#cd3 = 1.0 # Arecibo is limited to 0.64 because one of the receivers went bad at hi-res mode once
#naxis3 = int(np.ceil((velocityrange[1]-velocityrange[0])/cd3))+4 # +4 is BAD!  don't do that.
naxis3 = int((velocityrange[1]-velocityrange[0]) / cd3) + 1 # +1 is good: include -50
crval3 = 50.0
# dumb debug stuff
vels = crval3+cd3*(np.arange(naxis3)+1-naxis3/2-1)
# this will probably cause an error but I must insist..
#if velocityrange[0]<=vels.min() or velocityrange[1]>=vels.max():
#    raise ValueError("Add more points.  Something's going to be out of range for stupid star stupid link")
makecube.generate_header(49.209553, -0.277137, naxis1=192, naxis2=128,
                         pixsize=24, naxis3=naxis3, cd3=cd3, crval3=crval3,
                         clobber=True, restfreq=14.488479e9)
makecube.make_blank_images(cubename_lores,clobber=True)


for fn in [filename]:
    print "Adding file %s" % fn
    fullfn = fn
    makecube.add_file_to_cube(fullfn,
                              cubename_lores+".fits",
                              add_with_kernel=True,
                              kernel_fwhm=50./3600.,
                              nhits=cubename_lores+"_nhits.fits", wcstype='V',
                              diagnostic_plot_name=fullfn.replace('.fits','_data_scrubbed.png'),
                              velocityrange=velocityrange,excludefitrange=[40,75],noisecut=1.0)
                              # more aggressive noisecut

makecube.runscript(cubename_lores)
makecube.make_flats(cubename_lores,vrange=[45,75],noisevrange=[-15,30])
makecube.make_taucube(cubename_lores,cubename_lores+"_continuum.fits",etamb=0.886)
