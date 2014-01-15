"""
Quick outline of how the mapping works in principle.  Practically, this recipe
can be used for extremely quick-look continuum reductions of a mapping
observation
"""
from astropy.io import fits
from astropy import wcs
from astropy import coordinates
from astropy import units as u

def radec_to_gal(r,d):
    c = coordinates.ICRS(r,d,unit=(u.deg,u.deg))
    return c.galactic.lonangle.deg,c.galactic.latangle.deg

def quickmap(filename, headerfile, diagnostics=True):
    d = fits.getdata(filename)
    if diagnostics:
        import mpl_plot_templates
        p = mpl_plot_templates.imdiagnostics(d['DATA'])

    h = fits.getheader(headerfile)
    w = wcs.WCS(h)
    m = np.zeros([h['NAXIS2'],h['NAXIS1']])
    glon,glat = radec_to_gal(d['CRVAL2'],d['CRVAL3'])
    x,y = w.wcs_world2pix(glon,glat,0)
    md = (median(d['DATA'],axis=1))
    m[y.astype('int'),x.astype('int')] = md

    return m

