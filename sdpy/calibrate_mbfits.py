# -*- coding: utf-8 -*-

from astropy.io import fits

def get_scanid(header):
    if 'SCANNUM' in header and 'OBSNUM' in header:
        return header['SCANNUM'], header['SUBSNUM'], header['OBSNUM']

def hot_cold_sky(hdul, otypes, discard_fraction = 0.04):

    Tsky = {}
    Thot = {}
    Tcold = {}
    sky,hot,cold = {},{},{}

    hotcoldkws = ('THOTCOLD_A', 'THOTCOLD_HET230-XFFTS2_1', 'THOTCOLD_HET230-XFFTS2_2')

    for hdu in hdul:
        sid = get_scanid(hdu.header)
        if sid:
            scanno,subsno,obsno = sid
        else:
            continue
        otype = otypes[obsno]

        # Only interested in monitor values right now
        if 'MONPOINT' in hdu.data.dtype.names:
            if otype == 'HOT':
                vals = {}
                for kw in hotcoldkws:
                    sel = hdu.data['MONPOINT'] == kw
                    vals[kw] = hdu.data['MONVALUE'][sel]
                Thot[obsno] = vals
            elif otype == 'COLD':
                vals = {}
                for kw in hotcoldkws:
                    sel = hdu.data['MONPOINT'] == kw
                    vals[kw] = hdu.data['MONVALUE'][sel]
                Tcold[obsno] = vals
            elif otype == 'SKY':
                vals = {}
                for kw in hotcoldkws:
                    sel = hdu.data['MONPOINT'] == kw
                    vals[kw] = hdu.data['MONVALUE'][sel]
                Tsky[obsno] = vals
        elif 'INTEGTIM' in hdu.data.dtype.names:
            # this should always come before calvals
            integtime = hdu.data['INTEGTIM']
        elif 'DATA' in hdu.data.dtype.names:
            s1,s2,s3 = hdu.data['DATA'].shape
            if otype in ('SKY','COLD','HOT'):
                calvals = hdu.data['DATA'][:,0,discard_fraction*s3:(1-discard_fraction)*s3].sum(axis=1) / integtime
            if otype == 'SKY':
                sky[(obsno,hdu.header['BASEBAND'])] = calvals
            elif otype == 'COLD':
                cold[(obsno,hdu.header['BASEBAND'])] = calvals
            elif otype == 'HOT':
                hot[(obsno,hdu.header['BASEBAND'])] = calvals

    return hot,cold,sky

def tastar(inputs):
    """
    http://www.apex-telescope.org/documents/public/APEX-MPI-MAN-0012.pdf
    TA∗ = Tcal ∗ (Con − Cref )/(Chot − Csky )
    TA∗ = Tcal ∗ (Con − Cref )/Cref / ((Chot − Csky ) / Csky)
    """


def get_hot(hdu):
    pass

def get_scan_types(hdul, excludeabort=True, verbose=False):
    otypes = {}

    for hdu in hdul:
        if 'OBSTYPE' in hdu.header:
            if ((excludeabort and 'OBSSTATUS' in hdu.header and
                 hdu.header['OBSSTATUS'] != 'OK')):
                if verbose:
                    scanid = get_scanid(hdu.header)
                    print "Skipping scan %i in observation %i" % (scanid['SCANNUM'], scanid['OBSNUM'])
                continue
            otypes[hdu.header['OBSNUM']] = hdu.header['OBSTYPE']

    return otypes
