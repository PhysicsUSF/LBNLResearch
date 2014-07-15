# -*- coding: utf-8 -*-
import numpy as np
import os
import pyfits
import sncosmo as snc

from pprint import pprint

dirname = os.path.dirname(__file__)


##########################################################################################
##### LOAD SN2012CU SPECTRA ##############################################################


SN2012CU = []

# get files from 12cu data folder (exclude .yaml file)
FILES = [f for f in os.listdir( dirname + '/data/SNF-0201-INCR01a-2012cu' ) if f[-4:]!='yaml']

# date of B-max (given in .yaml file)
BMAX = 56104.7862735


phases = {}  # dictionary of spectra to be concatenated

for f in FILES:
    filename = dirname + '/data/SNF-0201-INCR01a-2012cu/' + f
    header = pyfits.getheader(filename)
    
    JD = header['JD']
    MJD = JD - 2400000.5  # convert JD to MJD
    PHASE = round(MJD - BMAX, 1)  # convert to phase by subtracting date of BMAX

    flux = pyfits.getdata(filename)  # this is the flux data

    CRVAL1 = header['CRVAL1']  # this is the starting wavelength of the spectrum
    CDELT1 = header['CDELT1']  # this is the wavelength incrememnet per pixel

    wave = np.array([float(CRVAL1) + i*float(CDELT1) for i in xrange(flux.shape[0])])

    if not phases.has_key(PHASE):
        phases[PHASE] = []
        
    phases[PHASE].append({'wave': wave, 'flux': flux})


# concatenate spectra at same phases
for phase, dict_list in phases.items():
    wave_concat = np.array([])
    flux_concat = np.array([])

    for d in dict_list:
        wave_concat = np.concatenate( (wave_concat, d['wave']) )
        flux_concat = np.concatenate( (flux_concat, d['flux']) )

    # sort wavelength array and flux array ordered by wavelength
    I = wave_concat.argsort()
    wave_concat = wave_concat[I]
    flux_concat = flux_concat[I]

    # make into Spectrum object and add to list with phase data
    SN2012CU.append( (phase, snc.Spectrum(wave_concat, flux_concat)) )

SN2012CU = sorted(SN2012CU, key=lambda t: t[0])


##########################################################################################
##### LOAD UBVRi FILTERS #################################################################

# http://ned.ipac.caltech.edu/forms/byname.html


ZP_CACHE_VEGA = {}  # dictionary for zero point fluxes of filters

for f in 'UBVRI':
    filter_name = 'tophat_'+f
    file_name = dirname + '/data/filters/' + filter_name + '.dat'
    
    try:
        snc.get_bandpass(filter_name)
    except:
        data = np.genfromtxt(file_name)
        bandpass = snc.Bandpass( data[:,0], data[:,1] )
        snc.registry.register(bandpass, filter_name)

    zpsys = snc.get_magsystem('vega')
    zp_phot = zpsys.zpbandflux(filter_name)

    ZP_CACHE_VEGA[f] = zp_phot
    





































