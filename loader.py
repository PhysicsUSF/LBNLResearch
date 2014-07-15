# -*- coding: utf-8 -*-
import numpy as np
import os
import pyfits
import sncosmo as snc

import matplotlib.pyplot as plt
from pprint import pprint

#dirname = os.path.dirname(__file__)


##########################################################################################
##### LOAD SN2012CU SPECTRA ##############################################################

SN2012CU = []

# get files from 12cu data folder (exclude .yaml file)
FILES = [f for f in os.listdir('data/SNF-0201-INCR01a-2012cu') if f[-4:]!='yaml']

# date of B-max (given in .yaml file)
BMAX = 56104.7862735

for f in FILES:
    filename = 'data/SNF-0201-INCR01a-2012cu/'+f
    header = pyfits.getheader(filename)
    
    JD = header['JD']
    MJD = JD - 2400000.5  # convert JD to MJD
    PHASE = MJD - BMAX  # convert to phase by subtracting date of BMAX

    flux = pyfits.getdata(filename)  # this is the flux data

    CRVAL1 = header['CRVAL1']  # this is the starting wavelength of the spectrum
    CDELT1 = header['CDELT1']  # this is the wavelength incrememnet per pixel

    wave = [float(CRVAL1) + i*float(CDELT1) for i in xrange(flux.shape[0])]

    # make into Spectrum object and add to list with phase data
    SN2012CU.append( (PHASE, snc.Spectrum(wave, flux)) )
