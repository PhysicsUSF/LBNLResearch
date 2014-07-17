# -*- coding: utf-8 -*-
'''
::Author::
Andrew Stocker

::Description::
Loader module for 2012cu and 2011fe spectra, Pereira (2013) tophat UBVRI filters, and
the Fitzpatrick-Massa (1999) and Goobar (2008) extinction laws.  The extinction laws are
modified to artificially apply reddening.

::Last Modified::
07/16/2014

::Notes::
Source for A_lambda values for 2012cu -> http://ned.ipac.caltech.edu/forms/byname.html

'''
import numpy as np
import os
import pyfits
import sncosmo as snc

from pprint import pprint
dirname = os.path.dirname(__file__)


################################################################################
##### LOAD UBVRi FILTERS #######################################################


def load_filters():
    '''
    Load UBVRI tophat filters defined in Pereira (2013) into the sncosmo
    registry.  Also returns a dictionary of zero-point values in Vega system
    flux for each filter.  Flux is given in [photons/s/cm^2].

    Example usage:

        ZP_CACHE = loader.load_filters()
        U_band_zp = ZP_CACHE['U']
    '''
    ZP_CACHE = {}  # dictionary for zero point fluxes of filters

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

        ZP_CACHE[f] = zp_phot

    return ZP_CACHE


################################################################################
##### EXTINCTION LAWS ##########################################################


# Goobar (2008) power law artificial reddening law
def redden_pl(wave, flux, Av, p):
    #wavelength in angstroms
    #lamv = 5500
    lamv = 5366
    a = 1.
    x = np.array(wave)
    #A_V = R_V * ebv

    Alam_over_Av = 1. - a + a*(x**p)/(lamv**p)
    A_lambda = -1* Av * Alam_over_Av

    Rv = 1./(a*(0.8**p - 1.))
    #print "###Rv value:", Rv
    VAL = flux * 10.**(0.4 * A_lambda)
    return VAL


# Fitzpatrick-Massa (1999) artificial reddening law
def redden_fm(wave, flux, ebv, R_V, *args, **kwargs):
    '''
    given wavelength and flux info of a spectrum, return:
    reddened spectrum fluxes (ebv < 0)
    unreddened spectrum fluxes (ebv > 0)
    '''
    # Import needed modules
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    import numpy as n

    # Set defaults
    lmc2_set, avglmc_set, extcurve_set = None, None, None
    gamma, x0, c1, c2, c3, c4 = None, None, None, None, None, None
    
    x = 10000. / n.array([wave])                # Convert to inverse microns
    curve = x * 0.

    # Read in keywords
    for arg in args:
        if arg.lower() == 'lmc2': lmc2_set = 1
        if arg.lower() == 'avglmc': avglmc_set = 1
        if arg.lower() == 'extcurve': extcurve_set = 1
        
    for key in kwargs:
        if key.lower() == 'x0':
            x0 = kwargs[key]
        if key.lower() == 'gamma':
            gamma = kwargs[key]
        if key.lower() == 'c4':
            c4 = kwargs[key]
        if key.lower() == 'c3':
            c3 = kwargs[key]
        if key.lower() == 'c2':
            c2 = kwargs[key]
        if key.lower() == 'c1':
            c1 = kwargs[key]

    if R_V == None: R_V = 3.1

    if lmc2_set == 1:
        if x0 == None: x0 = 4.626
        if gamma == None: gamma =  1.05	
        if c4 == None: c4 = 0.42   
        if c3 == None: c3 = 1.92	
        if c2 == None: c2 = 1.31
        if c1 == None: c1 = -2.16
    elif avglmc_set == 1:
        if x0 == None: x0 = 4.596  
        if gamma == None: gamma = 0.91
        if c4 == None: c4 = 0.64  
        if c3 == None: c3 =  2.73	
        if c2 == None: c2 = 1.11
        if c1 == None: c1 = -1.28
    else:
        if x0 == None: x0 = 4.596  
        if gamma == None: gamma = 0.99
        if c4 == None: c4 = 0.41
        if c3 == None: c3 =  3.23	
        if c2 == None: c2 = -0.824 + 4.717 / R_V
        if c1 == None: c1 = 2.030 - 3.007 * c2
    
    # Compute UV portion of A(lambda)/E(B-V) curve using FM fitting function and 
    # R-dependent coefficients
 
    xcutuv = 10000.0 / 2700.0
    xspluv = 10000.0 / n.array([2700.0, 2600.0])
   
    iuv = n.where(x >= xcutuv)
    iuv_comp = n.where(x < xcutuv)

    if len(x[iuv]) > 0: xuv = n.concatenate( (xspluv, x[iuv]) )
    else: xuv = xspluv.copy()

    yuv = c1  + c2 * xuv
    yuv = yuv + c3 * xuv**2 / ( ( xuv**2 - x0**2 )**2 + ( xuv * gamma )**2 )

    filter1 = xuv.copy()
    filter1[n.where(xuv <= 5.9)] = 5.9
    
    yuv = yuv + c4 * ( 0.5392 * ( filter1 - 5.9 )**2 + 0.05644 * ( filter1 - 5.9 )**3 )
    yuv = yuv + R_V
    yspluv = yuv[0:2].copy()                  # save spline points
    
    if len(x[iuv]) > 0: curve[iuv] = yuv[2:len(yuv)]      # remove spline points

    # Compute optical portion of A(lambda)/E(B-V) curve
    # using cubic spline anchored in UV, optical, and IR

    xsplopir = n.concatenate(([0], 10000.0 / n.array([26500.0, 12200.0, 6000.0, 5470.0, 4670.0, 4110.0])))
    ysplir   = n.array([0.0, 0.26469, 0.82925]) * R_V / 3.1
    ysplop   = [n.polyval(n.array([2.13572e-04, 1.00270, -4.22809e-01]), R_V ), 
                n.polyval(n.array([-7.35778e-05, 1.00216, -5.13540e-02]), R_V ),
                n.polyval(n.array([-3.32598e-05, 1.00184, 7.00127e-01]), R_V ),
                n.polyval(n.array([-4.45636e-05, 7.97809e-04, -5.46959e-03, 1.01707, 1.19456] ), R_V ) ]
    
    ysplopir = n.concatenate( (ysplir, ysplop) )
    
    if len(iuv_comp) > 0:
        cubic = spline(n.concatenate( (xsplopir,xspluv) ), n.concatenate( (ysplopir,yspluv) ), k=3)
        curve[iuv_comp] = cubic( x[iuv_comp] )

    # Now apply extinction correction to input flux vector
    curve = ebv * curve[0]
    flux = flux * 10.**(0.4 * curve)
    
    return flux


################################################################################
##### LOAD SN2012CU SPECTRA ####################################################


def get_12cu():
    '''
    Function to get SN2012CU spectra.  This returns a sorted list of the form;

        [(phase0, sncosmo.Spectrum), (phase1, sncosmo.Spectrum), ... ]

    To split the list it is easy to use a list comprehension;

        phases  = [t[0] for t in SN2012CU]
        spectra = [t[1] for t in SN2012CU]

    To get the synthetic photometry it is also easy to use a list comprehension coupled with
    sncosmo's built-in bandflux() method;

        bandfluxes = [t[1].bandflux(filter_name) for t in SN2012CU]

    However, sometimes SNCOSMO will return None for the bandflux if the filter
    transmission curve has a wider range than the given spectrum, in this case it is convenient
    to use a filter;

        filtered = filter( lambda x: x[1]!=None, izip(phases, bandfluxes) )
    
    '''
    SN2012CU = []

    # get fits files from 12cu data folder (exclude .yaml file)
    FILES = [f for f in os.listdir( dirname + '/data/SNF-0201-INCR01a-2012cu' ) if f[-4:]!='yaml']

    # date of B-max (given in .yaml file)
    BMAX = 56104.7862735


    _12CU = {}  # dictionary of spectra to be concatenated; keys are phases

    for f in FILES:
        filename = dirname + '/data/SNF-0201-INCR01a-2012cu/' + f
        header = pyfits.getheader(filename)
        
        JD = header['JD']
        MJD = JD - 2400000.5  # convert JD to MJD
        phase = round(MJD - BMAX, 1)  # convert to phase by subtracting date of BMAX

        flux = pyfits.getdata(filename)  # this is the flux data

        CRVAL1 = header['CRVAL1']  # this is the starting wavelength of the spectrum
        CDELT1 = header['CDELT1']  # this is the wavelength incrememnet per pixel

        wave = np.array([float(CRVAL1) + i*float(CDELT1) for i in xrange(flux.shape[0])])

        if not _12CU.has_key(phase):
            _12CU[phase] = []
            
        _12CU[phase].append({'wave': wave, 'flux': flux})


    # concatenate spectra at same phases
    for phase, dict_list in _12CU.items():
        wave_concat = np.array([])
        flux_concat = np.array([])

        for d in dict_list:
            wave_concat = np.concatenate( (wave_concat, d['wave']) )
            flux_concat = np.concatenate( (flux_concat, d['flux']) )

        # sort wavelength array and flux array ordered by wavelength
        #   argsort() docs with helpful examples ->
        #       http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
        I = wave_concat.argsort()
        wave_concat = wave_concat[I]
        flux_concat = flux_concat[I]

        # make into Spectrum object and add to list with phase data
        SN2012CU.append( (phase, snc.Spectrum(wave_concat, flux_concat)) )


    return sorted(SN2012CU, key=lambda t: t[0])  # sort by phase


################################################################################
##### LOAD SN2011FE SPECTRA ####################################################


def get_11fe(redtype=None, ebv=None, rv=None, av=None, p=None):
    '''
    This function operates similarly to get_12cu() and returns a list of tuples of the form;

        [(phase0, sncosmo.Spectrum), (phase1, sncosmo.Spectrum), ... ]

    However an artificial reddening can also be applied passing in as an argument either
    redtype='fm' to use the Fitzpatrick-Massa (1999) reddening law or redtype='pl' for
    Goobar's Power Law (2008).  For example;

        FMreddened_2011fe = loader.get_11fe('fm', ebv=-1.37, rv=1.4)
        PLreddened_2011fe = loader.get_11fe('pl', av=1.85, p=-2.1)
    '''
    SN2011FE = []  # list of info dictionaries

    # get fits files from 12cu data folder (exclude README file)
    FILES = [f for f in os.listdir( dirname + '/data/sn2011fe' ) if f[-3:]!='txt']

    for f in FILES:
        filename = dirname + '/data/sn2011fe/' + f

        header = pyfits.getheader(filename)

        CRVAL1 = header['CRVAL1'] # coordinate start value
        CDELT1 = header['CDELT1'] # coordinate increment per pixel
        TMAX   = header['TMAX']   # phase in days relative to B-band maximum
        
        phase = float(TMAX)
        flux = pyfits.getdata(filename,0)
        wave = [float(CRVAL1) + i*float(CDELT1) for i in xrange(flux.shape[0])]
        
        SN2011FE.append({
            'phase': phase,
            'wave' : wave,
            'flux' : flux
            })

    # sort list of dictionaries by phase
    SN2011FE = sorted([e for e in SN2011FE], key=lambda e: e['phase'])

    # return list of reddened spectra
    if redtype==None:
        return [(D['phase'], snc.Spectrum(D['wave'], D['flux'])) for D in SN2011FE]
    elif redtype=='fm':
        if ebv!=None and rv!=None:
            return [(D['phase'], snc.Spectrum(D['wave'], redden_fm(D['wave'], D['flux'], ebv, rv)))
                    for D in SN2011FE]
        else:
            msg = 'Fitzpatrick-Massa Reddening: Invalid values for [ebv] and/or [rv]'
            raise ValueError(msg)
    elif redtype=='pl':
        if av!=None and p!=None:
            return [(D['phase'], snc.Spectrum(D['wave'], redden_pl(D['wave'], D['flux'], av, p)))
                    for D in SN2011FE]
        else:
            msg = 'Goobar Power-Law Reddeing: Invalid values for [av] and/or [p]'
            raise ValueError(msg)
    else:
        msg = 'Invalid reddening law name; must be either \'fm\' or \'pl\'.'
        raise ValueError(msg)


################################################################################
##### SPECTRUM LINEAR INTERPOLATION HELPER FUNCTION ############################

    
def interpolate_spectra(phase_array, spectra):
    '''
    Function to linearly interpolate a spectrum at a specific phase or array of phases, given
    a list of spectra and their respectrive phases; i.e. [spectra] must be in the form:

        [(phase0, sncosmo.Spectrum), (phase1, sncosmo.Spectrum), ... ]

    This is the same form as the output of get_12cu() or get_11fe().  Example usage:

        interpolated_spectra = loader.interpolate_spectra( [0.0, 12.0, 24.0], loader.get_12cu() )

    The output is also going to be in the for as the input for spectra, with an interpolated
    spectrum in a tuple with each phase in phase_array.  However, there will be a None value
    in place of the spectrum if either the phase is out of the phase range for the given spectra
    or the wavelength arrays do not match, e.g.:

        [(-6.5, <sncosmo.spectral.Spectrum object at 0x3480f10>),
         (-3.5, <sncosmo.spectral.Spectrum object at 0x3480890>),
         ...
         (46.5, <sncosmo.spectral.Spectrum object at 0x3414cd0>),
         (188.8, None),
         (201.8, None)]

    Tuples in the list with None values can be filtered out like this;

        filtered_spectra = filter(lambda t: t[1]!=None, spectra)
    
    **NOTE: This function assumes that the wavelength array is the same for each spectrum in the
            the given list.  It will return None when trying to interpolate between two spectra
            with different wavelength arrays.
        
    '''
    interpolated = []
    phases  = [t[0] for t in spectra]
    spectra = [t[1] for t in spectra]

    if type(phase_array) == type([]):
        phase_array = np.array(phase_array)
    elif type(phase_array) != type(np.array([])):
        try:
            phase_array = np.array([float(phase_array)])
        except:
            return None
    
    def interpolate(phase, start=0):
        if phase < phases[0]:
            return (None, 0)

        LIM = len(phases)-1
        i = start
        while i < LIM:
            if phase < phases[i+1]:
                p1, p2 = float(phases[i]), float(phases[i+1])
                S1, S2 = spectra[i].flux, spectra[i+1].flux  # these are numpy arrays
                W1, W2 = spectra[i].wave, spectra[i+1].wave  # these are numpy arrays
                
                # check in wavelength arrays are the same, if not then return None for the
                #  interpolated spectrum
                if not np.array_equal(W1, W2):
                    return (None, i)
                
                S_interp = S1 + ((S2-S1)/(p2-p1))*(float(phase)-p1)  # compute linear interpolation
                return (snc.Spectrum(W1, S_interp), i)
            i += 1

        return (None, 0)

    LIM = phase_array.shape[0]
    search_index = 0  # keep track of current place in phases array in order to speed up interpolation
    for i in xrange(LIM):
        S_interp, search_index = interpolate(phase_array[i], search_index)
        # rounded phase below because of weird numpy float rounding errors when converting phase_array to
        # a numpy array
        interpolated.append((round(phase_array[i], 1), S_interp))  

    if len(interpolated) == 1:
        return interpolated[0]  # return only the tuple if only interpolating for one phase
    return interpolated
    

################################################################################
##### REDDENING LAW LEAST SQUARE FITTING HELPER ################################


def calc_lsq_fit(S1, S2, filters, zp, redtype, x0):
    '''
    ::NOTES FOR ME::
    must input spectra with matching phases

    fit S1 to S2
    
    1. calc 12cu mags
    2. interpolate with pristine 11fe spectra at 12cu phases
    3. run lsq_helper_func with pristine spectra
    
    x0 is a dict like:
        {'ebv': -1.37, 'rv':  1.4}
        { 'av':  1.85,  'p': -2.1}
        
    DOCS -> http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html

    xtol/ftol == something big like .01?
    
    '''
    from scipy.optimize import leastsq as lsq
    
    # check that given spectra have matching phases
    s1_phases, s2_phases = np.array([t[0] for t in S1]), np.array([t[0] for t in S2])
    
    if not np.array_equal(s1_phases, s2_phases):
        return ValueError('Phases in given spectra do not match!')

    if redtype=='fm':
        Y = np.array([float(x0['ebv']), float(x0['rv'])])
        REDLAW = redden_fm
        
    elif redtype=='pl':
        Y = np.array([float( x0['av']), float( x0['p'])])
        REDLAW = redden_pl
        
    else:
        msg = 'Invalid reddening law name; must be either \'fm\' or \'pl\'.'
        raise ValueError(msg)
    
    ############################################################################
    ##### FUNCTIONS ############################################################

    # returns a lightcurve for the given filter
    def bandmags(f, spectra):
        bandfluxes = [s.bandflux('tophat_'+f) for s in spectra]
        return -2.5*np.log10( bandfluxes/zp[f] )

    
    # function to be used in least-sq optimization; must only take a numpy array (y) as input
    def lsq_func(y):
        
        reddened = [snc.Spectrum(spec.wave, REDLAW(spec.wave, spec.flux, y[0], y[1])) for spec in s1_spectra]
        reddened_mags = [bandmags(f, reddened) for f in filters]
        s1_mins = np.array([np.min(lc) for lc in reddened_mags])

        global BMAX_SHIFTS
        BMAX_SHIFTS = S2_MINS - s1_mins
        
        S1_REF = np.concatenate( [mag+BMAX_SHIFTS[i] for i, mag in enumerate(reddened_mags)] )
        print S2_REF-S1_REF
        return np.concatenate(( S2_REF - S1_REF, BMAX_SHIFTS - 5*np.log10(61./21.) ))
        
    ############################################################################
    # s1 is 11fe, s2 is 12cu
    s1_spectra, s2_spectra = [t[1] for t in S1], [t[1] for t in S2]
    
    # get a concatenated array of lightcurves per filter
    s2_bandmags = [bandmags(f, s2_spectra) for f in filters]
                   
    S2_MINS = np.array([np.min(lc) for lc in s2_bandmags])
    S2_REF = np.concatenate(s2_bandmags)

    lsq_out = lsq(lsq_func, Y, full_output=False)

    from copy import deepcopy
    bmax_return = deepcopy(BMAX_SHIFTS)
    
    return lsq_out, bmax_return















