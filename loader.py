'''
::Author::
Andrew Stocker

::Description::
Loader module for 2012cu and 2011fe spectra, 2014j photometry, Pereira (2013) tophat
UBVRI filters, and the Fitzpatrick-Massa (1999) and Goobar (2008) extinction laws.  The
extinction laws are modified to artificially apply reddening.

::Last Modified::
07/22/2014

::Notes::
Source for A_lambda values for 2012cu -> http://ned.ipac.caltech.edu/forms/byname.html

'''
import numpy as np
import os
import pyfits
import sncosmo as snc

from pprint import pprint
dirname = os.getcwd()



################################################################################
##### LOAD UBVRi FILTERS #######################################################

# FILTER_PREFIX = 'NOT_' or 'tophat_'
def load_filters(FILTER_PREFIX='tophat_'):
    '''
    Load UBVRI tophat filters defined in Pereira (2013) into the sncosmo
    registry.  Also returns a dictionary of zero-point values in Vega system
    flux for each filter.  Flux is given in [photons/s/cm^2].

    Example usage:

        ZP_CACHE = loader.load_filters()
        U_band_zp = ZP_CACHE['U']
    '''
    # dictionary for zero point fluxes of filters
    ZP_CACHE = {'prefix': FILTER_PREFIX}

    for f in 'UBVRI':
        filter_name = FILTER_PREFIX+f
        file_name = dirname+'/data/filters/'+filter_name+'.dat'
        
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
def redden_pl(wave, flux, Av, p, return_excess=False):
    #wavelength in angstroms
    lamv = 5500
    a = 1.
    x = np.array(wave)
    #A_V = R_V * ebv

    Alam_over_Av = 1. - a + a*(x**p)/(lamv**p)
    A_lambda = -1* Av * Alam_over_Av

    #Rv = 1./(a*(0.8**p - 1.))
    #print "###Rv value:", Rv
    if not return_excess:
        VAL = flux * 10.**(0.4 * A_lambda)
        return VAL
    else:
        return Av + A_lambda
        
        
# Goobar (2008) power law artificial reddening law (with EBV and RV as params)
def redden_pl2(wave, flux, ebv, R_V, return_excess=False):
    #wavelength in angstroms
    lamv = 5500
    x = np.array(wave)
    
    Av = R_V * ebv
    p = np.log((1/R_V)+1)/np.log(0.8)
    
    Alam_over_Av = (x**p)/(lamv**p)
    A_lambda = -1* Av * Alam_over_Av

    #Rv = 1./(a*(0.8**p - 1.))
    #print "###Rv value:", Rv
    if not return_excess:
        VAL = flux * 10.**(0.4 * A_lambda)
        return VAL
    else:
        return Av + A_lambda


# Fitzpatrick-Massa (1999) artificial reddening law
def redden_fm(wave, flux, ebv, R_V, return_excess=False, *args, **kwargs):
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

    if not return_excess:
        # Now apply extinction correction to input flux vector
        curve = ebv * curve[0]
        flux = flux * 10.**(0.4 * curve)
        return flux
    else:
        Alam_over_AV = curve/float(R_V)
        A_V = R_V * ebv
        Alam = Alam_over_AV * A_V
        evx = A_V - Alam
        return -evx[0]



################################################################################
##### LOAD SN2012CU SPECTRA ####################################################


def get_12cu(redtype=None, ebv=None, rv=None, av=None, p=None):
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

    
    SN2012CU = sorted(SN2012CU, key=lambda t: t[0])
    
    if redtype==None:
        return SN2012CU
    elif redtype=='fm':
        if ebv!=None and rv!=None:
            return [(t[0], snc.Spectrum(t[1].wave, redden_fm(t[1].wave, t[1].flux, ebv, rv)))
                    for t in SN2012CU]
        else:
            msg = 'Fitzpatrick-Massa Reddening: Invalid values for [ebv] and/or [rv]'
            raise ValueError(msg)
    elif redtype=='pl':
        if av!=None and p!=None:
            return [(t[0], snc.Spectrum(t[1].wave, redden_pl(t[1].wave, t[1].flux, av, p)))
                    for t in SN2012CU]
        else:
            msg = 'Goobar Power-Law Reddeing: Invalid values for [av] and/or [p]'
            raise ValueError(msg)
    else:
        msg = 'Invalid reddening law name; must be either \'fm\' or \'pl\'.'
        raise ValueError(msg)
    


################################################################################
##### LOAD SN2011FE SPECTRA ####################################################


def get_11fe(redtype=None, ebv=None, rv=None, av=None, p=None,
             loadptf=True, loadsnf=True, loadmast=True):
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


    ## LOAD SNFACTORY SN2011FE SPECTRA ##
    
    if loadsnf:
        # get fits files from 12cu data folder (exclude README file)
        files = [f for f in os.listdir( dirname + '/data/sn2011fe' ) if f[-3:]!='txt']

        for F in files:
            filename = dirname + '/data/sn2011fe/' + F

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
                            'flux' : flux,
                            'set'  : 'SNF'
                            })


    ## LOAD MAST SN2011FE SPECTRA ##
            
    if loadmast:
        workingdir = dirname + '/data/sn2011fe_mast/'
        files = sorted(os.listdir(workingdir))

        # find proper phase buckets
        phase_buckets = {}
        for F in files:
            header = pyfits.getheader(workingdir+F)
            phase = ((int(header['TEXPEND' ])+int(header['TEXPSTRT']))/2)-55814.5

            actual_phase = ((float(header['TEXPEND' ])+float(header['TEXPSTRT']))/2)-55814.51

            if phase_buckets.has_key(phase):
                phase_buckets[phase].append(actual_phase)
            else:
                phase_buckets[phase] = []

        phase_buckets = {key:round(np.average(values), 2) for key, values in phase_buckets.items()}

        PHASES = {}
        for F in files:
            header = pyfits.getheader(workingdir+F)
            CENTRWV = 100*(int(header['CENTRWV'])/100)
            ### SN2011FE BMAX: 55814.5
            phase = ((int(header['TEXPEND' ])+int(header['TEXPSTRT']))/2)-55814.5
            phase = phase_buckets[phase]
            
            if not PHASES.has_key(phase):
                PHASES[phase] = {}
            if not PHASES[phase].has_key(CENTRWV):
                PHASES[phase][CENTRWV] = []
                
            D = pyfits.getdata(workingdir+F)
            wave = D.field('WAVELENGTH')[0]
            flux = D.field('FLUX')[0]
            
            PHASES[phase][CENTRWV].append({'wave': wave, 'flux': flux})
            
        for i, p in enumerate(sorted(PHASES.keys())):   
            SPECTRUM_DICT_LIST = PHASES[p]
            wave_concat = np.array([])
            flux_concat = np.array([])
            
            for LIST in SPECTRUM_DICT_LIST.values():
                n = len(LIST)
                wave_concat = np.concatenate( (wave_concat, (1.0/n)*sum([D['wave'] for D in LIST])) )
                flux_concat = np.concatenate( (flux_concat, (1.0/n)*sum([D['flux'] for D in LIST])) )
                
            I = wave_concat.argsort()
            wave_concat = wave_concat[I]
            flux_concat = flux_concat[I]

            SN2011FE.append({
                            'phase': p,
                            'wave' : wave_concat,
                            'flux' : flux_concat,
                            'set'  : 'MAST'
                            })

        del PHASES

    ## LOAD PTF11KLY SPECTRA ##
        
    if loadptf:
        workingdir = dirname + '/data/ptf11kly/'
        files = sorted(os.listdir(workingdir))

        for F in files:
            A = np.genfromtxt(workingdir + F, autostrip=True)

            mjd = float(F[9:15])/10  # get mjd from filename
            phase = round(mjd-55814.51, 2)
            
            wave, flux = A[:,0], A[:,1]

            SN2011FE.append({
                            'phase': phase,
                            'wave' : wave,
                            'flux' : flux,
                            'set'  : 'PTF11KLY'
                            })
                

    # sort list of dictionaries by phase
    SN2011FE = sorted([e for e in SN2011FE], key=lambda e: e['phase'])

    # return list of reddened spectra
    if redtype==None:
        return [(D['phase'], snc.Spectrum(D['wave'], D['flux']), D['set']) for D in SN2011FE]
    elif redtype=='fm':
        if ebv!=None and rv!=None:
            return [(D['phase'], snc.Spectrum(D['wave'], redden_fm(D['wave'], D['flux'], ebv, rv)), D['set'])
                    for D in SN2011FE]
        else:
            msg = 'Fitzpatrick-Massa Reddening: Invalid values for [ebv] and/or [rv]'
            raise ValueError(msg)
    elif redtype=='pl':
        if av!=None and p!=None:
            return [(D['phase'], snc.Spectrum(D['wave'], redden_pl(D['wave'], D['flux'], av, p)), D['set'])
                    for D in SN2011FE]
        else:
            msg = 'Goobar Power-Law Reddeing: Invalid values for [av] and/or [p]'
            raise ValueError(msg)
    else:
        msg = 'Invalid reddening law name; must be either \'fm\' or \'pl\'.'
        raise ValueError(msg)



################################################################################
##### LOAD SN2014J SPECTRA #####################################################


def get_14j():
    '''
    Gets a dictionary object with filters as keys; for example:

         'R': [{'AV': 0.19,
                'AX': 0.15,
                'Color': 0.06,
                'MJD': 56685.0,
                'Match': 'M',
                'Vmag': 10.92,
                'e_Vmag': 0.02,
                'e_mag': 0.01,
                'mag': 10.28,
                'phase': -3.5},
               {'AV': 0.19,
                'AX': 0.15,
                'Color': 0.06,
                'MJD': 56686.1,
                'Match': 'M',
                'Vmag': 10.84,
                ...
    
    Except for the V-band entry which is just data taken from the 'Vmag' column
    in the other entries, and look like this:

         'V': [{'AX': 0.19, 'mag': 10.84, 'phase': -4.4},
               {'AX': 0.15, 'mag': 10.97, 'phase': -3.6},
               {'AX': 0.19, 'mag': 10.92, 'phase': -3.5},
               {'AX': 0.19, 'mag': 10.92, 'phase': -3.4},
               {'AX': 0.19, 'mag': 10.84, 'phase': -2.7},
               ...


    ***Information on the column entries can be found in Amanullah (2008) table-1.
    
    '''
    # get data for SN2014J from data file
    SN2014J = {'V':[]}

    with open(dirname+"/data/amanulla_2014J_formatted.txt", "r") as f:
        for ROW in f:
            DATA = [item for item in ROW.split(" ") if item not in ['', '\n']]
            Filter = DATA[2].upper()

            if not SN2014J.has_key(Filter):
                SN2014J[Filter] = []
            
            SN2014J[Filter].append({'MJD':float(DATA[0]),
                                      'phase':float(DATA[1]),
                                      'mag':float(DATA[3]),
                                      'e_mag':float(DATA[4]),
                                      'AX':float(DATA[5]),
                                      'Match':DATA[6],
                                      'Vmag':float(DATA[7]),
                                      'e_Vmag':float(DATA[8]),
                                      'AV':float(DATA[9]),
                                      'Color':float(DATA[10])
                                      })
            if float(DATA[1]) not in [item['phase'] for item in SN2014J['V']]:
                SN2014J['V'].append({'phase':float(DATA[1]), 'mag':float(DATA[7]), 'AX':float(DATA[9])})

    for Filter, dictlist in SN2014J.items():
        SN2014J[Filter] = sorted([e for e in dictlist], key=lambda e: e['phase'])

    return SN2014J



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
    from scipy.interpolate import interp1d
    
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
                W1, W2 = spectra[i].wave, spectra[i+1].wave
                S1 = interp1d(W1, spectra[i].flux)
                S2 = interp1d(W2, spectra[i+1].flux)

                # get range of overlap between two spectra
                RNG = np.arange( max( np.min(W1), np.min(W2) ), min( np.max(W1), np.max(W2) ), 2)

                # compute linear interpolation
                S_interp = S1(RNG) + ((S2(RNG)-S1(RNG))/(p2-p1))*(float(phase)-p1)
                return (snc.Spectrum(RNG, S_interp), i)
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
### TOPHAT FILTER GENERATOR ####################################################

def generate_buckets(low, high, n, inverse_microns=False):
    '''
    function to generate [n] tophat filters within the
    range of [low, high] (given in Angrstroms), with one 'V' filter centered at
    the BESSEL-V filter's effective wavelength (5417.2 Angstroms).
    '''
    V_EFF = 5417.2

    if inverse_microns:
        V_EFF = 10000./V_EFF
        temp = low
        low = 10000./high
        high = 10000./temp
        STEP_SIZE = .01
        prefix = 'bucket_invmicron_'
    else:
        STEP_SIZE = 2
        prefix = 'bucket_angstrom_'

    zp_cache = {'prefix':prefix}

    hi_cut, lo_cut = high-V_EFF, V_EFF-low
    a = (n-1)/(1+(hi_cut/lo_cut))
    A, B = np.round(a), np.round(n-1-a)

    lo_bw, hi_bw = lo_cut/(A+0.5), hi_cut/(B+0.5)
    lo_diff = lo_cut-(A+0.5)*hi_bw
    hi_diff = hi_cut-(B+0.5)*lo_bw

    idx = np.argmin((lo_diff,hi_diff))
    BW = (lo_bw,hi_bw)[idx]
    LOW = (low,low+lo_diff)[idx]

    toregister = {}
    for i in xrange(n):
        start = LOW+i*BW
        end = LOW+(i+1)*BW

        wave = np.arange(start, end, STEP_SIZE)
        trans = np.ones(wave.shape[0])
        trans[0]=trans[-1]=0

        index = (str(i),'V')[ abs(V_EFF-(start+end)/2) < 1e-5 ]

        if inverse_microns:
            wave = sorted(10000./wave)

        toregister[index] = {'wave':wave, 'trans':trans}

    filters = []
    zpsys = snc.get_magsystem('vega')
    for index, info in toregister.items():
        wave, trans = info['wave'], info['trans']
        bandpass = snc.Bandpass(wave, trans)
        
        snc.registry.register(bandpass, prefix+index, force=True)
        zp_phot = zpsys.zpbandflux(prefix+index)
        zp_cache[index] = zp_phot

        filters.append(index)
    
    return filters, zp_cache



################################################################################
##### REDDENING LAW LEAST SQUARE FITTING HELPER ################################


def calc_ftz_lsq_fit(S1, S2, filters, zp, ebv, rv_guess,
                     dist_mod_true, dist_mod_weight, constrain_dist_mod=True, weight_dist_mod=True):
    '''
    Least Square Fitter for Fitzpatrick-Massa (1999) reddening law.  This function assumes
    that S1 is an unreddened twin of S2 -- in our case S1 is 


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


    prefix = zp['prefix']
        
    ############################################################################
    ##### FUNCTIONS ############################################################

    # returns a lightcurve for the given filter
    def bandmags(f, spectra):
        bandfluxes = [s.bandflux(prefix+f) for s in spectra]
        return -2.5*np.log10( bandfluxes/zp[f] )
    
    # function to be used in least-sq optimization; must only take a numpy array (y) as input
    def lsq_func(Y):
        rv = Y[0]
        #dist_mod_shift = Y[1]
        
        reddened = [snc.Spectrum(spec.wave, redden_fm(spec.wave, spec.flux, ebv, rv)) for spec in s1_spectra]
        reddened_mags = [bandmags(f, reddened) for f in filters]

##        if constrain_dist_mod:
##            dist_mod_shift = dist_mod_true
##        if weight_dist_mod:
##            S1_REF = np.concatenate( [mag+dist_mod_shift for mag in reddened_mags] )
##            return np.concatenate(( S2_REF - S1_REF , [dist_mod_weight*(dist_mod_shift - dist_mod_true)] ))
##        else:
##            S1_REF = np.concatenate( [mag+dist_mod_shift for mag in reddened_mags] )
##            return S2_REF - S1_REF

        S1_REF = np.concatenate( [mags+dist_mod_true for mags in reddened_mags] )
        return S2_REF - S1_REF

    
    ############################################################################
    s1_spectra, s2_spectra = [t[1] for t in S1], [t[1] for t in S2]

    SN12CU_MW = dict( zip( 'UBVRI', [0.117, 0.098, 0.074, 0.058, 0.041] ))  # 12cu Milky Way extinction
    
    # get a concatenated array of lightcurves per filter, also correct for 12cu milky way extinction
    S2_REF = np.concatenate( [bandmags(f, s2_spectra) - SN12CU_MW[f] for f in filters] )
    
    #Y = np.concatenate(( red_vars, [dist_mod_true] ))  # best guess vector
    #Y = np.array([rv_guess, dist_mod_true])
    Y = np.array([rv_guess])
    
    return lsq(lsq_func, Y)  # , full_output=False, xtol=0.01, ftol=0.01)















