'''
::AUTHOR::
Andrew Stocker

'''
import loader as l
from loader import redden_fm
from copy import deepcopy
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import cPickle
from time import localtime, strftime, gmtime

from scipy.interpolate import interp1d



def main():
    
        # config
        ebv_guess = 1.05
        ebv_pad = 0.25
        
        rv_guess = 3.0
        rv_pad = 0.6
        
        steps = 21
        #steps = 120
        
        
        FEATURES_ACTUAL = [(3425, 3820, 'CaII'), (3900, 4100, 'SiII'), (5640, 5900, 'SiII'),
                               (6000, 6280, 'SiII'), (8000, 8550, 'CaII')]

#FEATURES_ACTUAL = [(300, 310, 'Null'),]  # does NOT block anything.


        # Use an empty list of features to fit for the entire spectrum:
        #FEATURES_ACTUAL = []
        
        
        # load spectra, interpolate 11fe to 12cu phases (only first 12)
        pristine_12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)[:12]
        phases = [t[0] for t in pristine_12cu]
        
        
        pristine_11fe = l.interpolate_spectra(phases, l.get_11fe(loadmast=False, loadptf=False))
        
        
        ########################
        # Choose 'reddened' to be either an artificially reddened 11fe interpolated
        # to the phases of 12cu, or just choose 12cu itself.
        #
        #reddened = l.interpolate_spectra(phases, l.get_11fe('fm', ebv=-0.4, rv=2.45, loadmast=False, loadptf=False))
        reddened = pristine_12cu
        #
        ########################
        
        
        ########################
        ### helper functions ###


        def filter_features(features, wave):
            '''Returns a mask of boolean values the same size as
                the wave array.  True=wavelength not in features, False=wavelength
                is in featured.
        
            Can be used like:
            
            mask = filter_features(FEAURES, wave)
            wave_no_features = wave[mask]
            flux_no_features = flux[mask]
                '''
            intersection = np.array([False]*wave.shape[0])
            for feature in features:
                intersection |= ((wave>feature[0])&(wave<feature[1]))
                        
            return ~intersection



        def log(msg=""):
                # attach time stamp to print statements
                print "[{}] {}".format(strftime("%Y-%m-%d %H:%M:%S", localtime()), msg)
        
        
        
        def flux2mag(flux, var=None, calibration_err=None):
                mag_var = None
                calibration_err_mag = None

                mag = -2.5*np.log10(flux)
                
                # calculate magnitude error
                if type(var)!=type(None):
                        fr_err = np.sqrt(var)/flux
                        mag_var = (1.0857362*fr_err)**2  # 2.5/np.log(10) = 1.0857362
                
                # calculate calibration error in mag space
                if type(calibration_err)!=type(None):
                        calibration_err_mag = 1.0857362*calibration_err
                
                results = tuple([r for r in (mag, mag_var, calibration_err_mag) if type(r)!=type(None)])
                
                return (results, results[0])[len(results)==1]
        
        ########################
        
        
        
        def perphase_fit():
                
                tmpx = np.linspace(ebv_guess-ebv_pad, ebv_guess+ebv_pad, steps)
                tmpy = np.linspace(rv_guess-rv_pad, rv_guess+rv_pad, steps)
                
                log( "SIZE OF GRID: {}".format(steps) )
                log( "EBV SEARCH GRID:" )
                log( tmpx )
                log( "RV SEARCH GRID:" )
                log( tmpy )
                
                best_rvs = []
                best_ebvs = []
                best_avs = []
                chi2s = []
                min_chi2s = []
                chi2_reductions = []
                
                #V_band = [(5300., 5500., 'Vband')]

                del_lamb = 1.
                V_wave = 5413.5  # see my mag_spectrum_plot_excess.py
                band_steps = 1200
                V_band_range = np.linspace(V_wave - del_lamb*band_steps/2., V_wave + del_lamb*band_steps/2., band_steps+1)

                
                for phase_index in xrange(len(phases)):
                    
                    
                        ref = pristine_11fe[phase_index]
                        red = reddened[phase_index]
                        
                        # mask for spectral features not included in fit
                        mask = filter_features(FEATURES_ACTUAL, ref[1].wave)
                        
                        log()
                        log( "Phase: {}".format(ref[0]) )
                        
                        
                        ### COVARIANCE MATRIX ###
                        
                        # pristine_11fe
                        ref_wave = ref[1].wave
                        ref_flux = ref[1].flux
                        ref_flux_avg = np.average(ref_flux)
                        ref_flux_avg_mag = -2.5*np.log10(ref_flux_avg)
                        #ref_flux_avg_mag = -2.5*np.log10(ref_flux[np.abs(ref_wave - 5413.5).argmin()])  #normalize with single-wavelength V band magnitude.

                        ref_interp = interp1d(ref_wave, ref_flux)


                        ref_var = ref[1].error
                        ref_calibration_error = ref[2]
                        
                        # convert flux, variance, and calibration error to magnitude space

                        ref_mag, ref_mag_var, ref_calibration_error_mag \
                        = flux2mag(ref_flux, ref_var, ref_calibration_error)
                        
                        # normalize for later use
                        #Vband_mask = filter_features(V_band, ref_wave) # Not the most efficient way of doing things, but this statement is here because ref_wave is inside the for loop -- also inefficient. Should fix this.
                        ref_V_mag = -2.5*np.log10(ref_interp(V_band_range).mean())

#ref_flux_V_mag = -2.5*np.log10(np.average(ref_flux[Vband_mask]))

                        ref_mag_norm = ref_mag - ref_flux_avg_mag  #ref_V_mag
    
                        # get mask for nan-values
                        nanmask_ref = ~np.isnan(ref_mag_norm[mask])
                        
                        #fig = plt.figure()
                        #plt.plot(ref_wave[mask], ref_flux[mask], 'ro')
                        #plt.plot(ref_wave, ref_flux)
                        #plt.show()
                        
                        # 12cu/reddened 11fe
                        red_wave = red[1].wave
#     red_flux = interp1d(red_wave, red[1].flux)(ref_wave)

                        red_interp = interp1d(red_wave, red[1].flux)
    
                        red_flux = red_interp(ref_wave)



                        ## test ######################################################################
                        #red_noisy_flux = (1 + 0.05*np.random.randn(red[1].flux.shape[0]) )*red[1].flux
                        #red_flux = interp1d(red_wave, red_noisy_flux)(ref_wave)
                        ##############################################################################
                        
                        red_interp_var  = interp1d(red_wave, red[1].error)(ref_wave)
                        red_calibration_error = red[2]
                        
                        # convert flux, variance, and calibration error to magnitude space

                        if (red_flux <= 0).any():
                            print "some red_flux values are not positive:", red_flux[np.where(red_flux <= 0)]
                            print "These values will be rejected below as nan for the log."
                            print "(But it's better to deal with the non-pos values before taking the log.  Something to deal with later.)"

                        red_interp_mag, red_interp_mag_var, red_calibration_error_mag \
                        = flux2mag(red_flux, red_interp_var, red_calibration_error)
                        
                        # get mask for nanvalues
                        nanmask_red_interp = ~np.isnan(red_interp_mag[mask])
                        
                        
                        # calculate cov matrices
                        V_ref = np.diag(ref_mag_var[mask])
                        S_ref = (ref_calibration_error_mag**2)*np.ones(V_ref.shape)
                        C_ref = V_ref # + S_ref
                        
                        V_red = np.diag(red_interp_mag_var[mask])
                        S_red = (red_calibration_error_mag**2)*np.ones(V_red.shape)
                        C_red = V_red # + S_red
                        
                        #########################
                        
                        def print_diagnostics():
                                log( "ref_flux:" )
                                log( ref_flux )
                                log( "ref_mag:" )
                                log( ref_mag )
                                log( "ref_mag_var:" )
                                log( ref_mag_var )
                                log( "ref_calibration_error_mag:" )
                                log( ref_calibration_error_mag )
                                log( "V_ref:" )
                                log( V_ref )
                                log( "S_ref:" )
                                log( S_ref )
                                log( "C_ref:" )
                                log( C_ref )
                                log( "red_interp_mag:" )
                                log( red_interp_mag )
                                log( "red_interp_mag_var:" )
                                log( red_interp_mag_var )
                                log( "red_calibration_error_mag:" )
                                log( red_calibration_error_mag )
                                log( "V_red:" )
                                log( V_red )
                                log( "S_red:" )
                                log( S_red )
                                log( "C_red:" )
                                log( C_red )
                        
                        #print_diagnostics()
                        
                        #########################
                        # INVERT TOTAL COVARIANCE MATRIX
                        
                        C_total = C_ref + C_red
                        
                        log( "Computing inverse..." )
                        C_total_inv = np.matrix(np.linalg.inv(C_total))
                        
                        #########################
                        
                        
                        
                        #################################################
                        # hack thrown together to filter nan-values (which arrise from negative fluxes)
                        
                        # find any rows with nan-values in C_inv matrix (there shouldn't be any)
                        nanmask = np.array(~np.max(np.isnan(C_total_inv), axis=1))[:,0]
                        
                        # merge mask with nan-masks from red_interp_mag, and ref_mag (calc'd above)
                        nanmask = nanmask & nanmask_red_interp & nanmask_ref
                        
                        log( "num. points with negative flux discarded: {}".format(np.sum(~nanmask)) )
                        
                        # create temp version of C_total_inv without rows/columns corresponding to nan-values
                        nanmatrix = np.outer(nanmask, nanmask)
                        TMP_C_total_inv = np.matrix(C_total_inv[nanmatrix].reshape(np.sum(nanmask), np.sum(nanmask)))
                        #################################################
                        
                        
                        
                        # for calculation of CHI2 per dof
                        CHI2_reduction = np.sum(nanmask) - 2  # (num. data points)-(num. parameters)
                        log( "CHI2 reduction: {}".format(CHI2_reduction) )
                        
                        #################################################
                        
                        x = np.linspace(ebv_guess-ebv_pad, ebv_guess+ebv_pad, steps)
                        y = np.linspace(rv_guess-rv_pad, rv_guess+rv_pad, steps)
                        
                        X, Y = np.meshgrid(x, y)
                        CHI2 = np.zeros( X.shape )
                        
                        log( "Scanning CHI2 grid..." )
                        for j, EBV in enumerate(x):
                                for k, RV in enumerate(y):
                                        
                                        # unredden the reddened spectrum, convert to mag
                                        unred_flux = redden_fm(ref_wave, red_flux, EBV, RV)
                                        unred_mag = flux2mag(unred_flux)
                                        
                                        unred_interp = interp1d(ref_wave, unred_flux)
                                        
                                        # normalization
                                        unred_flux_avg_mag = -2.5*np.log10(np.average(unred_flux))
                                        
                                        unred_V_mag = -2.5*np.log10(unred_interp(V_band_range).mean())
                                        #unred_flux_avg_mag = -2.5*np.log10(unred_flux[np.abs(ref_wave - 5413.5).argmin()])  #normalize with single-wavelength V band magnitude.
                                        #unred_flux_V_mag = -2.5*np.log10(np.average(unred_flux[Vband_mask]))
                                        #unred_mag_norm = unred_mag - unred_flux_V_mag

                                        unred_mag_norm = unred_mag - unred_flux_avg_mag  #unred_V_mag
    
                                        # this is unreddened 12cu mag - pristine 11fe mag
                                        delta = unred_mag_norm[mask]-ref_mag_norm[mask]
                                        tmp_wave = ref_wave[mask]
                                        # convert to vector from array and filter nan-values
                                        delta = np.matrix(delta[nanmask])
                                        
                                        #delta_array = np.squeeze(np.asarray(delta))  # converting 1D matrix to 1D array.
                                        ## ----->I shoudl fix ylim<-------------------
                                        #fig = plt.figure()
                                        #plt.plot(tmp_wave[nanmask], delta_array, 'ro')
                                       
                                        
                                        # The original equation is delta.T * C_inv * delta, but delta
                                        #  is already a row vector in numpy so it is the other way around.
                                        CHI2[k,j] = (delta * TMP_C_total_inv * delta.T)[0,0]
                                        
                        
                        #plt.show()
                        
                        
                        #################################################
                        
                        ### report/save results
                        
                        chi2_min = np.min(CHI2)
                        log( "min chi2: {}".format(chi2_min) )
                        
                        mindex = np.where(CHI2==chi2_min)
                        mx, my = mindex[1][0], mindex[0][0]
                        
                        log( "\t {}".format(mindex) )
                        log( "\t RV={} EBV={} AV={}".format(y[my], x[mx], x[mx]*y[my]) )
                        
                        chi2s.append(CHI2)
                        chi2_reductions.append(CHI2_reduction)
                        min_chi2s.append(chi2_min)
                        best_rvs.append(y[my])
                        best_ebvs.append(x[mx])
                        best_avs.append(x[mx]*y[my])

                
                pprint( zip(phases, best_rvs, best_ebvs, best_avs, min_chi2s) )
                
                # save results
                #                filename = "spectra_mag_fit_results_{}.pkl".format(strftime("%H-%M-%S-%m-%d-%Y", gmtime()))

                filename = "spectra_mag_fit_results_FILTERED.pkl"
                cPickle.dump({'phases': phases, 'rv': best_rvs, 'ebv': best_ebvs, 'av': best_avs,
                                'chi2': chi2s, 'chi2_reductions': chi2_reductions, 'steps': steps,
                                'x': x, 'y': y, 'X': X, 'Y': Y},
                                open(filename, 'wb'))
                
                log( "Results successfully saved in: {}".format(filename) )

                
                
        perphase_fit()
        

if __name__=="__main__":
        main()
