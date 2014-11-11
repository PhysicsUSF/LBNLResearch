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

from scipy.interpolate import interp1d



def main():
        TITLE_FONTSIZE = 28
        AXIS_LABEL_FONTSIZE = 20
        TICK_LABEL_FONTSIZE = 16
        INPLOT_LEGEND_FONTSIZE = 20
        LEGEND_FONTSIZE = 18
        
        
        ebv_guess = 1.05
        ebv_pad = 0.3
        
        rv_guess = 2.7
        rv_pad = 0.5
        
        steps = 25
        
        FEATURES_ACTUAL = [(3425, 3820, 'CaII'), (3900, 4100, 'SiII'), (5640, 5900, 'SiII'),
                                (6000, 6280, 'SiII'), (8000, 8550, 'CaII')]
        
        pristine_12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)[:12]
        phases = [t[0] for t in pristine_12cu]
        
        pristine_11fe = l.interpolate_spectra(phases, l.get_11fe(loadmast=False, loadptf=False))
        
        reddened = l.interpolate_spectra(phases, l.get_11fe('fm', ebv=-1.05, rv=2.7, loadmast=False, loadptf=False))
        #reddened = pristine_12cu
        
        
        def filter_features(features, wave):
                intersection = np.array([False]*wave.shape[0])
                for feature in features:
                        intersection |= ((wave>feature[0])&(wave<feature[1]))
                
                return ~intersection
        
        
        def allphase_fit():
                
                ref_fluxs_allphases = np.concatenate( tuple([t[1].flux for t in pristine_11fe]) )
                MASK = np.concatenate( tuple([filter_features(FEATURES, t[1].wave) for t in pristine_11fe]) )
                
                x = np.linspace(ebv_guess-ebv_pad, ebv_guess+ebv_pad, steps)
                y = np.linspace(rv_guess-rv_pad, rv_guess+rv_pad, steps)
                
                X, Y = np.meshgrid(x, y)
                SSR = np.zeros( X.shape )
                
                for j, EBV in enumerate(x):
                        print j
                        for k, RV in enumerate(y):
                                
                                reddened_allphases = []
                                for p, t in enumerate(reddened):
                                        interpolated_red = interp1d(t[1].wave, t[1].flux)(pristine_11fe[p][1].wave)
                                        unred_flux = redden_fm(pristine_11fe[p][1].wave, interpolated_red, EBV, RV)
                                        ref_flux = pristine_11fe[p][1].flux
                                        
                                        unred_norm_factor = np.average(unred_flux)
                                        ref_norm_factor = np.average(ref_flux)
                                        unred_flux /= (unred_norm_factor/ref_norm_factor)
                                        
                                        reddened_allphases.append(unred_flux)
                                
                                reddened_allphases = np.concatenate( tuple(reddened_allphases) )
                                
                                SSR[k,j] = np.sum((reddened_allphases[MASK]-ref_fluxs_allphases[MASK])**2)
                
                
                ssr_min = np.min(SSR)
                
                print SSR
                
                mindex = np.where(SSR==ssr_min)
                mx, my = mindex[1][0], mindex[0][0]
                
                print mindex
                print x[mx], y[my]
        
        
        def flux2mag(flux, err, calibration_err):
                mag = -2.5*np.log10(flux)
                
                fr_err = np.sqrt(err)/flux
                mag_err = (1.0857362*fr_err)**2  # 2.5/np.log(10) = 1.0857362
                
                calibration_err_mag = 1.0857362*calibration_err
                
                return mag, mag_err, calibration_err_mag
        
        
        def perphase_fit():
                
                tmpx = np.linspace(ebv_guess-ebv_pad, ebv_guess+ebv_pad, steps)
                tmpy = np.linspace(rv_guess-rv_pad, rv_guess+rv_pad, steps)
                
                print "SIZE OF GRID:"
                print steps
                print "EBV SEARCH:"
                print tmpx
                print "RV SEARCH:"
                print tmpy
                
                best_rvs = []
                best_ebvs = []
                best_avs = []
                chi2s = []
                min_chi2s = []
                
                for phase_index in xrange(len(phases)):
                        
                        ref = pristine_11fe[phase_index]
                        red = reddened[phase_index]
                        
                        # mask for spectral features not included in fit
                        mask = filter_features(FEATURES_ACTUAL, ref[1].wave)
                        
                        print
                        print ref[0], red[0]
                        
                        
                        ### COVARIANCE MATRIX ###
                        ref_wave = ref[1].wave
                        ref_flux = ref[1].flux
                        ref_error = ref[1].error
                        ref_calibration_error = ref[2]
                        
                        ref_mag, ref_mag_error, ref_calibration_error_mag \
                        = flux2mag(ref_flux, ref_error, ref_calibration_error)
                        
                        V_ref = np.diag(ref_mag_error[mask])
                        S_ref = (ref_calibration_error_mag**2)*np.outer(ref_mag[mask], ref_mag[mask])
                        C_ref = V_ref + S_ref
                        
                        # interpolate reddened spectrum/error to match wavelengths of ref spectrum 
                        red_interp_flux = interp1d(ref_wave, red[1].flux)(ref_wave)
                        
                        ##### ARTIFICIALLY SCALE FLUX TO SIMULATE DISTANCE DIFFERENCE ###
                        red_interp_flux = interp1d(ref_wave, 1.0*red[1].flux)(ref_wave)
                        #################################################################
                        
                        red_interp_error  = interp1d(ref_wave, red[1].error)(ref_wave)
                        red_calibration_error = red[2]
                        
                        red_interp_mag, red_interp_mag_error, red_calibration_error_mag \
                        = flux2mag(red_interp_flux, red_interp_error, red_calibration_error)
                        
                        V_red = np.diag(red_interp_mag_error[mask])
                        S_red = (red_calibration_error_mag**2)*np.outer(red_interp_mag[mask], red_interp_mag[mask])
                        C_red = V_red + S_red
                        
                        C_total = C_ref + C_red
                        
                        print "Computing inverse..."
                        C_total_inv = np.matrix(np.linalg.inv(C_total))
                        #########################
                        
                        
                        x = np.linspace(ebv_guess-ebv_pad, ebv_guess+ebv_pad, steps)
                        y = np.linspace(rv_guess-rv_pad, rv_guess+rv_pad, steps)
                        
                        X, Y = np.meshgrid(x, y)
                        CHI2 = np.zeros( X.shape )
                        
                        CHI2_reduction = np.sum(mask) - 2  # (num. data points)-(num. parameters)
                        
                        print "Scanning CHI2 grid..."
                        for j, EBV in enumerate(x):
                                for k, RV in enumerate(y):
                                        
                                        unred_flux = redden_fm(ref_wave, red_interp_flux, EBV, RV)
                                        unred_mag = flux2mag(unred_flux, red_interp_error, red_calibration_error)[0]
                                        
                                        ### normalization ###
                                        unred_avg = np.average(unred_mag[mask])
                                        ref_avg = np.average(ref_mag[mask])
                                        unred_mag += (ref_avg - unred_avg)
                                        #####################
                                        
                                        delta = np.matrix(unred_mag[mask]-ref_mag[mask])
                                        CHI2[k,j] = (delta * C_total_inv * delta.T)[0,0] / CHI2_reduction
                        
                        #print "CHI2 RESULT:"
                        #print CHI2
                        
                        chi2s.append(CHI2)
                        
                        chi2_min = np.min(CHI2)
                        print "min chi2:", chi2_min
                        min_chi2s.append(chi2_min)
                        
                        mindex = np.where(CHI2==chi2_min)
                        mx, my = mindex[1][0], mindex[0][0]
                        
                        print "\t", mindex
                        print "\t RV={} EBV={}".format(y[my], x[mx])
                        
                        best_rvs.append(y[my])
                        best_ebvs.append(x[mx])
                        best_avs.append(x[mx]*y[my])

                
                pprint( zip(phases, best_rvs, best_ebvs, min_chi2s) )
                
                # save results
                cPickle.dump({'phases': phases, 'rv': best_rvs, 'ebv': best_ebvs, 'av': best_avs,
                                'chi2': chi2s, 'x': x, 'y': y, 'X': X, 'Y': Y},
                                open("spectra_fit_results_{}iter.pkl".format(steps), 'wb'))
                

                
                
        perphase_fit()
        #allphase_fit()
        

if __name__=="__main__":
        main()
