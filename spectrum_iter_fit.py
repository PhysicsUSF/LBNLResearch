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
        ebv_guess = 1.05
        ebv_pad = 0.5
        
        rv_guess = 2.7
        rv_pad = 1.0
        
        steps = 15
        
        FEATURES_ACTUAL = [(3425, 3820, 'CaII'), (3900, 4100, 'SiII'), (5640, 5900, 'SiII'),
                                (6000, 6280, 'SiII'), (8000, 8550, 'CaII')]
        
        pristine_12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)[:12]
        phases = [t[0] for t in pristine_12cu]
        
        pristine_11fe = l.interpolate_spectra(phases, l.get_11fe(loadmast=False, loadptf=False))
        
        #reddened = l.interpolate_spectra(phases, l.get_11fe('fm', ebv=-1.05, rv=2.7, loadmast=False, loadptf=False))
        reddened = pristine_12cu
        
        
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
        
        
        
        def perphase_fit():
                
                best_rvs = []
                best_ebvs = []
                best_avs = []
                chi2s = []
                min_chi2s = []
                
                for phase_index in xrange(len(phases)):
                        ref = pristine_11fe[phase_index]
                        ref_flux = ref[1].flux
                        red = reddened[phase_index]
                        
                        # mask for spectral features not included in fit
                        mask = filter_features(FEATURES_ACTUAL, ref[1].wave)
                        
                        print
                        print ref[0], red[0]
                        
                        
                        # init values for iterative fit
                        BEST_RV  = 2.75
                        BEST_EBV = 1.05
                        
                        NEW_RV_PAD = rv_pad
                        NEW_EBV_PAD = ebv_pad
                        
                        # interpolate reddened spectrum/error to match wavelengths of ref spectrum 
                        red_interp_flux = interp1d(red[1].wave, red[1].flux)(ref[1].wave)
                        red_interp_err  = interp1d(red[1].wave, red[1].error)(ref[1].wave)
                        
                        unred_interp_flux = red_interp_flux
                        
                        for itr in xrange(4):
                                
                                print itr, BEST_RV, NEW_RV_PAD, BEST_EBV, NEW_EBV_PAD
                                
                                ### COVARIANCE MATRIX ###
                                V_ref = np.diag((ref[1].error)[mask])
                                calibration_error_ref = ref[2]
                                S_ref = (calibration_error_ref**2)*np.outer((ref_flux)[mask], (ref_flux)[mask])
                                C_ref = V_ref + S_ref
                                
                                V_red = np.diag((red_interp_err)[mask])
                                calibration_error_red = red[2]
                                S_red = (calibration_error_red**2)*np.outer(unred_interp_flux[mask], unred_interp_flux[mask])
                                C_red = V_red + S_red
                                
                                C_total = C_ref + C_red
                                
                                print "Computing inverse..."
                                C_total_inv = np.matrix(np.linalg.inv(C_total))
                                #########################
                                
                                x = np.linspace(BEST_EBV-NEW_EBV_PAD, BEST_EBV+NEW_EBV_PAD, steps)
                                y = np.linspace(BEST_RV-NEW_RV_PAD, BEST_RV+NEW_RV_PAD, steps)
                                
                                X, Y = np.meshgrid(x, y)
                                CHI2 = np.zeros( X.shape )
                                
                                CHI2_reduction = np.sum(mask) - 2  # (num. data points) - (num. parameters)
                                
                                print "Scanning CHI2 grid..."
                                for j, EBV in enumerate(x):
                                        for k, RV in enumerate(y):
                                                
                                                unred_flux = redden_fm(ref[1].wave, red_interp_flux, EBV, RV)
                                                
                                                unred_flux /= (np.average(unred_flux[mask])/np.average(ref_flux[mask]))
                                                
                                                delta = np.matrix(unred_flux[mask]-ref_flux[mask])
                                                
                                                CHI2[k,j] = (delta * C_total_inv * delta.T)[0,0] / CHI2_reduction
                                
                                
                                print "CHI2 RESULTS:"
                                print CHI2
                                
                                chi2_min = np.min(CHI2)
                                print "min chi2:", chi2_min
                                
                                mindex = np.where(CHI2==chi2_min)
                                mx, my = mindex[1][0], mindex[0][0]
                                
                                print "\t", mindex
                                print "\t", y[my], x[mx]
                                
                                BEST_RV = y[my]
                                BEST_EBV = x[mx]
                                NEW_RV_PAD *= 0.5
                                NEW_EBV_PAD *= 0.5
                                #unred_interp_flux = redden_fm(ref[1].wave, red_interp_flux, BEST_EBV, BEST_RV)
                                #unred_interp_flux /= (np.average(unred_interp_flux[mask])/np.average(ref_flux[mask]))
                                
                        
                        
                        best_rvs.append(BEST_RV)
                        best_ebvs.append(BEST_EBV)
                        best_avs.append(BEST_EBV*BEST_RV)
                        chi2s.append(CHI2)
                        min_chi2s.append(chi2_min)
                        

                pprint( zip(phases, best_rvs, best_ebvs, min_chi2s) )
                
                # save results
                cPickle.dump({'phases': phases, 'rv': best_rvs, 'ebv': best_ebvs, 'av': best_avs,
                                'chi2': chi2s, 'x': x, 'y': y, 'X': X, 'Y': Y},
                                open("spectra_iter_fit_results_{}iter.pkl".format(steps), 'wb'))
                

                
                
        perphase_fit()
        #allphase_fit()
        

if __name__=="__main__":
        main()
