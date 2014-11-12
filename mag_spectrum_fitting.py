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
from time import gmtime, strftime

from scipy.interpolate import interp1d



def main():
        
        ebv_guess = 1.05
        ebv_pad = 0.5
        
        rv_guess = 2.7
        rv_pad = 0.8
        
        steps = 25
        
        FEATURES_ACTUAL = [(3425, 3820, 'CaII'), (3900, 4100, 'SiII'), (5640, 5900, 'SiII'),
                                (6000, 6280, 'SiII'), (8000, 8550, 'CaII')]
        
        pristine_12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)[:12]
        phases = [t[0] for t in pristine_12cu]
        
        pristine_11fe = l.interpolate_spectra(phases, l.get_11fe(loadmast=False, loadptf=False))
        
        
        ########################
        # Choose 'reddened' to be either 11fe interpolated to the phases of 12cu,
        # or 12cu itself.
        #
        #reddened = l.interpolate_spectra(phases, l.get_11fe('fm', ebv=-0.4, rv=2.45, loadmast=False, loadptf=False))
        reddened = pristine_12cu
        #
        ########################
        
        
        ########################
        ### helper functions ###
        
        def filter_features(features, wave):
                intersection = np.array([False]*wave.shape[0])
                for feature in features:
                        intersection |= ((wave>feature[0])&(wave<feature[1]))
                
                return ~intersection
        
        
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
                
                print "SIZE OF GRID:"
                print steps
                print "EBV SEARCH GRID:"
                print tmpx
                print "RV SEARCH GRID:"
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
                        
                        # pristine_11fe
                        ref_wave = ref[1].wave
                        ref_flux = ref[1].flux
                        ref_flux_avg = np.average(ref_flux)
                        ref_flux_avg_mag = -2.5*np.log10(ref_flux_avg)
                        ref_var = ref[1].error
                        ref_calibration_error = ref[2]
                        
                        ref_mag, ref_mag_var, ref_calibration_error_mag \
                        = flux2mag(ref_flux, ref_var, ref_calibration_error)
                        
                        ref_mag_norm = ref_mag - ref_flux_avg_mag
                        
                        # 12cu/reddened 11fe
                        red_wave = red[1].wave
                        red_interp_flux = interp1d(red_wave, red[1].flux)(ref_wave)
                        
                        ## test ######################################################################
                        #red_noisy_flux = (1 + 0.05*np.random.randn(red[1].flux.shape[0]) )*red[1].flux
                        #red_interp_flux = interp1d(red_wave, red_noisy_flux)(ref_wave)
                        ##############################################################################
                        
                        red_interp_var  = interp1d(red_wave, red[1].error)(ref_wave)
                        red_calibration_error = red[2]
                        
                        red_interp_mag, red_interp_mag_var, red_calibration_error_mag \
                        = flux2mag(red_interp_flux, red_interp_var, red_calibration_error)
                        
                        # calculate matrices
                        V_ref = np.diag(ref_mag_var[mask])
                        S_ref = (ref_calibration_error_mag**2)*np.ones(V_ref.shape)
                        C_ref = V_ref + S_ref
                        
                        V_red = np.diag(red_interp_mag_var[mask])
                        S_red = (red_calibration_error_mag**2)*np.ones(V_red.shape)
                        C_red = V_red + S_red
                        
                        #########################
                        
                        def print_diagnostics():
                                print "ref_flux:"
                                print ref_flux
                                print "ref_mag_var:"
                                print ref_mag_var
                                print "ref_calibration_error_mag:"
                                print ref_calibration_error_mag
                                print "V_ref:"
                                print V_ref
                                print "S_ref:"
                                print S_ref
                                print "C_ref:"
                                print C_ref
                                print "red_interp_mag_var:"
                                print red_interp_mag_var
                                print "red_calibration_error_mag:"
                                print red_calibration_error_mag
                                print "V_red:"
                                print V_red
                                print "S_red:"
                                print S_red
                                print "C_red:"
                                print C_red
                        
                        #print_diagnostics()
                        
                        #########################
                        # INVERT TOTAL COVARIANCE MATRIX
                        
                        C_total = C_ref + C_red
                        
                        print "Computing inverse..."
                        C_total_inv = np.matrix(np.linalg.inv(C_total))
                        
                        #########################
                        
                        
                        x = np.linspace(ebv_guess-ebv_pad, ebv_guess+ebv_pad, steps)
                        y = np.linspace(rv_guess-rv_pad, rv_guess+rv_pad, steps)
                        
                        X, Y = np.meshgrid(x, y)
                        CHI2 = np.zeros( X.shape )
                        
                        print "# Points with negative flux:"
                        print np.sum(np.isnan(np.log10(red_interp_flux)))
                        
                        print "Scanning CHI2 grid..."
                        for j, EBV in enumerate(x):
                                for k, RV in enumerate(y):
                                        
                                        unred_flux = redden_fm(ref_wave, red_interp_flux, EBV, RV)
                                        unred_mag = flux2mag(unred_flux)
                                        
                                        # normalization
                                        unred_flux_avg_mag = -2.5*np.log10(np.average(unred_flux))
                                        unred_mag_norm = unred_mag - unred_flux_avg_mag
                                        
                                        delta = unred_mag_norm[mask]-ref_mag_norm[mask]
                                        
                                        
                                        # hack thrown together to filter nan-values
                                        nanmask = ~np.isnan(delta)
                                        delta = np.matrix(delta[nanmask])
                                        nanmatrix = np.outer(nanmask, nanmask)
                                        TMP_C_total_inv = np.matrix(C_total_inv[nanmatrix].reshape(np.sum(nanmask), np.sum(nanmask)))
                                        
                                        # find chi2 per degree of freedom
                                        CHI2_reduction = delta.shape[1] - 2  # (num. data points)-(num. parameters)
                                        CHI2[k,j] = (delta * TMP_C_total_inv * delta.T)[0,0] / CHI2_reduction
                                        
                                        
                        ### report/save results
                        
                        chi2s.append(CHI2)
                        
                        chi2_min = np.min(CHI2)
                        print "min chi2:", chi2_min
                        min_chi2s.append(chi2_min)
                        
                        mindex = np.where(CHI2==chi2_min)
                        mx, my = mindex[1][0], mindex[0][0]
                        
                        print "\t", mindex
                        print "\t RV={} EBV={} AV={}".format(y[my], x[mx], x[mx]*y[my])
                        
                        best_rvs.append(y[my])
                        best_ebvs.append(x[mx])
                        best_avs.append(x[mx]*y[my])

                
                pprint( zip(phases, best_rvs, best_ebvs, best_avs, min_chi2s) )
                
                # save results
                ftime = strftime("%H-%M-%S-%m-%d-%Y", gmtime())
                cPickle.dump({'phases': phases, 'rv': best_rvs, 'ebv': best_ebvs, 'av': best_avs,
                                'chi2': chi2s, 'x': x, 'y': y, 'X': X, 'Y': Y},
                                open("spectra_mag_fit_results_{}.pkl".format(ftime), 'wb'))
                

                
                
        perphase_fit()
        

if __name__=="__main__":
        main()
