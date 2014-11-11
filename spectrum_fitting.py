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



def export_filtered_spectra():
        pristine_12cu = l.get_12cu()[:12]

        phases = [t[0] for t in pristine_12cu]

        pristine_11fe = l.interpolate_spectra(phases, l.get_11fe())

        FEATURES = [(3425, 3820), (3900, 4100), (5640, 5900), (6000, 6280), (8000, 8550)]

        def filter_features(features, spectrum):
                wave = spectrum.wave
                flux = spectrum.flux
                
                intersection = np.array([False]*wave.shape[0])
                for feature in features:
                        intersection|=( (wave>feature[0])&(wave<feature[1]) )
                
                return ~intersection


        valid = filter_features(FEATURES, pristine_11fe[0][1])

        filtered_11fe = []
        filtered_12cu = []
        data_dict = {'SN2012cu':[], 'SN2011fe':[]}
        for i in xrange(len(phases)):
                valid_11fe = filter_features(FEATURES, pristine_11fe[i][1])
                valid_12cu = filter_features(FEATURES, pristine_12cu[i][1])
                
                filtered_11fe.append( (phases[i],
                                        pristine_11fe[i][1].wave[valid_11fe],
                                        pristine_11fe[i][1].flux[valid_11fe]
                                        )
                )
                
                filtered_12cu.append( (phases[i],
                                        pristine_12cu[i][1].wave[valid_12cu],
                                        pristine_12cu[i][1].flux[valid_12cu]
                                        )
                )
                
                data_dict['SN2011fe'].append(  {'phase': phases[i],
                                        'pristine_11fe_wave': pristine_11fe[i][1].wave,
                                        'pristine_11fe_flux': pristine_11fe[i][1].flux,
                                        'spectrum_filter': valid_11fe
                                        }
                )
                
                data_dict['SN2012cu'].append(  {'phase': phases[i],
                                        'pristine_12cu_wave': pristine_12cu[i][1].wave,
                                        'pristine_12cu_flux': pristine_12cu[i][1].flux,
                                        'spectrum_filter': valid_12cu
                                        }
                )
                
        cPickle.dump(filtered_11fe, open("filtered_sn2011fe.pkl", 'wb'))
        cPickle.dump(filtered_12cu, open("filtered_SN2012cu.pkl", 'wb'))
        cPickle.dump(data_dict, open("sn11fe_sn12cu_data.pkl", 'wb'))




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
        
        steps = 50
        
        FEATURES_ACTUAL = [(3425, 3820, 'CaII'), (3900, 4100, 'SiII'), (5640, 5900, 'SiII'),
                                (6000, 6280, 'SiII'), (8000, 8550, 'CaII')]
        
        FEATURES = [(3280, 3820, 'CaII'), (3900, 4100, 'SiII'), (5640, 5900, 'SiII'),
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
                
                fig = plt.figure()
                ax = fig.gca()
                
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
                        V_ref = np.diag(((ref[1].error))[mask])
                        calibration_error_ref = ref[2]
                        S_ref = (calibration_error_ref**2)*np.outer((ref[1].flux)[mask], (ref[1].flux)[mask])
                        C_ref = V_ref + S_ref
                        
                        # interpolate reddened spectrum/error to match wavelengths of ref spectrum 
                        red_interp_flux = interp1d(red[1].wave, red[1].flux)(ref[1].wave)
                        red_interp_err  = interp1d(red[1].wave, red[1].error)(ref[1].wave)
                        
                        V_red = np.diag((red_interp_err)[mask])
                        calibration_error_red = red[2]
                        S_red = (calibration_error_red**2)*np.outer(red_interp_flux[mask], red_interp_flux[mask])
                        C_red = V_red + S_red
                        
                        C_total = C_ref + C_red
                        
                        print "Computing inverse..."
                        C_total_inv = np.matrix(np.linalg.inv(C_total))
                        #########################
                        
                        
                        x = np.linspace(ebv_guess-ebv_pad, ebv_guess+ebv_pad, steps)
                        y = np.linspace(rv_guess-rv_pad, rv_guess+rv_pad, steps)
                        
                        X, Y = np.meshgrid(x, y)
                        CHI2 = np.zeros( X.shape )
                        
                        CHI2_reduction = np.sum(mask) - 2  # (num. data points) - (num. parameters)
                        
                        print "Scanning CHI2 grid..."
                        for j, EBV in enumerate(x):
                                for k, RV in enumerate(y):
                                        
                                        unred_flux = redden_fm(ref[1].wave, red_interp_flux, EBV, RV)
                                        ref_flux = deepcopy(ref[1].flux)
                                        
                                        ### normalization ###
                                        unred_norm_factor = np.average(unred_flux)
                                        ref_norm_factor = np.average(ref_flux)
                                        unred_flux /= (unred_norm_factor/ref_norm_factor)
                                        #####################
                                        
                                        delta = np.matrix(unred_flux[mask]-ref_flux[mask])
                                        CHI2[k,j] = (delta * C_total_inv * delta.T)[0,0] / CHI2_reduction
                        
                        print "CHI2 RESULT:"
                        print CHI2
                        
                        chi2s.append(CHI2)
                        
                        chi2_min = np.min(CHI2)
                        print "min chi2:", chi2_min
                        min_chi2s.append(chi2_min)
                        
                        mindex = np.where(CHI2==chi2_min)
                        mx, my = mindex[1][0], mindex[0][0]
                        
                        print "\t", mindex
                        print "\t", y[my], x[mx]
                        
                        best_rvs.append(y[my])
                        best_ebvs.append(x[mx])
                        best_avs.append(x[mx]*y[my])
                        
                        #### plot best fit ###
                        
                        interpolated_red = interp1d(red[1].wave, red[1].flux)(ref[1].wave)
                        
                        unred_flux = redden_fm(ref[1].wave, interpolated_red, x[mx], y[my])
                        
                        ref_flux = ref[1].flux
                        
                        unred_norm_factor = np.average(unred_flux)
                        ref_norm_factor = np.average(ref_flux)
                        unred_flux /= (unred_norm_factor/ref_norm_factor)
                        
                        adjust = phase_index
                        
                        
                        # plot unfiltered spectrum used in fit
                        unred_flux_wave = ref[1].wave[mask]
                        unred_flux_log = np.log(unred_flux[mask])
                        unred_flux_log_nanmask = ~np.isnan(unred_flux_log)
                        unred_flux_log = unred_flux_log[unred_flux_log_nanmask]
                        unred_flux_wave = unred_flux_wave[unred_flux_log_nanmask]
                        
                        ref_flux_wave = ref[1].wave[mask]
                        ref_flux_log = np.log(ref_flux[mask])
                        ref_flux_log_nanmask = ~np.isnan(ref_flux_log)
                        ref_flux_log = ref_flux_log[ref_flux_log_nanmask]
                        ref_flux_wave = ref_flux_wave[ref_flux_log_nanmask]
                        
                        plt.plot(unred_flux_wave, unred_flux_log-adjust, 'r.', ms=3)
                        plt.plot(ref_flux_wave, ref_flux_log-adjust, 'g.', ms=3)
                        
                        
                        # plot blocked out region of spectrum
                        unred_flux_wave = ref[1].wave[~mask]
                        unred_flux_log = np.log(unred_flux[~mask])
                        unred_flux_log_nanmask = ~np.isnan(unred_flux_log)
                        unred_flux_log = unred_flux_log[unred_flux_log_nanmask]
                        unred_flux_wave = unred_flux_wave[unred_flux_log_nanmask]
                        
                        ref_flux_wave = ref[1].wave[~mask]
                        ref_flux_log = np.log(ref_flux[~mask])
                        ref_flux_log_nanmask = ~np.isnan(ref_flux_log)
                        ref_flux_log = ref_flux_log[ref_flux_log_nanmask]
                        ref_flux_wave = ref_flux_wave[ref_flux_log_nanmask]
                        
                        plt.plot(unred_flux_wave, unred_flux_log-adjust, 'k.', ms=2)
                        plt.plot(ref_flux_wave, ref_flux_log-adjust, 'k.', ms=2)
                        
                        # plot phase label
                        ty = (ref_flux_log-adjust)[0]
                        if phase_index==0:
                                plt.text(3250, ty, "Phase:\n{}".format(ref[0]), horizontalalignment='right',
                                         color='k', fontsize=INPLOT_LEGEND_FONTSIZE)
                        else:
                                plt.text(3250, ty, str(ref[0]), horizontalalignment='right',
                                         color='k', fontsize=INPLOT_LEGEND_FONTSIZE)
                
                
                pprint( zip(phases, best_rvs, best_ebvs, min_chi2s) )
                
                # save results
                cPickle.dump({'phases': phases, 'rv': best_rvs, 'ebv': best_ebvs, 'av': best_avs,
                                'chi2': chi2s, 'x': x, 'y': y, 'X': X, 'Y': Y},
                                open("spectra_fit_results_{}iter.pkl".format(steps), 'wb'))
                
                
                # gray bands at blocked out features
                for feature in FEATURES_ACTUAL:
                        plt.axvspan(feature[0], feature[1], facecolor='k', alpha=0.1)
                        plt.text((feature[0]+feature[1])/2, -27.5, '${}$'.format(feature[2]),
                                horizontalalignment='center',
                                #transform=ax.transAxes,
                                fontsize=AXIS_LABEL_FONTSIZE)
                
                # plot table with rv, ebv, and av values
                r = lambda n: round(n, 3)
                values = [phases, map(r,best_rvs), map(r,best_ebvs), map(r,best_avs)]
                alpha = 1 - 0.1
                cellColours = np.array([[[alpha,alpha,alpha,1]]*12] + [[[1,1,1,1]]*12]*3 )
                rowColours = np.array([[alpha,alpha,alpha,1]] + [[1,1,1,1]]*3 )
                
                table = plt.table(cellText=values,
                                  rowLabels=['PHASE', '$R_V$', '$E(B-V)$', '$A_V$'],
                                  cellColours=cellColours,
                                  rowColours=rowColours,
                                  loc='bottom')
                
                table.set_fontsize(14)                  
                
                
                ### format plot ###
                
                # y-axis
                plt.yticks([])
                plt.ylabel('$log(f_{\lambda})$ (adjusted)', fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
                plt.ylim(-45,-27)
                
                # x-axis
                plt.xlim(2800,10000)
                labels = ax.get_xticks().tolist()
                labels = map(lambda x: int(x), labels)
                labels[-1] = ''
                ax.set_xticklabels(labels)
                ax.get_xaxis().set_tick_params(direction='in', pad=-20)
                plt.setp(ax.get_xticklabels(), fontsize=TICK_LABEL_FONTSIZE)
                ax.text(.5,.045,'Wavelength ($\AA$)',
                        horizontalalignment='center',
                        transform=ax.transAxes,
                        fontsize=AXIS_LABEL_FONTSIZE)
                
                # legend
                plt.plot([], [], 'r.', ms=18, label='SN2012cu')
                plt.plot([], [], 'g.', ms=18, label='SN2011fe')
                plt.legend(prop={'size':LEGEND_FONTSIZE})
                
                plt.subplots_adjust(left=0.07, bottom=0.13, right=0.95, top=0.92)
                
                plt.title('Spectrum Comparison: Unreddened SN2012cu vs. SN2011fe', fontsize=TITLE_FONTSIZE)
                plt.show()
                
                
                
                
        perphase_fit()
        #allphase_fit()
        

if __name__=="__main__":
        main()
