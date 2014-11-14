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


def main(title, info_dict, FEATURES_ACTUAL):
        TITLE_FONTSIZE = 18
        AXIS_LABEL_FONTSIZE = 20
        TICK_LABEL_FONTSIZE = 16
        INPLOT_LEGEND_FONTSIZE = 20
        LEGEND_FONTSIZE = 18
        
        
        # config
        ebv_guess = 1.05
        ebv_pad = 0.25
        
        rv_guess = 3.0
        rv_pad = 0.6
        
        steps = 120
        
        
        
        
        
        pristine_12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)[:12]
        phases = [t[0] for t in pristine_12cu]
        
        pristine_11fe = l.interpolate_spectra(phases, l.get_11fe(loadmast=False, loadptf=False))
        
        reddened = pristine_12cu
        
        
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
                        
                        CHI2 = info_dict['chi2'][phase_index]
                        CHI2_reduction = info_dict['chi2_reductions'][phase_index]
                        x = info_dict['x']
                        y = info_dict['y']
                        
                        CHI2 = CHI2/CHI2_reduction
                        chi2_min = np.min(CHI2)
                        
                        
                        mindex = np.where(CHI2==chi2_min)
                        mx, my = mindex[1][0], mindex[0][0]
                        
                        print "\t", mindex
                        print "\t", y[my], x[mx]
                        
                        best_rvs.append(y[my])
                        best_ebvs.append(x[mx])
                        best_avs.append(x[mx]*y[my])
                        
                        
                        
                        #### plot best fit ###
                        
                        mask = filter_features(FEATURES_ACTUAL, ref[1].wave)
                        
                        interpolated_red = interp1d(red[1].wave, red[1].flux)(ref[1].wave)
                        unred_flux = redden_fm(ref[1].wave, interpolated_red, x[mx], y[my])
                        
                        
                        # normalization
                        unred_flux_avg_mag = 2.5*np.log10(np.average(unred_flux))
                        unred_mag = 2.5*np.log10(unred_flux)
                        unred_mag_norm = unred_mag - unred_flux_avg_mag
                        
                        ref_flux_avg_mag = 2.5*np.log10(np.average(ref[1].flux))
                        ref_mag = 2.5*np.log10(ref[1].flux)
                        ref_mag_norm = ref_mag - ref_flux_avg_mag
                        
                        adjust = 1.3*phase_index
                        
                        
                        
                        # plot unfiltered spectrum used in fit
                        mag_wave = (ref[1].wave)[mask]
                        
                        unred_nanmask = ~np.isnan(unred_mag_norm[mask])
                        unred_mag_norm_u = unred_mag_norm[mask][unred_nanmask]
                        unred_mag_wave_u = mag_wave[unred_nanmask]
                        
                        ref_nanmask = ~np.isnan(ref_mag_norm[mask])
                        ref_mag_norm_u = ref_mag_norm[mask][ref_nanmask]
                        ref_mag_wave_u = mag_wave[ref_nanmask]
                        
                        plt.plot(unred_mag_wave_u, unred_mag_norm_u-adjust, 'r.', ms=4)
                        plt.plot(ref_mag_wave_u, ref_mag_norm_u-adjust, 'g.', ms=4)
                        
                        
                        # plot blocked out region of spectrum
                        mag_wave = (ref[1].wave)[~mask]
                        
                        unred_nanmask = ~np.isnan(unred_mag_norm[~mask])
                        unred_mag_norm_f = unred_mag_norm[~mask][unred_nanmask]
                        unred_mag_wave_f = mag_wave[unred_nanmask]
                        
                        ref_nanmask = ~np.isnan(ref_mag_norm[~mask])
                        ref_mag_norm_f = ref_mag_norm[~mask][ref_nanmask]
                        ref_mag_wave_f = mag_wave[ref_nanmask]
                        
                        plt.plot(unred_mag_wave_f, unred_mag_norm_f-adjust, 'k.', ms=2)
                        plt.plot(ref_mag_wave_f, ref_mag_norm_f-adjust, 'k.', ms=2)
                        
                        
                        # plot phase label
                        #ty = (ref_mag_norm_u-adjust)[0]
                        ty = (ref_mag_norm-adjust)[0]
                        if phase_index==0:
                                plt.text(3250, ty, "Phase:\n{}".format(ref[0]), horizontalalignment='right',
                                         color='k', fontsize=INPLOT_LEGEND_FONTSIZE)
                        else:
                                plt.text(3250, ty, str(ref[0]), horizontalalignment='right',
                                         color='k', fontsize=INPLOT_LEGEND_FONTSIZE)
                
                
                
                # gray bands at blocked out features
                for feature in FEATURES_ACTUAL:
                        plt.axvspan(feature[0], feature[1], facecolor='k', alpha=0.1)
                        plt.text((feature[0]+feature[1])/2, 1.9, '${}$'.format(feature[2]),
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
                plt.ylabel('$-mag_{\lambda}$ (adjusted)', fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
                plt.ylim(-17.5, 2.5)
                
                # x-axis
                plt.xlim(2800,10000)
                labels = ax.get_xticks().tolist()
                labels = map(int, labels)
                labels[-1] = ''
                ax.set_xticklabels(labels)
                ax.get_xaxis().set_tick_params(direction='in', pad=-20)
                plt.setp(ax.get_xticklabels(), fontsize=TICK_LABEL_FONTSIZE)
                ax.text(.5,.045,'Wavelength ($\AA$)',
                        horizontalalignment='center',
                        transform=ax.transAxes,
                        fontsize=AXIS_LABEL_FONTSIZE)
                
                # legend
                plt.plot([], [], 'r.', ms=18, label='SN2012cu (Unreddened)')
                plt.plot([], [], 'g.', ms=18, label='SN2011fe')
                plt.legend(prop={'size':LEGEND_FONTSIZE})
                
                plt.subplots_adjust(left=0.07, bottom=0.13, right=0.95, top=0.92)
                
                plt.title('Spectrum Comparison: {} vs. SN2011fe'.format(title), fontsize=TITLE_FONTSIZE)
                plt.show()
                
                
        perphase_fit()



if __name__=="__main__":
        features1 = [(3425, 3820, 'CaII'), (3900, 4100, 'SiII'), (5640, 5900, 'SiII'),
                                (6000, 6280, 'SiII'), (8000, 8550, 'CaII')]
        features2 = []
                                
        info_dict1 = cPickle.load(open("spectra_mag_fit_results_FILTERED.pkl", 'rb'))
        info_dict2 = cPickle.load(open("spectra_mag_fit_results_UNFILTERED.pkl", 'rb'))
        
        for t in zip(["SN2012cu (Feature Filtered)", "SN2012cu"], [info_dict1, info_dict2], [features1, features2]):
                main(t[0], t[1], t[2])
