## spectra_example.py
## for presentation


import loader as L
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pprint import pprint
from itertools import izip

from scipy.interpolate import interp1d


TITLE_FONTSIZE = 28
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 16
INPLOT_LEGEND_FONTSIZE = 16
LEGEND_FONTSIZE = 15

SN12CU_CHISQ_DATA = pickle.load(open('sn12cu_chisq_data.pkl', 'rb'))

cu_all = L.get_12cu(redtype='fm', ebv=1.05, rv=2.7)[:12]
phases = [t[0] for t in cu_all]

fe_uninterp = L.get_11fe()
fe_all  = L.interpolate_spectra( phases, fe_uninterp )


#sn12cu_flux_scale = 10**(.4*2.3)

#print "DISTANCE SCALING: {}".format(sn12cu_flux_scale)

plt.figure()

adjust = 0
for i, fe in enumerate(fe_all):
        #adjust += 1e-12
        adjust += 5
        
        dat = SN12CU_CHISQ_DATA[i]
        
        print fe[0], dat['phase']
        
        cu = L.get_12cu(redtype='fm', ebv=dat['BEST_EBV'], rv=dat['BEST_RV'])[i]
        
        LIM = 10000
        
        cu_wave = cu[1].wave[cu[1].wave<LIM]
        cu_flux = cu[1].flux[:cu_wave.shape[0]]
        
        fe_wave = fe[1].wave[fe[1].wave<LIM]
        fe_flux = fe[1].flux[:fe_wave.shape[0]]
        
        cu_interp = interp1d(cu_wave, cu_flux)
        fe_interp = interp1d(fe_wave, fe_flux)
        
        NORM_POINT = 7250
        
        cu_flux_norm = cu_flux/cu_interp(NORM_POINT)
        fe_flux_norm = fe_flux/fe_interp(NORM_POINT)
        
        plt.plot( fe_wave, fe_flux_norm - adjust, 'g', lw=1.5)
        plt.plot( cu_wave, cu_flux_norm - adjust, 'k', lw=1.5)
        
        ty = np.average((fe_flux_norm - adjust)[:20])
        
        
        #threshold = 2
        #fe_flux_norm_interp = interp1d(fe_wave, fe_flux_norm)
        
        #in_region = False
        #idx_start = 0
        #for i, lam in enumerate(cu_wave):
                #if in_region:
                        #if lam<9500 and np.abs(cu_flux_norm[i]-fe_flux_norm_interp(lam))<threshold:
                                #tmp_avg = np.average(cu_flux_norm[idx_start:i]-adjust)
                                #plt.axvspan(cu_wave[idx_start], cu_wave[i], facecolor='0.5', alpha=0.5)
                                #in_region = False
                #else:
                        #if lam<9500 and np.abs(cu_flux_norm[i]-fe_flux_norm_interp(lam))>threshold:
                                #idx_start=i
                                #in_region = True
        
        if i==0:
                plt.text(3250, ty-0.02e-11, "Phase:\n{}".format(cu[0]), horizontalalignment='right',
                         color='k', fontsize=INPLOT_LEGEND_FONTSIZE)
        else:
                plt.text(3250, ty-0.02e-11, str(cu[0]), horizontalalignment='right',
                         color='k', fontsize=INPLOT_LEGEND_FONTSIZE)



plt.plot([], [], 'k', lw=1.5, label='SN2012cu')
plt.plot([], [], 'g', lw=1.5, label='SN2011fe')

#plt.ylabel('flux ($ergs/sec/m^2$) + offset', fontsize=AXIS_LABEL_FONTSIZE)
plt.xlabel('wavelength ($\AA$)', fontsize=AXIS_LABEL_FONTSIZE)

plt.tick_params(\
    axis='y',          
    which='both',      
    left='off',      
    right='off',         
    labelleft='off')
    
plt.tick_params(axis='both', which='both', labelsize=TICK_LABEL_FONTSIZE)

plt.xlim(2900, 7000)
plt.setp(plt.gca(), 'yticklabels', [])


plt.title('Spectrum Time Series', fontsize=TITLE_FONTSIZE)

plt.legend(prop={'size':LEGEND_FONTSIZE})

plt.show(), plt.clf()


























