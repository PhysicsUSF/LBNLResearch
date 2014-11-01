'''
::AUTHOR::
Andrew Stocker

'''
import loader as l
from loader import redden_fm

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
        NORM_MIN = 7000
        NORM_MAX = 8000
        
        
        ebv_guess = 1.05
        ebv_pad = .3
        
        rv_guess = 2.7
        rv_pad = .5
        
        steps = 10
        
        pristine_12cu = l.get_12cu('fm', ebv=0.024, rv=3.1)[:12]
        phases = [t[0] for t in pristine_12cu]
        
        pristine_11fe = l.interpolate_spectra(phases, l.get_11fe())
        #reddened = l.interpolate_spectra(phases, l.get_11fe('fm', ebv=-0.5, rv=3.1))
        reddened = pristine_12cu
        
        
        def filter_features(spectrum):
                features = [(3425, 3820), (3900, 4100), (5640, 5900),
                                (6000, 6280), (8000, 8550)]
                
                wave = spectrum.wave
                flux = spectrum.flux
                
                intersection = np.array([False]*wave.shape[0])
                for feature in features:
                        intersection |= ((wave>feature[0])&(wave<feature[1]))
                
                return ~intersection
        
        
        #def allphase_fit():
                #pad=-2
                
                #ref_fluxs_allphases = np.concatenate( tuple(
                        #[interp1d(t[1].wave, t[1].flux)(reddened[i][1].wave[:pad]) for i, t in enumerate(pristine_11fe)]
                #))
                
                
                #x = np.linspace(ebv_guess-ebv_pad, ebv_guess+ebv_pad, steps)
                #y = np.linspace(rv_guess-rv_pad, rv_guess+rv_pad, steps)
                
                #X, Y = np.meshgrid(x, y)
                #SSR = np.zeros( X.shape )
                
                #for i, EBV in enumerate(x):
                        #for j, RV in enumerate(y):
                                #reddened_allphases = np.concatenate(tuple(
                                        #[redden_fm(t[1].wave[:pad], t[1].flux[:pad], EBV, RV) for t in reddened]
                                #))
                                
                                
                                #SSR[j,i] = np.sum((reddened_allphases-ref_fluxs_allphases)**2)
                
                ##print SSR
                #ssr_min = np.min(SSR)
                
                #mindex = np.where(SSR==ssr_min)
                #mx, my = mindex[1][0], mindex[0][0]
                
                #print mindex
                #print x[mx], y[my]
        
        
        
        
        #plt.figure()
        def perphase_fit():
                for phase in xrange(len(phases)):
                        ref = pristine_11fe[phase]
                        red = reddened[phase]
                        
                        print ref[0], red[0]
                        
                        x = np.linspace(ebv_guess-ebv_pad, ebv_guess+ebv_pad, steps)
                        y = np.linspace(rv_guess-rv_pad, rv_guess+rv_pad, steps)
                        
                        X, Y = np.meshgrid(x, y)
                        SSR = np.zeros( X.shape )
                        
                        print "11fe:"
                        print "wave:", ref[1].wave
                        print "flux:", ref[1].flux
                        
                        print "12cu:"
                        print "wave:", red[1].wave
                        print "flux:", red[1].flux
                        
                        for j, EBV in enumerate(x):
                                for k, RV in enumerate(y):
                                        
                                        interpolated_red = interp1d(red[1].wave, red[1].flux)(ref[1].wave)
                                        if k==9 and j==9:
                                                print "INTERPOLATED:", interpolated_red
                                        
                                        unred_flux = redden_fm(ref[1].wave, interpolated_red, EBV, RV)
                                        
                                        ref_flux = ref[1].flux
                                        
                                        mask = (ref[1].wave>NORM_MIN)&(ref[1].wave<NORM_MAX)
                                        unred_norm_factor = np.average(unred_flux[mask])
                                        ref_norm_factor = np.average(ref_flux[mask])
                                        
                                        unred_flux /= unred_norm_factor
                                        ref_flux /= ref_norm_factor
                                        
                                        #plt.plot(ref[1].wave, unred_flux, 'r')
                                        #plt.plot(ref[1].wave, ref_flux, 'k')
                                        
                                        SSR[k,j] = np.sum((unred_flux-ref_flux)**2)
                        
                        #print SSR
                        
                        ssr_min = np.min(SSR)
                        
                        mindex = np.where(SSR==ssr_min)
                        mx, my = mindex[1][0], mindex[0][0]
                        
                        print "\t", mindex
                        print "\t", x[mx], y[my]
                
                #plt.show()
                
                
        perphase_fit()
        
        

if __name__=="__main__":
        main()
