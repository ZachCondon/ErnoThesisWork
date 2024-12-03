# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:41:54 2024

@author: zacht

Neural Network algorithm used for my PhD. One of the things I noticed is that 
 the neural network is generally better when doing only the direct unfolding. I
 think that it is able to solve more accurately with that limited data set than
 with including the 2-step (direct and indirect) unfolding process. It's
 something I want to explore more, but just haven't yet.
 
The way I store my data is probably confusing at first, but hopefully it will
 make sense. It was helpful for me to recall information and write new
 functions when I needed to. I used dictionaries to store all of the unfolding
 data and then saved them as pickles. This script is written to do the complete
 calculations with the neural network portion of my PhD.

This script also saves everything that it does and will look for that save data
 every time it runs. Basically the order of operations is:
     - If results file exists:
         - make figures
     - If results file doesn't exist and if the networks have been trained
         - calculate results
         - make figures
     - If networks have not been trained
         - train networks
         - calculate results
         - make figures

 
***DISCLAIMER: This neural network is not optimized nor is it perfect. My goal
                in the PhD was to show that a NN can be used to unfold. Future
                work will include optimization.
                ***
 
ALGORITHM SECTIONS:                                              
    0. Import modules
    1. Unfolding Functions
    2. Import General Data
    3. Figure Functions
    4. Get Values for Reference Spectra
    5. Unfold with NN
    
NN Unfolding Overview:
    The network I used to unfold has two hidden layers. It takes as an input
     the detector response and gives the energy spectrum as an output. I
     eventually modified this so that multiple networks could be trained and
     the outputs of the networks could be averaged
"""

# The following variables are things you can adjust to reduce calculation time
#  especially when testing different things. 
num_NN_models = 1       # Number of different models to average with
num_epochs = 50         # Number of epochs to train on
neurons_layer1 = 1024   # Number of neurons in hidden layer 1
neurons_layer2 = 512    # Number of neurons in hidden layer 2
###############################################################################
#%% 0. Import modules                                                           #
###############################################################################
from Data.data import Constants
constants = Constants()         # This module is where I import all of the repeated data that I used through my PhD

import numpy as np
from numpy.linalg import norm
import pandas as pd
import pickle

import matplotlib.pyplot as plt

import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

###############################################################################
#%% 1. Unfolding functions                                                      #
###############################################################################

def import_dr(whichDR, D):
    # Import the experimental DR and remove the duplicate center values
    if whichDR == 'Cf':
        dr = np.concatenate((constants.DR_LLNL_Cf252_mR_xAxis, constants.DR_LLNL_Cf252_mR_yAxis, constants.DR_LLNL_Cf252_mR_zAxis))
    elif whichDR == 'AmBe':
        dr = np.concatenate((constants.DR_LLNL_AmBe_mR_xAxis, constants.DR_LLNL_AmBe_mR_yAxis, constants.DR_LLNL_AmBe_mR_zAxis))
    elif whichDR == 'GODIVA':
        dr = np.concatenate((constants.DR_LLNL_Godiva_TLD600_nC_xAxis, constants.DR_LLNL_Godiva_TLD600_nC_yAxis, constants.DR_LLNL_Godiva_TLD600_nC_zAxis))
    elif whichDR == 'NIF':
        # This is the NIF1 experimental detector response
        dr = np.concatenate((constants.DR_LLNL_NIF1_Net_mR_xAxis, constants.DR_LLNL_NIF1_Net_mR_yAxis, constants.DR_LLNL_NIF1_Net_mR_zAxis))
    # We also have the remove the duplicate center detector values
    dr = np.delete(dr, [28,47], 0)
    # The detector also needs to get converted to the correct units
    dr = dr/1000                # convert from mRad to Rad
    dr = dr/100                 # 1 Gy = 100 Rad, convert to Gy
    dr = dr/D                   # The TLD calibration factor to convert to mSv
    return dr

def normalize(array_to_be_normalized):
    # This takes a 1D array and normalizes it by its sum.
    return array_to_be_normalized/sum(array_to_be_normalized)

def cosine_sim(A,B):
    # Calculate the cosine similarity between two spectra. This is the metric used to train the neural network
    cos_sim = np.dot(A,B)/(norm(A)*norm(B))
    return cos_sim

def find_peak_max(spectrum, left_index, right_index):
    sub_array = spectrum[left_index:right_index]
    peak_location = np.where(spectrum == max(sub_array))[0][0]
    peak_range = np.array([peak_location-1, peak_location+1])
    energy_range = np.array([constants.E_BINS_MCNP[peak_range[0]], constants.E_BINS_MCNP[peak_range[1]]])
    return peak_location, energy_range

def calculate_normalized_dose(spectrum):
    dose_pSv_norm = np.sum(constants.H10Dose_MCNP * spectrum)
    return dose_pSv_norm

def get_full_DR(data, experimental_dr, full_drm):
    # Calculate a full detector response based off of the unfolded spectrum
    data['Unfolded Calculated Full DR'] = full_drm.dot(data['Mean Spectrum'])
    # Scale it to the correct units
    data['Unfolded Calculated Full DR'] = data['Unfolded Calculated Full DR']*data['Unit Ratio']

    # Find the indirect detector response
    data['Indirect DR'] = experimental_dr - data['Unfolded Calculated Full DR']
    data['Indirect DR'][data['Indirect DR'] < 0] = 0
    data['Indirect DR Error'] = data['Indirect DR'] * percE
    return data

def find_weights(data, drm_p, drm_s):
    # Find the correct weights of each unfolded spectrum to match the
    #  experimental DR
    direct_scaling_factors = (np.arange(100)+1)/100
    accuracies = np.zeros(100)
    indirect_scaling_factors = np.zeros(100)
    for i in range(len(direct_scaling_factors)):
        scaled_direct_dr = direct_scaling_factors[i] * data['Direct']['Unfolded Calculated Full DR']
        indirect_scaling_factor = 0.01
        scaled_indirect_dr = indirect_scaling_factor * data['Indirect']['Unfolded Calculated Full DR']
        sumSquares_new = np.sqrt(np.sum(np.square(scaled_direct_dr + scaled_indirect_dr - data['Experimental DR'])))
                                 
        sumSquares_old = sumSquares_new * 1.1
        while sumSquares_new < sumSquares_old:
            sumSquares_old = sumSquares_new
            indirect_scaling_factor += 0.01
            scaled_indirect_dr = indirect_scaling_factor * data['Indirect']['Unfolded Calculated Full DR']
            sumSquares_new = np.sqrt(np.sum(np.square(scaled_direct_dr + scaled_indirect_dr - data['Experimental DR'])))
            # if indirect_scaling_factor >= 1:
            #     break
        indirect_scaling_factor -= 0.01
        indirect_scaling_factors[i] = indirect_scaling_factor
        accuracies[i] = np.sqrt(np.sum(np.square(scaled_direct_dr + scaled_indirect_dr - data['Experimental DR'])))
    minimum_direct_scaling_factor = direct_scaling_factors[accuracies==accuracies.min()]
    minimum_indirect_scaling_factor = indirect_scaling_factors[accuracies==accuracies.min()]
    print(minimum_direct_scaling_factor)
    print(minimum_indirect_scaling_factor)
    
    weights = {}
    weights['min_direct_scaling_factor'] = minimum_direct_scaling_factor
    weights['min_indirect_scaling_factor'] = minimum_indirect_scaling_factor
    weights['Mean Spectrum'] = minimum_direct_scaling_factor * data['Direct']['Mean Spectrum'] + minimum_indirect_scaling_factor * data['Indirect']['Mean Spectrum']
    weights['Mean Spectrum'] = normalize(weights['Mean Spectrum'])
    stdDev_withDRM_direct = np.zeros(drm_p.shape)
    stdDev_withDRM_indirect = np.zeros(drm_s.shape)
    StdDev_direct = np.zeros(drm_p.shape[0])
    StdDev_indirect = np.zeros(drm_s.shape[0])
    for i in range(drm_p.shape[0]):
        stdDev_withDRM_direct[i] = np.multiply(drm_p[i], data['Direct']['Unfolded Spectra StdDev'])
        stdDev_withDRM_indirect[i] = np.multiply(drm_s[i], data['Indirect']['Unfolded Spectra StdDev'])
        variances_direct = np.square(stdDev_withDRM_direct[i])
        variances_indirect = np.square(stdDev_withDRM_indirect[i])
        sum_variances_direct = sum(variances_direct)
        sum_variances_indirect = sum(variances_indirect)
        StdDev_direct[i] = np.sqrt(sum_variances_direct) * data['Direct']['Unit Ratio']
        StdDev_indirect[i] = np.sqrt(sum_variances_indirect) * data['Indirect']['Unit Ratio']
    weights['Weighted Unfolded DR StdDev'] = minimum_direct_scaling_factor * StdDev_direct + minimum_indirect_scaling_factor * StdDev_indirect
    # Find the Weighted StdDev for the unfolded spectrum
    StdDev_p_squared_weighted = minimum_direct_scaling_factor * np.square(data['Direct']['Unfolded Spectra StdDev'])
    StdDev_r_squared_weighted = minimum_indirect_scaling_factor * np.square(data['Indirect']['Unfolded Spectra StdDev'])
    StdDev_pAndr_squared_added = StdDev_p_squared_weighted + StdDev_r_squared_weighted
    weights['Weighted StdDev'] = np.sqrt(StdDev_pAndr_squared_added)
    weights['Mean Spectrum+'] = weights['Mean Spectrum'] + weights['Weighted StdDev']
    weights['Mean Spectrum-'] = weights['Mean Spectrum'] - weights['Weighted StdDev']
    
    weights['Weighted Unfolded DR'] = minimum_direct_scaling_factor * data['Direct']['Unfolded Calculated Full DR'] + minimum_indirect_scaling_factor * data['Indirect']['Unfolded Calculated Full DR']
    weights['Weighted Unit Ratio'] = minimum_direct_scaling_factor * data['Direct']['Unit Ratio'] + minimum_indirect_scaling_factor * data['Indirect']['Unit Ratio']
    # sum_variances = np.sum(weighted_variances/num_subsets)
    # weighted_StdDev = np.sqrt(sum_variances)
    # weights['Weighted Unfolded DR StdDev'] = np.ones(len(weights['Weighted Unfolded DR'])) * weighted_StdDev * np.sqrt(sum(experimental_dr))
    weights['Error'] = np.sqrt(np.sum(np.square(weights['Weighted Unfolded DR'] - data['Experimental DR'])))
    #### MAKE CALCULATIONS ####
    weights['Calculations'] = calculate_results(Cf_spec,
                                                weights['Mean Spectrum'],
                                                data['Experimental DR'],
                                                weights['Weighted Unfolded DR'])
    return weights

def calculate_results(real_spec, unfolded_spec, exp_dr, unfolded_dr):
    calculations = {}
    # Calculate cosine similarity of unfolded spectrum with the real spectrum
    calculations['Cosine Similarity'] = cosine_sim(real_spec, unfolded_spec)
    # Calculate the DR percent error
    calculations['Unfolded DR %error'] = abs(unfolded_dr-exp_dr)/exp_dr
    calculations['Unfolded DR mean %error'] = np.mean(calculations['Unfolded DR %error'])
    # Low energy peak
    peak_location, energy_range = find_peak_max(unfolded_spec, 0, 13)
    calculations['Low Energy Peak Bin'] = peak_location
    calculations['Low Energy Peak Info'] = f'Energy Range: {energy_range[0]*1e3:.2f} meV - {energy_range[1]*1e3:.2f} meV'
    # High energy peak
    peak_location, energy_range = find_peak_max(unfolded_spec, 40, 80)
    calculations['High Energy Peak Bin'] = peak_location
    calculations['High Energy Peak Info'] = f'Energy Range: {energy_range[0]/1e6:.2f} MeV - {energy_range[1]/1e6:.2f} MeV'
    # Calculate Dose
    calculations['Dose'] = calculate_normalized_dose(unfolded_spec)
    calculations['Dose Info'] = f"Normalized Dose: {calculations['Dose']:.2e} pGy/Fluence"
    # Calculate the mean percent error of the Unfolded DR
    calculations['Unfolded DR % error'] = abs(normalize(exp_dr)-normalize(unfolded_dr))/normalize(exp_dr)
    calculations['Unfolded DR mean % error'] = np.mean(calculations['Unfolded DR % error'])
    return calculations

###############################################################################
# 2. Import general data                                                      #
###############################################################################
ebins = constants.E_BINS_MCNP
percE = 0.08                                 # percent error on the TLD readings

# Import the direct DRM
drm_d = constants.full_TLD_DRM                    # planar drm
# The way I have the DRM stored, I have the center detector in all three axes.
drm_d = np.delete(drm_d, [28,47], 0)                # Delete the center detectors that are oriented with the x-axis detectors

# Import the spherical DRM
drm_s = constants.full_TLD_DRM_spherical
drm_s = np.delete(drm_s, [28,47], 0)

# The DRMs are the direct tallies from MCNP (in units of MeV/g) and need to be 
#  converted to real units. The following are the unit conversions:
q = 1.6e-13                                 # convert MeV to J, units: J/g
M = 1000                                    # convert g to kg,  units: J/kg
A = 1509                                    # Area of source,   units: J cm^2/kg
Q = .211                                    # light conversion efficiency
D = 1.33e-3                                 # calibration factor units: Gy/mSv
# And now the unit conversions are applied to get units of mSv * cm^2
drm_d = drm_d*q*M*A*Q/D
drm_s = drm_s*q*M*4*np.pi*50**2*Q/D # This DRM technically has six times the area.

# Reference IAEA Spectra (normalize the interpolated IAEA spectrum using my energy structure)
#  normalize(np.interp(ebins, IAEA_ebins, IAEA_spec))
Cf_spec = normalize(np.interp(ebins, constants.E_BINS_IAEA, constants.IAEA_spectra[4])) # Bare Cf Spectra Measured without shadow cone
Cf_spec_shadowcone = normalize(np.interp(ebins, constants.E_BINS_IAEA, constants.IAEA_spectra[5])) # Bare Cf Spectra Measured with shadow cone
AmBe_spec = normalize(np.interp(ebins, constants.E_BINS_IAEA, constants.IAEA_spectra[17])) # Bare AmBe Spectra Measured without shadow cone
AmBe_spec_shadowcone = normalize(np.interp(ebins, constants.E_BINS_IAEA, constants.IAEA_spectra[16])) # Bare AmBe Spectra Measured with shadow cone

#### DATA FOR GODIVA NADS EXPERIMENT ####
NAD_E_BINS_MeV = np.array([1.58e-9, 2.51e-9, 3.98e-9, 6.31e-9,
                           1.00e-8, 1.58e-8, 2.51e-8, 3.98e-8, 6.31e-8,
                           1.00e-7, 1.58e-7, 2.51e-7, 3.98e-7, 6.31e-7,
                           1.00e-6, 1.00e-2, 5.04e-2, 5.72e-2, 6.51e-2, 7.39e-2, 8.37e-2, 9.55e-2,
                           1.08e-1, 1.23e-1, 1.40e-1, 1.58e-1, 2.01e-1, 2.22e-1, 2.47e-1, 2.74e-1,
                           3.05e-1, 3.38e-1, 3.78e-1, 4.21e-1, 4.66e-1, 5.18e-1, 6.01e-1, 6.71e-1,
                           7.41e-1, 8.25e-1, 9.16e-1, 1.02e0, 1.13e0, 1.26e0, 1.42e0, 1.61e0, 1.85e0,
                           2.09e0, 2.38e0, 2.69e0, 3.07e0, 3.48e0, 3.48e0, 3.96e0, 4.73e0, 
                           5.31e0, 5.88e0, 6.43e0, 6.98e0, 7.52e0, 8.04e0, 8.56e0, 9.07e0,
                           9.57e0, 1.06e1])
NAD_E_BINS = NAD_E_BINS_MeV * 1e6
NAD_fluencePerLethargy = np.array([4.13e5, 3.06e6, 2.19e7, 1.07e8, 4.07e8, 9.42e8,
                                   1.77e9, 2.58e9, 2.87e9, 2.97e9, 2.51e9, 1.64e9,
                                   1.11e9, 1.08e9, 1.13e9, 1.63e9, 2.10e9, 1.59e9,
                                   2.16e9, 2.06e9, 2.34e9, 1.66e9, 2.37e9, 2.20e9,
                                   2.81e9, 2.61e9, 3.62e9, 3.90e9, 5.24e9, 4.59e9,
                                   6.37e9, 5.83e9, 7.67e9, 8.91e9, 8.20e9, 1.06e10,
                                   9.71e9, 1.23e10, 1.21e10, 1.34e10, 1.20e10, 1.03e10,
                                   1.48e10, 1.30e10, 1.09e10, 8.28e9, 1.03e10, 1.05e10,
                                   4.57e9, 5.37e9, 6.76e9, 3.24e9, 2.09e9, 2.85e9, 1.59e9,
                                   6.11e8, 7.54e8, 9.18e8, 8.21e8, 6.32e8, 0, 0, 0, 0, 0])
NAD_apriori = normalize(np.interp(ebins, NAD_E_BINS, NAD_fluencePerLethargy))

###############################################################################
# 3. Figure Functions                                                         #
###############################################################################
def plot_direct_dr(DRd, DRd_error, DRe, DRe_error, title, xlabel='Detector Number', ylabel='Detector Response (mSv)'):
    fig, ax = plt.subplots()
    ax.errorbar(range(len(DRe)), DRd, DRd_error, color=constants.palette[0], ecolor=constants.palette[0], capsize=4, label='Unfolded')
    ax.errorbar(range(len(DRe)), DRe, DRe_error, color=constants.palette[1], ecolor=constants.palette[1], capsize=4, label='Experimental')
    ax.legend()
    ax.set_title(title, fontdict=constants.plt_title_text)
    ax.set_xlabel(xlabel, fontdict=constants.plt_label_text)
    ax.set_ylabel(ylabel, fontdict=constants.plt_label_text)
    return fig, ax

def plot_direct_spec(ebins, meanSpec, meanSpecP, meanSpecM, title, xlabel='Energy (eV)', ylabel='Normalized Fluence\nper Unit Lethargy', ref_spec=False):
    fig, ax = plt.subplots()
    ax.step(ebins, meanSpec, label='Unfolded Spectrum', color=constants.palette[0])
    ax.step(ebins, meanSpecP/sum(meanSpec), color='grey', linestyle='dashed', label='+/- StdDev')
    ax.step(ebins, meanSpecM/sum(meanSpec), color='grey', linestyle='dashed')
    try:
        ax.step(ebins, ref_spec, label='Referece Spectrum', color=constants.palette[1])
    except:
        print('No reference spec given')
    ax.semilogx()
    ax.legend()
    ax.set_title(title, fontdict=constants.plt_title_text)
    ax.set_xlabel(xlabel, fontdict=constants.plt_label_text)
    ax.set_ylabel(ylabel, fontdict=constants.plt_label_text)
    return fig, ax

def plot_full_direct_dr(DRd, DRd_error, DRe, DRe_error, DRsubtracted, DRsubtracted_error, title, xlabel='Detector Number', ylabel='Detector Response (mSv)'):
    fig, ax = plt.subplots()
    ax.errorbar(range(len(DRe)), DRd, DRd_error, color=constants.palette[0], ecolor=constants.palette[0], capsize=4, label='Unfolded')
    ax.errorbar(range(len(DRe)), DRe, DRe_error, color=constants.palette[1], ecolor=constants.palette[1], capsize=4, label='Experimental')
    ax.errorbar(range(len(DRe)), DRsubtracted, DRsubtracted_error, color=constants.palette[2], ecolor=constants.palette[2], capsize=4, label='Subtracted')
    ax.legend()
    ax.set_title(title, fontdict=constants.plt_title_text)
    ax.set_xlabel(xlabel, fontdict=constants.plt_label_text)
    ax.set_ylabel(ylabel, fontdict=constants.plt_label_text)
    return fig, ax

def plot_indirect_dr(DRi, DRi_error, DRe, DRe_error, title, xlabel='Detector Number', ylabel='Detector Response (mSv)'):
    fig, ax = plt.subplots()
    ax.errorbar(range(len(DRe)), DRi, DRi_error, color=constants.palette[0], ecolor=constants.palette[0], capsize=4, label='Unfolded')
    ax.errorbar(range(len(DRe)), DRe, DRe_error, color=constants.palette[1], ecolor=constants.palette[1], capsize=4, label='Experimental')
    ax.legend()
    ax.set_title(title, fontdict=constants.plt_title_text)
    ax.set_xlabel(xlabel, fontdict=constants.plt_label_text)
    ax.set_ylabel(ylabel, fontdict=constants.plt_label_text)
    return fig, ax

def plot_indirect_spec(ebins, meanSpec, meanSpecP, meanSpecM, title, xlabel='Energy (eV)', ylabel='Normalized Fluence\nper Unit Lethargy', ref_spec=False):
    fig, ax = plt.subplots()
    ax.step(ebins, meanSpec, label='Unfolded Spectrum', color=constants.palette[0])
    ax.step(ebins, meanSpecP/sum(meanSpec), color='grey', linestyle='dashed', label='+/- StdDev')
    ax.step(ebins, meanSpecM/sum(meanSpec), color='grey', linestyle='dashed')
    try:
        ax.step(ebins, ref_spec, label='Referece Spectrum', color=constants.palette[1])
    except:
        print('No reference spec given')
    ax.semilogx()
    ax.legend()
    ax.set_title(title, fontdict=constants.plt_title_text)
    ax.set_xlabel(xlabel, fontdict=constants.plt_label_text)
    ax.set_ylabel(ylabel, fontdict=constants.plt_label_text)
    return fig, ax

def plot_weighted_dr(DRw, DRw_error, DRe, DRe_error, title, xlabel='Detector Number', ylabel='Detector Response (mSv)'):
    fig, ax = plt.subplots()
    ax.errorbar(range(len(DRe)), DRw, DRw_error, color=constants.palette[0], ecolor=constants.palette[0], capsize=4, label='Unfolded')
    ax.errorbar(range(len(DRe)), DRe, DRe_error, color=constants.palette[1], ecolor=constants.palette[1], capsize=4, label='Experimental')
    ax.legend()
    ax.set_title(title, fontdict=constants.plt_title_text)
    ax.set_xlabel(xlabel, fontdict=constants.plt_label_text)
    ax.set_ylabel(ylabel, fontdict=constants.plt_label_text)
    return fig, ax

def plot_weighted_spec(ebins, meanSpec, meanSpecP, meanSpecM, title, xlabel='Energy (eV)', ylabel='Normalized Fluence\nper Unit Lethargy', ref_spec=False):
    fig, ax = plt.subplots()
    ax.step(ebins, meanSpec, label='Unfolded Spectrum', color=constants.palette[0])
    ax.step(ebins, meanSpecP/sum(meanSpec), color='grey', linestyle='dashed', label='+/- StdDev')
    ax.step(ebins, meanSpecM/sum(meanSpec), color='grey', linestyle='dashed')
    try:
        ax.step(ebins, ref_spec, label='Referece Spectrum', color=constants.palette[1])
    except:
        print('No reference spec given')
    ax.semilogx()
    ax.legend()
    ax.set_title(title, fontdict=constants.plt_title_text)
    ax.set_xlabel(xlabel, fontdict=constants.plt_label_text)
    ax.set_ylabel(ylabel, fontdict=constants.plt_label_text)
    return fig, ax

###############################################################################
# 4. Get Values for Reference Spectra                                         #
###############################################################################
validation = {}
validation['Reference'] = {}

validation['Reference']['Cf'] = {}
validation['Reference']['Cf']['Spectrum'] = Cf_spec
validation['Reference']['Cf']['DR'] = np.dot(drm_d, Cf_spec)/sum(np.dot(drm_d, Cf_spec))
validation['Reference']['Cf']['Calculations'] = calculate_results(validation['Reference']['Cf']['Spectrum'],
                                                                       validation['Reference']['Cf']['Spectrum'],
                                                                       validation['Reference']['Cf']['DR'],
                                                                       validation['Reference']['Cf']['DR'])

validation['Reference']['AmBe'] = {}
validation['Reference']['AmBe']['Spectrum'] = AmBe_spec
validation['Reference']['AmBe']['DR'] = np.dot(drm_d, AmBe_spec)/sum(np.dot(drm_d, AmBe_spec))
validation['Reference']['AmBe']['Calculations'] = calculate_results(validation['Reference']['AmBe']['Spectrum'],
                                                                         validation['Reference']['AmBe']['Spectrum'],
                                                                         validation['Reference']['AmBe']['DR'],
                                                                         validation['Reference']['AmBe']['DR'])

###############################################################################
# 5. Unfold with NN                                                           #
###############################################################################
def unfold_NN(drm, exp_dr, ref_spec, dir_or_indir):
    num_models = num_NN_models
    source = {}
    # Calculate a detector response
    source['Sum Experimental DR'] = sum(exp_dr.copy())
    source['Experimental DR'] = exp_dr.copy()
    source['Experimental DR NN'] = normalize(source['Experimental DR'].copy())
    # source['Experimental DR'] = source['Experimental DR']/sum(source['Experimental DR'])
    source['Experimental DR error'] = source['Experimental DR NN'] * percE
    
    num_detectors = source['Experimental DR'].shape[0]
    
    # Train the network if needed, otherwise load in the trained network
    if not os.path.isfile(f'Models/Model0_{dir_or_indir}.keras'):    
        spectra = constants.load_fruit_spectra()
        Y = spectra.copy()
        X = np.dot(Y,drm.T)
        for i in range(Y.shape[0]):
            X[i] = X[i]/sum(X[i])
        
        Models = []
        for i in range(num_models):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42+i)

            #### Build and train the network ####
            # Callbacks
            early_stopping = EarlyStopping(
                min_delta=0.00001,
                patience=20,
                restore_best_weights=True)

            inputs = keras.Input(shape = (num_detectors))
            x = layers.Dense(neurons_layer1)(inputs)
            x = layers.BatchNormalization()(x)
            x = keras.activations.relu(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(neurons_layer2)(x)
            x = layers.BatchNormalization()(x)
            x = keras.activations.relu(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(84, activation='softmax')(x)
            Models.append(keras.Model(inputs=inputs, outputs=outputs))
            # model = keras.Model(inputs=inputs, outputs=outputs)
        
            # model.compile(
            Models[i].compile(
                optimizer = keras.optimizers.Adam(0.0001),
                loss = keras.losses.MeanSquaredError(),
                metrics=['CosineSimilarity']
                )
        
            # Train the Model
            # history = model.fit(
            history = Models[i].fit(
                X_train, Y_train,
                validation_split = 0.2,
                batch_size=20000,
                epochs=num_epochs,
                verbose=2,
                callbacks = [early_stopping]
                )
            history_df = pd.DataFrame(history.history)
        
            Models[i].save(f'Models/Model{i}_{dir_or_indir}.keras')    
            # model.save(f'Models/Model{i}.keras')
        
            plot_loss = history_df.loc[5:, ['loss', 'val_loss']].plot(
                title='Loss vs Epoch',
                xlabel='Epochs',
                ylabel='Loss (mse)'
                )
            fig_loss = plot_loss.get_figure()
            fig_loss.savefig(f'Models/Model{i}_{dir_or_indir}_Loss.jpeg', dpi=300)
        
            plot_accuracy = history_df.loc[5:, ['cosine_similarity', 'val_cosine_similarity']].plot(
                title='Accuracy vs Epoch',
                xlabel='Epochs',
                ylabel='Accuracy (%)'
                )
            fig_accuracy = plot_accuracy.get_figure()
            fig_accuracy.savefig(f'Models/Model{i}_{dir_or_indir}_Accuracy.jpeg', dpi=300)
        
    elif os.path.isfile(f'Models/Model0_{dir_or_indir}.keras'):
        # Load in the models
        Models = []
        for i in range(num_models):
            Models.append(keras.models.load_model(f'Models/Model{i}_{dir_or_indir}.keras'))
    
    source['Unfolded Spectra'] = np.zeros((num_models, 84))
    for i in range(num_models):
        source['Unfolded Spectra'][i] = Models[i].predict(source['Experimental DR NN'].reshape(1,-1))[0]
    
    # Get the StdDev of all the unfolded spectra
    source['Unfolded Spectra StdDev'] = np.std(source['Unfolded Spectra'], axis=0)
    # Get the mean of all the unfolded spectra
    source['Mean Spectrum'] = np.mean(source['Unfolded Spectra'], axis=0)
    # Add/subtract the StdDev
    source['Mean Spectrum+'] = source['Mean Spectrum'] + source['Unfolded Spectra StdDev']
    source['Mean Spectrum-'] = source['Mean Spectrum'] - source['Unfolded Spectra StdDev']
    # Set any values in the Mean-StdDev to zero if they are below zero
    source['Mean Spectrum-'][source['Mean Spectrum-'] < 0] = 0
    
    source['Unfolded DR'] = drm.dot(source['Mean Spectrum'])
    source['Unit Ratio'] = source['Sum Experimental DR']/sum(source['Unfolded DR'])
    source['Unfolded DR'] = source['Unfolded DR'] * source['Unit Ratio']
    # Get StdDev for the DR
    stdDev_withDRM = np.zeros(drm.shape)
    StdDev = np.zeros(drm.shape[0])
    for i in range(drm.shape[0]):
        stdDev_withDRM[i] = np.multiply(drm[i], source['Unfolded Spectra StdDev'])
        variances = np.square(stdDev_withDRM[i])
        sum_variances = sum(variances)
        StdDev[i] = np.sqrt(sum_variances)
    source['Unfolded DR StdDev'] = StdDev
        
    # Get stats of the unfolded spectrum
    source['Calculations'] = calculate_results(ref_spec,
                                               source['Mean Spectrum'],
                                               source['Experimental DR'],
                                               source['Unfolded DR'])
    
    return source

def NN_algorithm(which_source, ref_spec_d, ref_spec_i):
    print(f'Unfolding {which_source} with NN')
    which_source_dict = {}
    
    # Import detector response and get its associated standard error
    which_source_dict['Experimental DR'] = import_dr(which_source, D)
    which_source_dict['Experimental DR Standard Error'] = which_source_dict['Experimental DR']*percE
    which_source_dict['Number of Detectors'] = len(which_source_dict['Experimental DR'])
    
    #### UNFOLD DIRECT RESPONSE ####
    which_source_dict['Direct'] = unfold_NN(drm_d[:10], which_source_dict['Experimental DR'][:10], ref_spec_d, 'direct')
    which_source_dict['Direct'] = get_full_DR(which_source_dict['Direct'], which_source_dict['Experimental DR'], drm_d)
    
    #### UNFOLD INDIRECT RESPONSE ####
    which_source_dict['Indirect'] = unfold_NN(drm_s[10:],  which_source_dict['Direct']['Indirect DR'][10:], ref_spec_i, 'indirect')
    which_source_dict['Indirect'] = get_full_DR(which_source_dict['Indirect'], which_source_dict['Experimental DR'], drm_s)
    
    #### FIND WEIGHTS ####
    which_source_dict['Weighted Sum'] = find_weights(which_source_dict, drm_d, drm_s)
    
    #### Make Plots ####
    # Direct DR
    title_text = f'NN Direct {which_source}'
    fig, ax = plot_direct_dr(which_source_dict['Direct']['Unfolded DR'],
                             which_source_dict['Direct']['Unfolded DR StdDev'],
                             which_source_dict['Experimental DR'][:10],
                             which_source_dict['Experimental DR Standard Error'][:10],
                             title=title_text
                             )
    fig.savefig(f'Results\\{which_source}_directDR.jpeg', dpi=300, bbox_inches='tight')
    # Direct Spectrum
    title_text = f'NN Direct {which_source}'
    fig, ax = plot_direct_spec(ebins,
                               which_source_dict['Direct']['Mean Spectrum'],
                               which_source_dict['Direct']['Mean Spectrum+'],
                               which_source_dict['Direct']['Mean Spectrum-'],
                               title=title_text,
                               ref_spec=ref_spec_d
                               )
    fig.savefig(f'Results\\{which_source}_directSpec.jpeg', dpi=300, bbox_inches='tight')
    # Full Direct DR
    title_text = f'NN Full Direct {which_source}'
    fig, ax = plot_full_direct_dr(which_source_dict['Direct']['Unfolded Calculated Full DR'],
                             which_source_dict['Direct']['Unfolded Calculated Full DR']*percE,
                             which_source_dict['Experimental DR'],
                             which_source_dict['Experimental DR Standard Error'],
                             which_source_dict['Direct']['Indirect DR'],
                             which_source_dict['Direct']['Indirect DR Error'],
                             title=title_text
                             )
    fig.savefig(f'Results\\{which_source}_directDR_full.jpeg', dpi=300, bbox_inches='tight')
    # Indirect DR
    title_text = f'NN Indirect {which_source}'
    fig, ax = plot_indirect_dr(which_source_dict['Indirect']['Unfolded DR'],
                             which_source_dict['Indirect']['Unfolded DR StdDev'],
                             which_source_dict['Direct']['Indirect DR'][10:],
                             which_source_dict['Direct']['Indirect DR Error'][10:],
                             title=title_text
                             )
    fig.savefig(f'Results\\{which_source}_indirectDR.jpeg', dpi=300, bbox_inches='tight')
    # Indirect Spectrum
    title_text = f'NN Indirect {which_source}'
    fig, ax = plot_indirect_spec(ebins,
                                 which_source_dict['Indirect']['Mean Spectrum'],
                                 which_source_dict['Indirect']['Mean Spectrum+'],
                                 which_source_dict['Indirect']['Mean Spectrum-'],
                                 title=title_text,
                                 ref_spec=ref_spec_i
                                 )
    fig.savefig(f'Results\\{which_source}_indirectSpec.jpeg', dpi=300, bbox_inches='tight')
    # Weighted DR
    title_text = f'NN Weighted {which_source}'
    fig, ax = plot_weighted_dr(which_source_dict['Weighted Sum']['Weighted Unfolded DR'],
                               which_source_dict['Weighted Sum']['Weighted Unfolded DR StdDev'],
                               which_source_dict['Experimental DR'],
                               which_source_dict['Experimental DR Standard Error'],
                               title=title_text
                               )
    fig.savefig(f'Results\\{which_source}_weightedDR.jpeg', dpi=300, bbox_inches='tight')
    # Weighted Spectrum
    title_text = f'NN Weighted {which_source}'
    fig, ax = plot_weighted_spec(ebins,
                                 which_source_dict['Weighted Sum']['Mean Spectrum'],
                                 which_source_dict['Weighted Sum']['Mean Spectrum+'],
                                 which_source_dict['Weighted Sum']['Mean Spectrum-'],
                                 title=title_text,
                                 ref_spec=ref_spec_d
                                 )
    fig.savefig(f'Results\\{which_source}_weightedSpec.jpeg', dpi=300, bbox_inches='tight')
    
    final_results_file = f'Results\\NN_{which_source}.pkl'
    with open(final_results_file, 'wb') as f:
        pickle.dump(which_source_dict, f)
    
    return which_source_dict

if not os.path.isfile('Results\\NN_Cf.pkl'):
    Cf = NN_algorithm('Cf', Cf_spec, Cf_spec_shadowcone)

if not os.path.isfile('Results\\NN_AmBe.pkl'):
    AmBe = NN_algorithm('AmBe', AmBe_spec, AmBe_spec_shadowcone)

if not os.path.isfile('Results\\NN_GODIVA.pkl'):
    GODIVA = NN_algorithm('GODIVA', NAD_apriori, NAD_apriori)

if not os.path.isfile('Results\\NN_NIF.pkl'):
    NIF_apriori = normalize(np.ones(84))
    NIF = NN_algorithm('NIF', NIF_apriori, NIF_apriori)