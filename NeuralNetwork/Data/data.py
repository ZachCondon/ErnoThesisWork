# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 07:39:19 2024

@author: zacht

This script has all of the data needed for creating all of the results in my 
article.
"""
###############################################################################
# 0. Import modules                                                           #
###############################################################################
import numpy as np
import pandas as pd

class Constants():
    
    def __init__(self):
        # Import energy values
        self.E_BINS_MCNP = np.array([1.00e-3, 1.58e-3, 2.51e-3, 3.98e-3, 6.31e-3,1.00e-2, 1.58e-2, 2.51e-2, 3.98e-2, 6.31e-2, 1.00e-1, 1.58e-1, 2.51e-1, 3.98e-1, 6.31e-1, 1.00e-0, 1.58e-0, 2.51e-0, 3.98e-0, 6.31e-0, 1.00e+1, 1.58e+1, 2.51e+1, 3.98e+1, 6.31e+1, 1.00e+2, 1.58e+2, 2.51e+2, 3.98e+2, 6.31e+2, 1.00e+3, 1.58e+3, 2.51e+3, 3.98e+3, 6.31e+3, 1.00e+4, 1.58e+4, 2.51e+4, 3.98e+4, 6.31e+4, 1.00e+5, 1.26e+5, 1.58e+5, 2.00e+5, 2.51e+5, 3.16e+5, 3.98e+5, 5.01e+5, 6.31e+5, 7.94e+5, 1.00e+6, 1.12e+6, 1.26e+6, 1.41e+6, 1.58e+6, 1.78e+6, 2.00e+6, 2.24e+6, 2.51e+6, 2.82e+6, 3.16e+6, 3.55e+6, 3.98e+6, 4.47e+6, 5.01e+6, 5.62e+6, 6.31e+6, 7.08e+6, 7.94e+6, 8.91e+6, 1.00e+7, 1.12e+7, 1.26e+7, 1.41e+7,1.58e+7, 1.78e+7, 2.00e+7, 2.51e+7, 3.16e+7, 3.98e+7,5.01e+7, 6.31e+7, 7.94e7, 1.00e+8]) # eV
        self.E_BINS_IAEA = np.array([1.00E-03, 2.15E-03, 4.64E-03, 1.00E-02, 2.15E-02, 4.64E-02, 1.00E-01, 2.15E-01, 4.64E-01, 1.00E-00, 2.15E-00, 4.64E-00, 1.00E+01, 2.15E+01, 4.64E+01, 1.00E+02, 2.15E+02, 4.64E+02, 1.00E+03, 2.15E+03, 4.64E+03, 1.00E+04, 1.25E+04, 1.58E+04, 1.99E+04, 2.51E+04, 3.16E+04, 3.98E+04, 5.01E+04, 6.30E+04, 7.94E+04, 1.00E+05, 1.25E+05, 1.58E+05, 1.99E+05, 2.51E+05, 3.16E+05, 3.98E+05, 5.01E+05, 6.30E+05, 7.94E+05, 1.00E+06, 1.25E+06, 1.58E+06, 1.99E+06, 2.51E+06, 3.16E+06, 3.98E+06, 5.01E+06, 6.30E+06, 7.94E+06, 1.00E+07, 1.58E+07, 2.51E+07, 3.98E+07, 6.30E+07, 1.00E+08, 1.58E+08, 2.51E+08, 3.98E+08]) # eV
        self.E_BINS_ANSI = np.array([1.50e-03, 3.24e-03, 6.98e-03, 1.50e-02, 3.24e-02, 6.98e-02, 1.50e-01, 3.24e-01, 6.98e-01, 1.50e-00, 3.24e-00, 6.98e-00, 1.50e+01, 3.24e+01, 6.98e+01, 1.50e+02, 3.24e+02, 6.98e+02, 1.50e+03, 3.24e+03, 6.98e+03, 1.12e+04, 1.41e+04, 1.78e+04, 2.24e+04, 2.82e+04, 3.55e+04, 3.55e+04, 4.48e+04, 5.63e+04, 7.09e+04, 8.93e+04, 1.12e+05, 1.41e+05, 1.78e+05, 2.24e+05, 2.82e+05, 3.55e+05, 4.48e+05, 5.63e+05, 7.09e+05, 8.93e+05, 1.12e+06, 1.41e+06, 1.78e+06, 2.24e+06, 2.82e+06, 3.55e+06, 4.48e+06, 5.63e+06, 7.09e+06, 8.93e+06, 1.27e+07, 1.78e+07])
        self.MCNP_MAX_ENERGY = 1.58e+8 # eV
        self.IAEA_MAX_ENERGY = 6.30e+8 # eV
        self.ANSI_MAX_ENERGY = 2.24e+7 # eV
        self.MCNP_LETHARGIES = self.calculate_lethargies(self.E_BINS_MCNP, self.MCNP_MAX_ENERGY)
        self.ANSI_LETHARGIES = self.calculate_lethargies(self.E_BINS_ANSI, self.ANSI_MAX_ENERGY)
        self.IAEA_LETHARGIES = self.calculate_lethargies(self.E_BINS_IAEA, self.IAEA_MAX_ENERGY)
        # Have a variable with the detector locations within the PNS
        self.detector_locations = np.array([14,13,12,11,10,9,8,6,3,0,-3,-6,-8,-9,-10,-11,-12,-13,-14])
        # Load in the DRMs
        self.load_drms()
        # Load in the experimental DRs
        self.load_experimental_drs()
        # Define various plot parameters
        # Create an array with the default matplotlib colors to make consistent plots
        self.mpl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.palette = ['#23a2c7', '#d8523a', '#e3b547', '#ce8a44', '#dbc951']
        self.plt_label_text = {'fontname':'Arial', 'fontsize':16}
        self.plt_title_text = {'fontname':'Arial', 'fontsize':18}
        # Load in IAEA data
        self.loadData_IAEA()
        self.load_dose_conversions()
        
    def load_drms(self):
        self.full_TLD_DRM_df = pd.read_csv('Data\\DRMs\\PNS_TLD_DRM_PlaneSource.csv')
        self.full_TLD_DRM = self.full_TLD_DRM_df.to_numpy()[:,1:].astype('float64')
        
        self.full_TLD_DRM_spherical_df = pd.read_csv('Data\\DRMs\\PNS_TLD_DRM_SphericalShellSource.csv')
        self.full_TLD_DRM_spherical = self.full_TLD_DRM_spherical_df.to_numpy()[:,1:].astype('float64')
    
    def load_experimental_drs(self):
        # Cf-252
        self.DR_LLNL_Cf252_mR_df = pd.read_csv('Data\\ExperimentalDRs\\DetRes_LLNL_Cf252_mR.csv')
        self.DR_LLNL_Cf252_mR_xAxis = self.DR_LLNL_Cf252_mR_df['xAxis (mR)'].to_numpy().astype('float64')
        self.DR_LLNL_Cf252_mR_yAxis = self.DR_LLNL_Cf252_mR_df['yAxis (mR)'].to_numpy().astype('float64')
        self.DR_LLNL_Cf252_mR_zAxis = self.DR_LLNL_Cf252_mR_df['zAxis (mR)'].to_numpy().astype('float64')
        # AmBe
        self.DR_LLNL_AmBe_mR_df = pd.read_csv('Data\\ExperimentalDRs\\DetRes_LLNL_AmBe_mR.csv')
        self.DR_LLNL_AmBe_mR_xAxis = self.DR_LLNL_AmBe_mR_df['xAxis (mR)'].to_numpy().astype('float64')
        self.DR_LLNL_AmBe_mR_yAxis = self.DR_LLNL_AmBe_mR_df['yAxis (mR)'].to_numpy().astype('float64')
        self.DR_LLNL_AmBe_mR_zAxis = self.DR_LLNL_AmBe_mR_df['zAxis (mR)'].to_numpy().astype('float64')
        # Godiva
        self.DR_LLNL_Godiva_TLD600_nC_df = pd.read_csv('Data\\ExperimentalDRs\\DetRes_LLNL_Godiva_TLD600_nC.csv')
        self.DR_LLNL_Godiva_TLD600_nC_xAxis = self.DR_LLNL_Godiva_TLD600_nC_df['xAxis (nC)'].to_numpy().astype('float64')
        self.DR_LLNL_Godiva_TLD600_nC_yAxis = self.DR_LLNL_Godiva_TLD600_nC_df['yAxis (nC)'].to_numpy().astype('float64')
        self.DR_LLNL_Godiva_TLD600_nC_zAxis = self.DR_LLNL_Godiva_TLD600_nC_df['zAxis (nC)'].to_numpy().astype('float64')
        # NIF shot 1
        self.DR_LLNL_NIF1_Net_mR_df = pd.read_csv('Data\\ExperimentalDRs\\DetRes_LLNL_NIF1_Net_mR.csv')
        self.DR_LLNL_NIF1_Net_mR_xAxis = self.DR_LLNL_NIF1_Net_mR_df['xAxis (mR)'].to_numpy().astype('float64')
        self.DR_LLNL_NIF1_Net_mR_yAxis = self.DR_LLNL_NIF1_Net_mR_df['yAxis (mR)'].to_numpy().astype('float64')
        self.DR_LLNL_NIF1_Net_mR_zAxis = self.DR_LLNL_NIF1_Net_mR_df['zAxis (mR)'].to_numpy().astype('float64')
    
    def load_dose_conversions(self):
        self.ANSI_DOSE_CONV = np.loadtxt('Data\\DoseConversion\\ANSI_N_13_3_DP10_pGy_per_fluence_cm2.txt')
        # H10 dose is in pSv*cm^2
        H10Dose_df = pd.read_csv('Data\\DoseConversion\\IAEA_H10Dose.csv',delimiter=',')
        H10Dose_IAEA = H10Dose_df.to_numpy()[:60,1].astype('float32')
        self.H10Dose_MCNP = np.interp(self.E_BINS_MCNP, self.E_BINS_IAEA, H10Dose_IAEA)#/self.IAEA_LETHARGIES)
    
    def calculate_lethargies(self, ebins, max_e):
        lethargies = np.zeros((len(ebins)))
        for i in range(len(lethargies)-1):
            lethargies[i] = np.log(ebins[i+1]/ebins[i])
        lethargies[-1] = np.log(max_e/ebins[-1])
        return lethargies
    
    def loadData_IAEA(self):
        self.IAEA_data = pd.read_pickle('Data\\IAEA_data\\iaea_data.pkl')
        num_spectra = len(self.IAEA_data)
        self.IAEA_spectra = np.zeros((num_spectra, len(self.E_BINS_IAEA))) # (251,60)
        for i in range(num_spectra):
            self.IAEA_spectra[i,:] = self.IAEA_data['Spectrum'][i]
    
    def load_fruit_spectra(self):
        spectra = np.load('Data\\FruitSpectra\\Fruit_1e6spec_MCNPebins.npy')
        return spectra