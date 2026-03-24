# Soil Moisture Analysis for Bushfire Risk (Australia)

## Overview
This project analyzes NASA SMAP Level-4 soil moisture data to study land dryness across Australia. The goal is to extract and visualize surface and root zone soil moisture, which are important indicators for bushfire risk.

## Dataset
- Source: NASA SMAP (NSIDC DAAC)
- Files: 5 HDF5 datasets (Dec 2023 – Jan 2024)
- Resolution: 9 km grid

## What I Did
- Extracted soil moisture data (`sm_surface`, `sm_rootzone`) from HDF5 files  
- Filtered data for Australia using latitude and longitude  
- Cleaned and combined multiple datasets into a single CSV  
- Visualized spatial maps of soil moisture across Australia  
- Analyzed changes in soil moisture over time  

## Key Findings
- Inland Australia shows lower soil moisture (drier regions)  
- Coastal and eastern regions show higher moisture  
- Surface soil moisture varies more over time than root zone moisture  
- Lower soil moisture indicates higher potential bushfire risk  

## Why This Matters
Soil moisture is a key factor in bushfire conditions. Drier soil and vegetation increase the likelihood of fire ignition and spread. This dataset can be used as an input for future bushfire prediction models.

## Tech Stack
- Python  
- NumPy, Pandas  
- Matplotlib  
- h5py  

## Output
- Clean dataset: `australia_soil_moisture_dataset.csv`  
- Visualizations of soil moisture across Australia  
- Time-series analysis of soil moisture trends  

## Future Work
- Integrate weather data (temperature, wind, humidity)  
- Build a machine learning model for bushfire prediction  