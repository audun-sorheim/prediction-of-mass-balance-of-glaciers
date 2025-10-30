"""
Preparation of mass balance data for glaciers (as .csv)
and climate data from ERA5 (as .nc).
There should be an existing folder containing the data
at the same level as this script like this:
--data_prep.py
--data_folder(e.g. 'data_EU')
    --FoG_MB_1.csv
    --FOG_MB_2.csv
    ...
    --ERA5_EU.nc
The resulting pickle file containing the pandas dataframe
will be saved in the data folder as well.
Names for the data folder and the climate file can be changed
at lines 112 and 113.
"""

import os

import pandas as pd
import numpy as np
import xarray as xr


def get_one_glacier(data_folder, glacier_file, climate_file, t_th):
    """Get mass balance and climate data for one glacier"""

    print(glacier_file)
    # import mass balance data
    mb = pd.read_csv(
        data_folder + '/' + glacier_file, skiprows=8, encoding='latin',
        usecols=[
            'WGMS_ID', 'LATITUDE', 'LONGITUDE',
            'REFERENCE_YEAR', 'SURVEY_YEAR',
            'WINTER_BALANCE', 'SUMMER_BALANCE', 'ANNUAL_BALANCE'
        ]
    )

    # check if survey year follows reference year
    # if not: set values to NaN
    mb_check = mb
    mb = mb.where(mb.SURVEY_YEAR - 1 == mb.REFERENCE_YEAR)
    print('Survey Years with NaN in Annual Balance:')
    print(list(mb_check.SURVEY_YEAR.loc[mb.REFERENCE_YEAR.isna()]))

    # drop rows with NaNs in Annual Balance
    mb = mb.dropna(subset=['ANNUAL_BALANCE'])

    mb['SURVEY_YEAR'] = pd.to_datetime(mb.SURVEY_YEAR, format='%Y')
    mb = mb.set_index('SURVEY_YEAR')

    # import climate data with xarray
    clim_xr = xr.open_dataset(climate_file)
    clim_xr = clim_xr.sel(expver=1)

    # get coordinates of glacier
    coords = pd.Series(list(zip(mb.LATITUDE, mb.LONGITUDE)))
    coords = list(coords.unique())

    # choose climate data only for coordinates of glacier
    clim_xr = clim_xr.sel(latitude=coords[0][0], method='nearest')
    clim_xr = clim_xr.sel(longitude=coords[0][1], method='nearest')

    # convert to pandas dataframe
    clim = clim_xr.to_dataframe().drop(columns=['expver'])

    # reorder, so that we have columns for monthly temperatures and snowfall
    clim = clim.reset_index()
    clim['time'] = pd.to_datetime(clim['time'])
    clim['year'] = pd.to_datetime(clim['time'].dt.year, format='%Y')
    clim['month'] = clim['time'].dt.month

    # convert Kelvin to °C
    clim['d2m'] = clim['d2m'] - 273.15

    clim = clim.pivot_table(
        index='year',
        columns=['month'],
        values=['d2m', 'sf']
    )

    # rename columns
    clim.columns.name = None

    sf_names = ['SF_' + str(month) for month in np.arange(1, 13)]
    t_names = ['T_' + str(month) for month in np.arange(1, 13)]

    clim.columns = sf_names + t_names

    # calculate TMPP and Annual Snowfall
    clim[t_names] = clim[t_names].where(clim[t_names] > t_th, other=t_th)
    clim['TMPP'] = clim[t_names].sum(axis=1)
    clim['Annual_SF'] = clim[sf_names].sum(axis=1)

    # add mass balance data and glacier coordinates
    clim['MB_Summer'] = mb.WINTER_BALANCE
    clim['MB_Winter'] = mb.SUMMER_BALANCE
    clim['MB_Year'] = mb.ANNUAL_BALANCE
    clim['Lat'] = mb.LATITUDE
    clim['Lon'] = mb.LONGITUDE
    clim['Glacier_ID'] = mb.WGMS_ID

    # drop years for which no MB data is available
    clim = clim.dropna(subset='MB_Year')

    # index should not be based on date if we add more glaciers
    # (there may be data for two different glaciers in the same year)
    clim = clim.reset_index()

    print('------------------------------------------------------')

    return clim


# define some user-dependet variables
data_folder = 'data_EU/south_scandinavia'     # name of folder that contains climate (.nc) and glacier data (.csv)
climate_file = 'era5_monthly_Europe.nc'     # name of ERA5 climate data file
tmpp_th = 0     # treshold for positive degree month in °C

# get a list of the files in the data folder
climate_path = data_folder + '/' + climate_file
file_list = os.listdir(data_folder)
glacier_dfs = []

for glacier_file in file_list:
    if glacier_file.startswith('FoG_MB') and glacier_file.endswith('.csv'):  # only if MB file for glaciers
        df = get_one_glacier(data_folder, glacier_file, climate_path, tmpp_th)
        if df is not None:
            glacier_dfs.append(df)


# add all dataframes to one big dataframe
df = pd.concat(glacier_dfs, ignore_index=True)

# print some information about the dataframe
print('All glaciers dataframe:')
df.info()

# save dataframe
print('\n')
pickle_path = 'data_EU/EU_south_scandinavia_all_glaciers.pkl'
df.to_pickle(pickle_path)
print('Dataframe saved to \'' + pickle_path + '\'.')
