# -*- coding: utf-8 -*-
"""Library to get the Topocentric observations. 

(might have become obsolete)
"""

import csv
import numpy as np
from astropy.coordinates import Longitude, EarthLocation
from astropy import units as uts
import math
from astropy import units as u
from astropy.time import Time, TimezoneInfo
import datetime
import pytz

################################################################################################################

sin_obliq_2000 = 0.397777155931913701597179975942380896684
cos_obliq_2000 = 0.917482062069181825744000384639406458043
# rot_equat_to_eclip = rotation_matrix( obliquity_j2000, 'x') #rotation matrix from equatorial to ecliptic frames
# rot_eclip_to_equat = rotation_matrix(-obliquity_j2000, 'x')
rot_eclip_to_equat =  np.array([[1, 0, 0],  
                                [0, cos_obliq_2000, -sin_obliq_2000], 
                                [0, sin_obliq_2000, cos_obliq_2000]])

rot_equat_to_eclip = rot_eclip_to_equat.T

Re = 6378.140
################################################################################################################

def find_values(code):
    with open("MPC_obs_code.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Code"] == code:
                return {
                    "Longitude": row["Longitude"],
                    "Latitude": row["Latitude"],
                    "Rho_cos": row["rho_cos"],
                    "Rho_sin_phi": row["rho_sin_phi"]
                }
        return None
    

def utc_to_local(utc_time, obs_code):
    # Define Earth location based on longitude
    long_site = Longitude( float(find_values(obs_code)["Longitude"]), 
                          uts.degree, wrap_angle=360.0*uts.degree)
    # compute sidereal time of observation at site

    utc_datetime = utc_time.to_datetime()
    utc_datetime_utc = pytz.utc.localize(utc_datetime)
    
    # Convert to local time zone
    local_tz = pytz.timezone('UTC')  # Set to local time zone
    local_datetime = utc_datetime_utc.astimezone(local_tz)
    
    local_time
    
    return local_time


def observerpos_mpc(obs_code, t_utc):
    """Compute geocentric observer position at UTC instant t_utc, for Sun-centered orbits,
    at a given observation site defined by its longitude, and parallax constants S and C.
    Formula taken from top of page 266, chapter 5, Orbital Mechanics book (Curtis).
    The parallax constants S and C are defined by:
    S=rho cos phi' C=rho sin phi', where
    rho: slant range
    phi': geocentric latitude

       Args:
           long (float): longitude of observing site
           parallax_s (float): parallax constant S of observing site
           parallax_c (float): parallax constant C of observing site
           t_utc (astropy.time.Time): UTC time of observation

       Returns:
           1x3 numpy array: cartesian components of observer's geocentric position
    """
    # Earth's equatorial radius in kilometers
    # Re = cts.R_earth.to(uts.Unit('km')).value

    # define Longitude object for the observation site longitude
    long_site = Longitude( float(find_values(obs_code)["Longitude"]), uts.degree, wrap_angle=360.0*uts.degree)
    # compute sidereal time of observation at site
    t_site_lmst = t_utc.sidereal_time('mean', longitude=long_site)
    lmst_rad = t_site_lmst.rad # np.deg2rad(lmst_hrs*15.0) # radians

    # compute cartesian components of geocentric (non rotating) observer position
    x_gc = Re*float(find_values(obs_code)["Rho_cos"])*np.cos(lmst_rad)
    y_gc = Re*float(find_values(obs_code)["Rho_cos"])*np.sin(lmst_rad)
    z_gc = Re*float(find_values(obs_code)["Rho_sin_phi"])
    
    RS = np.array([x_gc,y_gc,z_gc])  #Geocentric - equatorial so we need to transform it
    
     
    return np.matmul(rot_equat_to_eclip, RS)


