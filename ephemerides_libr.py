# -*- coding: utf-8 -*-
"""
Contains the functions to access to the SPICE repository of the desired ephemerides for planets and cpv. Provided in Solar
 System barycentre in J2000 ecliptic reference frame  are based on:
o	de430.bsp for planets
o	naif0012.tls for the leapsecond kernel and time frame information
o	codes_300ast_20100725.bsp for the asteroid (usual Jim Baer’s 300 asteroid ephemerides)
o	codes_300ast_20100725.tf necessary options file to read the asteroid ephems.
These files must be loaded at the beginning of every main file.
In here, the correction at -67 second (the inferred leapsecond from the analyses) is applied

"""

import spiceypy as spice
from astropy import units as u
#need to add a -67 seconds leapsecond


MAX_FILES = 5300
def check_unload_necessary():
    current_files = spice.ktotal('ALL')
    return current_files >= MAX_FILES - 10  # Unload if close to the limit

if check_unload_necessary():
        spice.unload('de430.bsp')
        spice.unload('naif0012.tls')
        spice.unload('codes_300ast_20100725.tf')


# def SV_sun(date_start):
#     #SUN
#     # sun=Ephem.from_horizons("Sun", date_start, plane=Planes.EARTH_EQUATOR)
#     sun=Ephem.from_horizons("Sun", date_start, plane=Planes.EARTH_ECLIPTIC)
#     pSUN=sun.rv(date_start)[0].to(u.km).value
#     vSUN=sun.rv(date_start)[1].to(u.km/u.s).value
    
    # return pSUN, vSUN


def SV_sun_405(date_start):

    # spice.furnsh('de430.bsp')
    # spice.furnsh('naif0012.tls')
    
    date_str = str(date_start-67*u.second)
    et = spice.str2et(date_str)
    
    sun_state, _ = spice.spkgeo(targ=10, et=et, ref='ECLIPJ2000', obs=0)
    pSUN = sun_state[0:3]
    vSUN = sun_state[3:6]
    
    # Unload the ephemeris kernel
    if check_unload_necessary():
        spice.unload('de430.bsp')
        # spice.unload('naif0012.tls')
        
        spice.furnsh('de430.bsp')
        # spice.furnsh('naif0012.tls')

    
    return pSUN, vSUN

# def SV_earth(date_start):
#     #EARTH
#     # ter=Ephem.from_horizons("399", date_start, plane=Planes.EARTH_EQUATOR)    #3 for earth-moon bary
#     ter=Ephem.from_horizons("399", date_start, plane=Planes.EARTH_ECLIPTIC)
#     pEA=ter.rv(date_start)[0].to(u.km).value
#     vEA=ter.rv(date_start)[1].to(u.km/u.s).value 
#     return pEA, vEA


def SV_earth_405(date_start):

    
    date_str = str(date_start-67*u.second)
    et = spice.str2et(date_str)
    
    sun_state, _ = spice.spkgeo(targ=399, et=et, ref='ECLIPJ2000', obs=0)
    pSUN = sun_state[0:3]
    vSUN = sun_state[3:6]
    
    if check_unload_necessary():
        # Unload the ephemeris kernel
        # start=time.time()
        spice.unload('de430.bsp')
        # spice.unload('naif0012.tls')
        
        spice.furnsh('de430.bsp')
        # spice.furnsh('naif0012.tls')
        # print(time.time()-start)

    
    return pSUN, vSUN


def SV_earthmoon_405(date_start):

    
    date_str = str(date_start-67*u.second)
    et = spice.str2et(date_str)
    
    sun_state, _ = spice.spkgeo(targ=3, et=et, ref='ECLIPJ2000', obs=0)
    pSUN = sun_state[0:3]
    vSUN = sun_state[3:6]
    
    if check_unload_necessary():
        spice.unload('de430.bsp')
        spice.furnsh('de430.bsp')
   
    return pSUN, vSUN

def SV_saturn_405(date_start):

    
    date_str = str(date_start-67*u.second)
    et = spice.str2et(date_str)
    
    sun_state, _ = spice.spkgeo(targ=6, et=et, ref='ECLIPJ2000', obs=0)
    pSUN = sun_state[0:3]
    vSUN = sun_state[3:6]
    
    if check_unload_necessary():
        spice.unload('de430.bsp')
        spice.furnsh('de430.bsp')
   
    return pSUN, vSUN

def SV_jupiter_405(date_start):

    
    date_str = str(date_start-67*u.second)
    et = spice.str2et(date_str)
    
    sun_state, _ = spice.spkgeo(targ=5, et=et, ref='ECLIPJ2000', obs=0)
    pSUN = sun_state[0:3]
    vSUN = sun_state[3:6]
    
    if check_unload_necessary():
        spice.unload('de430.bsp')
        spice.furnsh('de430.bsp')
   
    return pSUN, vSUN

def SV_mars_405(date_start):

    
    date_str = str(date_start-67*u.second)
    et = spice.str2et(date_str)
    
    sun_state, _ = spice.spkgeo(targ=4, et=et, ref='ECLIPJ2000', obs=0)
    pSUN = sun_state[0:3]
    vSUN = sun_state[3:6]
    
    if check_unload_necessary():
        spice.unload('de430.bsp')
        spice.furnsh('de430.bsp')
   
    return pSUN, vSUN

def SV_venus_405(date_start):

    
    date_str = str(date_start-67*u.second)
    et = spice.str2et(date_str)
    
    sun_state, _ = spice.spkgeo(targ=299, et=et, ref='ECLIPJ2000', obs=0)
    pSUN = sun_state[0:3]
    vSUN = sun_state[3:6]
    
    if check_unload_necessary():
        spice.unload('de430.bsp')
        spice.furnsh('de430.bsp')
   
    return pSUN, vSUN

def SV_mercury_405(date_start):

    
    date_str = str(date_start-67*u.second)
    et = spice.str2et(date_str)
    
    sun_state, _ = spice.spkgeo(targ=199, et=et, ref='ECLIPJ2000', obs=0)
    pSUN = sun_state[0:3]
    vSUN = sun_state[3:6]
    
    if check_unload_necessary():
        spice.unload('de430.bsp')
        spice.furnsh('de430.bsp')
   
    return pSUN, vSUN

def SV_uranus_405(date_start):

    
    date_str = str(date_start-67*u.second)
    et = spice.str2et(date_str)
    
    sun_state, _ = spice.spkgeo(targ=7, et=et, ref='ECLIPJ2000', obs=0)
    pSUN = sun_state[0:3]
    vSUN = sun_state[3:6]
    
    if check_unload_necessary():
        spice.unload('de430.bsp')
        spice.furnsh('de430.bsp')
   
    return pSUN, vSUN

def SV_neptune_405(date_start):

    
    date_str = str(date_start-67*u.second)
    et = spice.str2et(date_str)
    
    sun_state, _ = spice.spkgeo(targ=8, et=et, ref='ECLIPJ2000', obs=0)
    pSUN = sun_state[0:3]
    vSUN = sun_state[3:6]
    
    if check_unload_necessary():
        spice.unload('de430.bsp')
        spice.furnsh('de430.bsp')
   
    return pSUN, vSUN

def SV_pluto_405(date_start):

    
    date_str = str(date_start-67*u.second)
    et = spice.str2et(date_str)
    
    sun_state, _ = spice.spkgeo(targ=9, et=et, ref='ECLIPJ2000', obs=0)
    pSUN = sun_state[0:3]
    vSUN = sun_state[3:6]
    
    if check_unload_necessary():
        spice.unload('de430.bsp')
        spice.furnsh('de430.bsp')
   
    return pSUN, vSUN

def SV_ceres_405(date_start):

    
    date_str = str(date_start-67*u.second)
    et = spice.str2et(date_str)
    
    sun_state, _ = spice.spkgeo(targ=2000001, et=et, ref='ECLIPJ2000_DE405', obs=0)
    pSUN = sun_state[0:3]
    vSUN = sun_state[3:6]
    
    if check_unload_necessary():
        spice.unload('de430.bsp')
        spice.furnsh('de430.bsp')
   
    return pSUN, vSUN

def SV_pallas_405(date_start):

    
    date_str = str(date_start-67*u.second)
    et = spice.str2et(date_str)
    
    sun_state, _ = spice.spkgeo(targ=2000002, et=et, ref='ECLIPJ2000', obs=0)
    pSUN = sun_state[0:3]
    vSUN = sun_state[3:6]
    
    if check_unload_necessary():
        spice.unload('de430.bsp')
        spice.furnsh('de430.bsp')
   
    return pSUN, vSUN

def SV_vesta_405(date_start):

    
    date_str = str(date_start-67*u.second)
    et = spice.str2et(date_str)
    
    sun_state, _ = spice.spkgeo(targ=2000004, et=et, ref='ECLIPJ2000', obs=0)
    pSUN = sun_state[0:3]
    vSUN = sun_state[3:6]
    
    if check_unload_necessary():
        spice.unload('de430.bsp')
        spice.furnsh('de430.bsp')
   
    return pSUN, vSUN