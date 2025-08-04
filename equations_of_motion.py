# -*- coding: utf-8 -*-
"""
Contains the equations of motion (to be integrated in SOL by a RK5 algorithm) for the 2 body and 3 body propagation.
Moreover, contains the function to include the third body perturbations from planets and cpv.
Contains the information (as a struct) of the third bodies that can be included:  planets and cpv.
. Teir position are read from ephemerides_libr.
"""

import numpy as np
import ephemerides_libr as EPHLIB
from ephemerides_libr import SV_sun_405
from astropy import units as u



bary = 0

################################################################################################################################

m_planets = np.array([1.660114153054348e-07  ,  2.447838287784771e-06  ,   3.040432648022641e-06  , 
                          3.227156037554996e-07 ,9.547919101886966e-04,2.858856727222416e-04, 4.366249662744965e-05,
                          5.151383772628673e-05 ,  7.350487833457740e-09, 
                          4.725582914451237e-10 , 1.018229468217943e-10, 1.302666122601934e-10   ])
codex_planets = ['1','2','3','4','5','6','7','8','c','p','v']

object_mapping = {
    '1': {'mass': m_planets[0],  'position_function': EPHLIB.SV_mercury_405},
    '2': {'mass': m_planets[1],  'position_function': EPHLIB.SV_venus_405},
    '3': {'mass': m_planets[2],  'position_function': EPHLIB.SV_earthmoon_405},
    '4': {'mass': m_planets[3],  'position_function': EPHLIB.SV_mars_405},
    '5': {'mass': m_planets[4],  'position_function': EPHLIB.SV_jupiter_405},
    '6': {'mass': m_planets[5],  'position_function': EPHLIB.SV_saturn_405},
    '7': {'mass': m_planets[6],  'position_function': EPHLIB.SV_uranus_405},
    '8': {'mass': m_planets[7],  'position_function': EPHLIB.SV_neptune_405},
    '9': {'mass': m_planets[8],  'position_function': EPHLIB.SV_pluto_405},
    'c': {'mass': m_planets[9],  'position_function': EPHLIB.SV_ceres_405},
    'p': {'mass': m_planets[10], 'position_function': EPHLIB.SV_pallas_405},
    'v': {'mass': m_planets[11], 'position_function': EPHLIB.SV_vesta_405},
    
}


################################################################################################################################

def daKmaAU(dist_km):
    fatt_trans=149597870.691 
    return dist_km/fatt_trans;


def include_thrown_in_planets(r, pert_list = []):
    #If pert_list not provided, the fucntion includes all the planets without caring of the perturbation
    M_central_body = 0 
    m_planets = np.array([1, 1.660114153054348e-07  ,  2.447838287784771e-06  ,   3.040432648022641e-06 , 
                          3.227156037554996e-07 ,9.547919101886966e-04,2.858856727222416e-04, 4.366249662744965e-05,
                          5.151383772628673e-05 ,  7.350487833457740e-09   ])
    dist_planets_au = np.array([0., .38709927, .72333566, 1.00000261,
                  1.52371034, 5.20288799, 9.53667594,  19.18916464,  30.06992276])
    
    r = daKmaAU(r)
    
    for i, dist in enumerate (dist_planets_au):
        if r > 1.2*dist and str(i) not in pert_list:
            M_central_body += m_planets[i]
        elif r > dist and str(i) not in pert_list:
            
            f =    ((1.2-r/dist)/0.2)  #1-  #this equation has to be corrected on top of the find_orb version used to do orbit determination with 1-((1.2-r/dist)/0.2)
            M_central_body += m_planets[i] * f
            # print(r/dist)
            # print(f)
    
    return M_central_body



################################################################################################################################


def fun_2body(t, z ,G,masses,epoch0, perturbers_vec = []):
   x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, vx2, vy2, vz2 = z
   
   
   
   if len(perturbers_vec) > 0:
       aPx, aPy, aPz = third_body_pert(t, epoch0, [x1, y1, z1], G, masses[1], perturbers_vec)
   else:
       aPx = 0
       aPy = 0
       aPz = 0
   

   # Compute the distances between bodies
   r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
   
   M_central_body = include_thrown_in_planets(r12, perturbers_vec) * masses[1]   #ssun is [1] 


   # Compute the accelerations
   ax1 = G * M_central_body * (x2 - x1) / r12**3 + aPx
   ay1 = G * M_central_body * (y2 - y1) / r12**3 + aPy
   az1 = G * M_central_body * (z2 - z1) / r12**3 + aPz

   ax2 = G * masses[0] * (x1 - x2) / r12**3 
   ay2 = G * masses[0] * (y1 - y2) / r12**3 
   az2 = G * masses[0] * (z1 - z2) / r12**3
   

   # Return the velocities and accelerations
   return [vx1, vy1, vz1, vx2, vy2, vz2, ax1, ay1, az1, ax2, ay2, az2]



def fun_3body(t, z ,G,masses,epoch0, perturbers_vec = []):
     x1, y1, z1, x2, y2, z2, x3, y3, z3, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3 = z
    
     # print([x1,y1,z1],epoch0 + t*u.second)
     
     if len(perturbers_vec) > 0:
         aPx1, aPy1, aPz1, aPx2, aPy2, aPz2 = third_body_pert(t, epoch0, [x1, y1, z1, x2, y2, z2], G, masses[2], perturbers_vec)

     else:
         aPx1 = 0
         aPy1 = 0
         aPz1 = 0
         aPx2 = 0
         aPy2 = 0
         aPz2 = 0
     
    
    
     # Compute the distances between bodies
     r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
     r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2)
     r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2)
     
     M_central_body = include_thrown_in_planets(r13, perturbers_vec) * masses[2]   #Sun is [2] 
    
     # Compute the accelerations
     ax1 = G * masses[1] * (x2 - x1) / r12**3 + G * M_central_body * (x3 - x1) / r13**3 + aPx1     
     ay1 = G * masses[1] * (y2 - y1) / r12**3 + G * M_central_body * (y3 - y1) / r13**3 + aPy1   
     az1 = G * masses[1] * (z2 - z1) / r12**3 + G * M_central_body * (z3 - z1) / r13**3 + aPz1   
    
     ax2 = G * masses[0] * (x1 - x2) / r12**3 + G * M_central_body * (x3 - x2) / r23**3 + aPx2
     ay2 = G * masses[0] * (y1 - y2) / r12**3 + G * M_central_body * (y3 - y2) / r23**3 + aPy2
     az2 = G * masses[0] * (z1 - z2) / r12**3 + G * M_central_body * (z3 - z2) / r23**3 + aPz2 
    
     ax3 =  G * masses[0] * (x1 - x3) / r13**3 + G * masses[1] * (x2 - x3) / r23**3
     ay3 =  G * masses[0] * (y1 - y3) / r13**3 + G * masses[1] * (y2 - y3) / r23**3
     az3 =  G * masses[0] * (z1 - z3) / r13**3 + G * masses[1] * (z2 - z3) / r23**3
     
    
     # Return the velocities and accelerations
     return [vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3]
 
    
################################################################################################################################
    
def third_body_pert(t, epoch0, z, G, M_sun, perturbers_vec):
    aPx = 0
    aPy = 0
    aPz = 0
    if len(z) > 3:
        x1, y1, z1, x2, y2, z2 = z
        aP2x = 0
        aP2y = 0
        aP2z = 0
    else:
        x1, y1, z1 = z
    for pert in perturbers_vec:
        #Here I should add the perturbing effects on x,y,z
        if pert in object_mapping:
            object_info = object_mapping[pert]
            mass_P = object_info['mass'] * M_sun 
            position_function = object_info['position_function']
            xP, yP, zP = position_function(epoch0 + t*u.second)[0] - SV_sun_405(epoch0 + t*u.second)[0] 
            
    
            # r1P = np.sqrt((xP - x1)**2 + (yP - y1)**2 + (zP - z1)**2)
            # t_light = r1P/c_light
            # xP, yP, zP = position_function(epoch0 + (t-t_light)*u.second)[0] - SV_sun_405(epoch0 + (t-t_light)*u.second)[0] #relativistic-effect?
            R_pert = np.linalg.norm(np.array([xP,yP,zP]))
            
            r1P = np.sqrt((xP - x1)**2 + (yP - y1)**2 + (zP - z1)**2)
            
            aPx += G * mass_P * ((xP - x1) / r1P**3  - xP/R_pert**3)
            aPy += G * mass_P * ((yP - y1) / r1P**3  - yP/R_pert**3)
            aPz += G * mass_P * ((zP - z1) / r1P**3  - zP/R_pert**3)
            
            if len(z) > 3:
                r2P = np.sqrt((xP - x2)**2 + (yP - y2)**2 + (zP - z2)**2)
                
                aP2x += G * mass_P * ((xP - x2) / r2P**3  - xP/R_pert**3)
                aP2y += G * mass_P * ((yP - y2) / r2P**3  - yP/R_pert**3)
                aP2z += G * mass_P * ((zP - z2) / r2P**3  - zP/R_pert**3)
                
    if len(z) > 3:        
        return aPx, aPy, aPz, aP2x, aP2y, aP2z
    else:
        return aPx, aPy, aPz,
            
            