# -*- coding: utf-8 -*-
"""


MD_library: functions called by the main automatic_MD to perform the asteroid mass determination
"""

import math
import numpy as np
import MD_auxiliary as auxy
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import threading


def daAUaKm(dist_AU):
    fatt_trans=149597870.691 
    return dist_AU*fatt_trans;

plt.ioff()


folder_output="Results/"
file_no_enc = folder_output + "no_encounter.txt"
file_one_enc = folder_output + "single_encounter.txt"
file_multi_enc = folder_output + "multi_encounters.txt"
file_outlier_pert =  folder_output + "outliers_unp.txt"
file_outlier_unpert = folder_output + "outliers_pert.txt"

#ADD IF NOT CREATED ALREADY
# auxy.create_file(file_no_enc)
# auxy.create_file(file_one_enc)
# auxy.create_file(file_multi_enc)
# auxy.create_file(file_outlier_pert)
# auxy.create_file(file_outlier_unpert)



local_database_folder="/mnt/d/local/orbit/database/folder/"
databaseFile="MPCORB.dat"
Data_File=open(databaseFile)
file_massive_ast = "MASSIVE_ASTS_ORBIT_DATABASE.txt"


environ_modified =  "/environ/environ_generico.dat"
environ_original =  "/environ/environ_generico_original.dat"
environ_all_perts =  "/environ/environ_all_perts.dat"
environ_unp =  "/environ/environ_unp_99999sigma.dat"


RMS_unp_treshold = 100
RMS_pert_threshold = 5
RMS_encounter = 0.07 
MOID_threshold = daAUaKm(0.2)
RMS_sensitivity = 0.001
oppositions_min = 2

vec_molt = [3.5,2,0.7]    

linea_inizio= 1147230  #Here to select the range of asteroids as per lines in the MPCORB.dat file
linea_fine = 1177484 
# search_gap= 
linea_inizio_eff=linea_inizio-2


def queing_files(file_queue):
    
    with open(databaseFile) as Data_File:
        line_origin=0;
        for x in Data_File:
               
                if line_origin > linea_fine:
                    break
                if line_origin > linea_inizio_eff:
                    
                    oppositions = int(x[123:127])
                    
                    if oppositions > oppositions_min:
                    
                    
                        AstNum_string=((x[166:174]).strip()).replace("(","").replace(")","")
                        
                        
                        if not AstNum_string == "":
                            
                            
                            AstNum=int(AstNum_string)
                            
                            Ast_folder=math.floor(AstNum/1000)
                            
                            Ast_fileName=(local_database_folder+"numbered/{:04d}/{}.rwo").format(Ast_folder,AstNum)
                          
                        
                        else:
                            
                            AstName=((x[166:193]).strip()).replace(" ","")
                           
                            
                            if "P-L" in AstName:
                               
                            
                                Ast_fileName=local_database_folder+"unnumbered/unusual/P-L/"+AstName+".rwo"
                                
                            else:
                                
                               
                                Ast_folder=AstName[0:5]
                                
                                Ast_fileName=local_database_folder+"unnumbered/"+Ast_folder+"/"+AstName+".rwo"
                        
                        #Need to change into if the file exists than put it in queue
                        if os.path.exists(Ast_fileName.replace("/mnt/d", "D:")):
                            file_queue.put(Ast_fileName)
                
                        
                line_origin+=1   
        
    
def CA_detection(file, wsl_name, folder_linux,subfolder):
    
    
    
    RMS_unp =  RMS_from_FO(wsl_name, file, subfolder, environ_unp, folder_linux)[1]
    
    if RMS_unp < RMS_unp_treshold :
        el_orb_perturbed, RMS_pert = RMS_from_FO(wsl_name, file, subfolder, environ_all_perts, folder_linux)
        
        if RMS_pert < RMS_pert_threshold:
            #check difference
             if RMS_unp - RMS_pert > RMS_encounter:
                 #there is an encounter, let's see in another function who is the responsible
                 CA_analysis(el_orb_perturbed,folder_linux,wsl_name,file,subfolder,RMS_unp)
             # else:
             #     #no encounter
        
        else:
            #outlier pert
           ast_name = file.rsplit("/", 1)[-1].split(".rwo")[0]

           auxy.write_to_file(file_outlier_pert, ast_name + "\n", mode='a')
    else:
        #outlier unpert
        ast_name = file.rsplit("/", 1)[-1].split(".rwo")[0]
      
        auxy.write_to_file(file_outlier_unpert, ast_name + "\n", mode='a')
    
    
    

def RMS_from_FO(wsl_name, file, subfolder, environ_file, folder_linux):
    
    el_orbs = np.zeros(5)
    
    wsl_command = f'wsl {wsl_name}fo "{file}" -D /home/{subfolder}/.find_orb/{environ_file} -c'
    file_RMS_computed=folder_linux+"\\elem_short.json"
    try:
        os.system(wsl_command)
    finally:
        file_orb_data = open(file_RMS_computed)
        j=0 
       
        try:
            for j,linea_orb_data in enumerate(file_orb_data):
                if j== 26:
                    el_orbs[0]=daAUaKm(float(linea_orb_data[15:25]))    #a
                if j== 27:
                    el_orbs[1]=float(linea_orb_data[15:25])    #e
                if j==30:
                    el_orbs[2]=math.radians(float(linea_orb_data[15:25]))    #i
                if j==31:
                        el_orbs[4]=math.radians(float(linea_orb_data[20:30]) )   #Om
                if j==32:
                        el_orbs[3]=math.radians(float(linea_orb_data[20:30]) )   #om
                
                if j== 38:
                    RMS_1=float(linea_orb_data[33:39])   #weighted rms residuals
                elif j>38:
                    break
       
                  
        finally:
            file_orb_data.close()
    
        return el_orbs,RMS_1   
    
    

    
    
def CA_analysis(el_orb_perturbed,folder_linux,wsl_name,file,subfolder,RMS_unpert):
    
    ast_name = file.rsplit("/", 1)[-1].split(".rwo")[0]
    perturbing_asteroids_per_asteroid = []
    orbital_positions_perturbed = auxy.orbital_3D_positions(el_orb_perturbed)
    
    file_mass_asts = open(file_massive_ast, 'r')
    DRES_vec = []
    try:
        for linea in file_mass_asts:
            pert_ast = (linea.split()[0])
            pert_ast_orb_elems = [float(x) for x in linea.split()[1:6]] 
            
            if auxy.filter_pert_asteroids(pert_ast,1) == 1:

        
        
                if auxy.MOID(orbital_positions_perturbed,  pert_ast_orb_elems) < 250*auxy.RH(pert_ast,pert_ast_orb_elems[0],folder_linux): 
                    # print(pert_ast)
                    # we have a generic environ file, all perts, high sigma thrshold,
                    # must change the line of the perturbing body with the number of the asteroid 
                    # here with low MOID
                    
                    auxy.modify_generic_environ(environ_modified,environ_original,pert_ast,folder_linux)
                    
                    Res_perturbed =  RMS_from_FO(wsl_name, file, subfolder, environ_modified, folder_linux)[1]
                    
                    if RMS_unpert - Res_perturbed > RMS_sensitivity:
                        
                        perturbing_asteroids_per_asteroid.append(pert_ast)
                        
                        iterative_mass_refinement(int(pert_ast),folder_linux,wsl_name,subfolder,file,RMS_unpert,Res_perturbed)
                        
                        DRES_vec.append(RMS_unpert - Res_perturbed)


            
    #Now that we have scanned all the perturbing bodies and seen before with the MOID and then
    #with the integration, who is pertubing, we now have to separate those that have
    #  1 actually no encounters
    #  2 just one encounter
    #  3 more than one encounter
    
    finally:
        file_mass_asts.close()
    
    auxy.save_encounter_data(perturbing_asteroids_per_asteroid,file_no_enc,file_one_enc,file_multi_enc,ast_name,DRES_vec)
        
    



    
def iterative_mass_refinement(perturber_numb,folder_linux,wsl_name,subfolder,file,Res_unp,Res_pert):
    ast_name = file.rsplit("/", 1)[-1].split(".rwo")[0]
    
    mass0 = auxy.read_mass(perturber_numb,folder_linux) 
    mass_original = mass0
    images = []
    
    i = 0
    while i < 3:
        
        value = vec_molt[i]
        res = []
        d_mass = value *  mass0 
        mass_vec = np.linspace(mass0-d_mass,mass0+d_mass,8)    
        
        for mass_trial in mass_vec :
            
            #we must modify the mass in mu
            
            auxy.modify_mass_file_mu1(perturber_numb, mass_trial,folder_linux)
            
            res.append(RMS_from_FO(wsl_name, file, subfolder, environ_original, folder_linux)[1])
            
            
        res=np.array(res)       
        
        #hereafter a test on the convergence must be proposed. R>0.9
        
        R = auxy.R_test(mass_vec,res,2)
        
        if R > 0.9:
            
            fig = plt.figure()      
            plt.scatter(mass_vec,res   ,  color='blue',marker = 'o')
            # plt.title(f"Iteration {i}")
            
            
            
            interpolated_func = interp1d(mass_vec,res, kind='quadratic', fill_value='extrapolate')
            
            
            
            
            mass_interp = np.linspace(mass_vec.min(), mass_vec.max(), 500)
            
            minimum_value = np.min(interpolated_func(mass_interp))
            minimum_x = mass_interp[np.argmin(interpolated_func(mass_interp))]
            
            # Plot the interpolated function and the scatter plot of the original data
            plt.figure
            plt.scatter(mass_vec, res, color='red', label='Data')
            plt.plot(mass_interp, interpolated_func(mass_interp), label='Interpolated Function')
            plt.xlabel('Mass')
            plt.ylabel('RMS')
            title = f"Interpolation of the mass determination for close approach between\n{ast_name} and {perturber_numb} iteration {i+1}"
            plt.text(0.5, 1.1, title, ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            
            
            x_min, x_max = plt.xlim()
            y_min, y_max = plt.ylim()
            
            
            subscript_x = (x_max + x_min) / 2
            subscript_y = (y_max + y_min) / 2
            vertical_offset = 0.1 * (y_max - y_min)  # Adjust the offset as needed
            
            # Add the subscript text
            plt.text(subscript_x, subscript_y - vertical_offset, f'$(RMS)_{{unpert}}-(RMS)_{{ast}}$ = {Res_unp-Res_pert:.4f}', fontsize=12,
                     ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
            
            plt.text(subscript_x, subscript_y + vertical_offset, f'$m_{i+1}$ = {minimum_x:.4e}', fontsize=12,
                     ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
            
            plt.text(subscript_x, subscript_y + 2* vertical_offset, f'$m_{i}$ = {mass0:.4e}', fontsize=12,
                     ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
            
            
            plt.legend()
            
            images.append(fig)
            i+=1
            mass0 = minimum_x
            
        else:
            
            
            break
        
    
    
    #record encounter happened
    
  
    
    if i > 0:
    
        auxy.save_file_data(minimum_x,perturber_numb,ast_name,mass_original,i,Res_unp-Res_pert)
        #depending on the value of i the whole package mass plus pictures will have to  be moved in the i converged alg.
        output_directory = f"Results/{perturber_numb}/{i}_iterations/{ast_name}"
        
        directory_creation_lock = threading.Lock()
        with directory_creation_lock:
            if not os.path.exists( output_directory):
                os.makedirs( output_directory)
        
        
        
        
        for i, image in enumerate(images):
            image_filename = os.path.join(output_directory, f"{ast_name}_{perturber_numb}_{i+1}_iteration.png")
            image.savefig(image_filename)
            
        plt.close('all')
    
    

    
    

        
        
    

   
