##################### 
# load modules
import time
#model_run_time = time.time()
#time.sleep(16000)
import numpy as np
import pylab
from joblib import Parallel, delayed
from all_classes import * 
from Main_code_callable import forward_model
import sys
import os
import shutil
import contextlib
####################

num_runs = 120 # Number of forward model runs
num_cores = 60 # For parallelization, check number of cores with multiprocessing.cpu_count()
os.mkdir('switch_garbage3')

#Choose planet
which_planet = "E" # Earth
#which_planet = "V" # Venus

if which_planet=="E":
    Earth_inputs = Switch_Inputs(print_switch = "n", speedup_flag = "n", start_speed=15e6 , fin_speed=100e6,heating_switch = 0,C_cycle_switch="y",Start_time=10e6)   
    Earth_Numerics = Numerics(total_steps = 3 ,step0 = 50.0, step1=10000.0 , step2=1e6, step3=1e5, step4=-999, tfin0=Earth_inputs.Start_time+10000, tfin1=Earth_inputs.Start_time+10e6, tfin2=4.4e9, tfin3=4.5e9, tfin4 = -999) # Standard Earth parameters, 0 - 4.5 Gyrs
    #Earth_Numerics = Numerics(total_steps = 3 ,step0 = 50.0, step1=10000.0 , step2=1e6, step3=1e5, step4=-999, tfin0=Earth_inputs.Start_time+10000, tfin1=Earth_inputs.Start_time+10e6, tfin2=6.2e9, tfin3=7.7e9, tfin4 = -999) # Modified Earth into future, 0 - 7.7 Gyrs

    ## PARAMETER RANGES ##
    #initial volatile inventories
    init_water = 10**np.random.uniform(20,22,num_runs) #nominal  model, for reproducing Fig. 2 and 3
    init_CO2 = 10**np.random.uniform(20,22,num_runs) #nominal  model, for reproducing Fig. 2 and 3
    #init_water = 10**np.random.uniform(22,23.5,num_runs) #Waterworld, for reproducing Fig. 5. 
    #init_CO2 = 10**np.random.uniform(20,22.5,num_runs) #Waterworld, for reproducing Fig. 5. 
    #init_water = 10**np.random.uniform(19.8,np.log10(5e20),num_runs) #Desertworld, for reproducing Fig. 7
    #init_CO2 = 10**np.random.uniform(19.5,np.log10(3e20),num_runs) #Desertworld, for reproducing Fig. 7
    init_O = np.random.uniform(2e21,6e21,num_runs) #free oxygen (controls mantle redox)
    #init_O = 10**np.random.uniform(np.log10(0.5e21),np.log10(1.5e21),num_runs) #reduced mantle sensitivity test

    #Weathering and ocean chemistry parameters
    Tefold = np.random.uniform(5,30,num_runs) #Te-folding for continental weathering
    alphaexp = np.random.uniform(0.1,0.5,num_runs) #CO2-dependence of weathering
    suplim_ar = 10**np.random.uniform(5,7,num_runs) #Weathering supply limit
    ocean_Ca_ar = 10**np.random.uniform(-4,np.log10(0.3),num_runs) #Ocean [Ca2+]
    ocean_Omega_ar = np.random.uniform(1.0,10.0,num_runs)  #Ocean saturation state
    
    #impact parameters (only used for impact sensitivty test)
    imp_coef = 10**np.random.uniform(11,14.5,num_runs)
    tdc = np.random.uniform(0.06,0.14,num_runs)
    
    #escape parameters
    mult_ar = 10**np.random.uniform(-2,2,num_runs) #governs transition from diffusion limited escape to XUV-limited escape
    mix_epsilon_ar = np.random.uniform(0.0,1.0,num_runs)  #governs energy that goes into O drag in XUV-limited regime
    Epsilon_ar = np.random.uniform(0.01,0.3,num_runs) #initial escape efficiency
    Tstrat_array = np.random.uniform(209.5,210.5,num_runs) #Stratospheric temperature
    #Tstrat_array = np.random.uniform(150,250,num_runs) # Sensitivity test to stratospheric temperature
    #Tstrat_array = np.random.uniform(199.5,200.5,num_runs) #Tstrat = 200 K

    #Albedo parameters
    Albedo_C_range = np.random.uniform(0.25,0.35,num_runs) #cold state albedo
    Albedo_H_range = np.random.uniform(0.0,0.30,num_runs) #hot state albedo
    for k in range(0,len(Albedo_C_range)): #make sure hot state less than cold state
        if Albedo_C_range[k] < Albedo_H_range[k]:
            Albedo_H_range[k] = Albedo_C_range[k]-1e-5 

    #Stellar evolution parameters
    #1.8, 6.2, 45.6 Tu et al. 2015 10,50,90% confidenc 
    Omega_sun_ar = 10**np.random.uniform(np.log10(1.8),np.log10(45),num_runs)
    tsat_sun_ar = (2.9*Omega_sun_ar**1.14)/1000 #XUV saturation time
    fsat_sun = 10**(-3.13)
    beta_sun_ar = 1.0/(0.35*np.log10(Omega_sun_ar) - 0.98)
    beta_sun_ar = 0.86*beta_sun_ar #conversion after equation 1, Tu et al.

    # Interior parameters
    offset_range = 10**np.random.uniform(1.0,3.0,num_runs) ## Mantle viscosity scalar
    heatscale_ar = np.random.uniform(0.33,3.0,num_runs) ## Radiogenic inventory relative to Earth
    Mantle_H2O_max_ar = 10**np.random.uniform(np.log10(0.5),np.log10(15.0),num_runs) # Max mantle water

    #Oxidation parameters
    surface_magma_frac_array = 10**np.random.uniform(-4,0,num_runs) #max surface area molten frac
    MFrac_hydrated_ar = 10**np.random.uniform(np.log10(0.001),np.log10(0.03),num_runs) #hydration efficiency
    dry_ox_frac_ac = 10**np.random.uniform(-4,-1,num_runs) #dry oxidation efficiency
    wet_oxid_eff_ar = 10**np.random.uniform(-3,-1,num_runs) #wet oxidatoin efficiency 
    

elif which_planet=="V":
    Venus_inputs = Switch_Inputs(print_switch = "n", speedup_flag = "n", start_speed=15e6 , fin_speed=100e6,heating_switch = 0,C_cycle_switch="y",Start_time=30e6)   
    Venus_Planet_inputs = Planet_inputs(RE = 0.9499, ME = 0.815, rc=0.9499*3.4e6, pm=4000.0, Total_Fe_mol_fraction = 0.06, Planet_sep=0.7, albedoC=0.3,albedoH=0.3)   
    Venus_Init_conditions = Init_conditions(Init_solid_H2O=0.0, Init_fluid_H2O=1.0*2.6e21 , Init_solid_O=0.0, Init_fluid_O=0.815*4e21, Init_solid_FeO1_5 = 0.0, Init_solid_FeO=0.0, Init_solid_CO2=0.0, Init_fluid_CO2 =  5e20)   
    Venus_Numerics = Numerics(total_steps = 2 ,step0 = 50.0, step1=10000.0 , step2=1e6, step3=-999, step4=-999, tfin0=Venus_inputs.Start_time+10000, tfin1=Venus_inputs.Start_time+30e6, tfin2=4.5e9, tfin3=-999, tfin4 = -999)

    ## PARAMETER RANGES ##
    #initial volatile inventories
    init_water = 10**np.random.uniform(20,22,num_runs)
    init_CO2 = 10**np.random.uniform(20,22,num_runs)
    init_O = np.random.uniform(2e21,6e21,num_runs)

    #Weathering and ocean chemistry parameters
    Tefold = np.random.uniform(5,30,num_runs)
    alphaexp = np.random.uniform(0.1,0.5,num_runs)
    suplim_ar = 10**np.random.uniform(5,7,num_runs)
    ocean_Ca_ar = 10**np.random.uniform(-4,-1,num_runs)
    ocean_Omega_ar = np.random.uniform(1.0,10.0,num_runs) 

    #impact parameters
    imp_coef = 10**np.random.uniform(11,14.5,num_runs)
    tdc = np.random.uniform(0.06,0.14,num_runs)

    #escape parameters
    mult_ar = 10**np.random.uniform(-2,2,num_runs)
    mix_epsilon_ar = np.random.uniform(0.0,1.0,num_runs)
    Epsilon_ar = np.random.uniform(0.01,0.3,num_runs)
    #Tstrat_array = np.random.uniform(209.5,210.5,num_runs)
    Tstrat_array = np.random.uniform(199.5,200.5,num_runs)

    #Albedo parameters
    Albedo_C_range = np.random.uniform(0.2,0.7,num_runs)
    Albedo_H_range = np.random.uniform(0.0001,0.3,num_runs)
    for k in range(0,len(Albedo_C_range)):
        if Albedo_C_range[k] < Albedo_H_range[k]:
            Albedo_H_range[k] = Albedo_C_range[k]-1e-5   

    #Stellar evolution parameters  
    #1.8, 6.2, 45.6 Tu et al. 2015 10,50,90% confidenc 
    Omega_sun_ar = 10**np.random.uniform(np.log10(1.8),np.log10(45),num_runs)
    tsat_sun_ar = (2.9*Omega_sun_ar**1.14)/1000
    fsat_sun = 10**(-3.13)
    beta_sun_ar = 1.0/(0.35*np.log10(Omega_sun_ar) - 0.98)
    beta_sun_ar = 0.86*beta_sun_ar #conversion after equation 1, Tu
 
    # Interior parameters
    offset_range = 10**np.random.uniform(1.0,3.0,num_runs) ## NEW FOR FUTURE
    heatscale_ar = np.random.uniform(0.33,3.0,num_runs)
    Mantle_H2O_max_ar = 10**np.random.uniform(np.log10(0.5),np.log10(15.0),num_runs) 

    #Oxidation parameters
    MFrac_hydrated_ar = 10**np.random.uniform(np.log10(0.001),np.log10(0.03),num_runs) 
    dry_ox_frac_ac = 10**np.random.uniform(-4,-1,num_runs)
    wet_oxid_eff_ar = 10**np.random.uniform(-3,-1,num_runs)
    surface_magma_frac_array = 10**np.random.uniform(-4,0,num_runs)

##Output arrays and parameter inputs to be filled:
inputs = range(0,len(init_water))
output = []

for zzz in inputs:
    ii = zzz
    
    if which_planet=="E":
        Earth_Planet_inputs = Planet_inputs(RE = 1.0, ME = 1.0, rc=3.4e6, pm=4000.0, Total_Fe_mol_fraction = 0.06, Planet_sep=1.0, albedoC=Albedo_C_range[ii], albedoH=Albedo_H_range[ii])   
        Earth_Init_conditions = Init_conditions(Init_solid_H2O=0.0, Init_fluid_H2O=init_water[ii] , Init_solid_O=0.0, Init_fluid_O=init_O[ii],Init_solid_FeO1_5 = 0.0, Init_solid_FeO=0.0, Init_solid_CO2=0.0, Init_fluid_CO2 = init_CO2[ii])   
        Sun_Stellar_inputs = Stellar_inputs(tsat_XUV=tsat_sun_ar[ii], Stellar_Mass=1.0, fsat=fsat_sun, beta0=beta_sun_ar[ii], epsilon=Epsilon_ar[ii] )
        MC_inputs_ar =  MC_inputs(esc_a=imp_coef[ii], esc_b=tdc[ii], esc_c = mult_ar[ii],esc_d = mix_epsilon_ar[ii],ccycle_a=Tefold[ii] , ccycle_b=alphaexp[ii], supp_lim =suplim_ar[ii], interiora =offset_range[ii], interiorb=MFrac_hydrated_ar[ii],interiorc=dry_ox_frac_ac[ii],interiord = wet_oxid_eff_ar[ii],interiore = heatscale_ar[ii], interiorf = Mantle_H2O_max_ar[ii],ocean_a=ocean_Ca_ar[ii],ocean_b=ocean_Omega_ar[ii],Tstrat = Tstrat_array[ii], surface_magma_frac = surface_magma_frac_array[ii]) 
        inputs_for_later = [Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar]

    elif which_planet=="V":    
        Venus_Planet_inputs = Planet_inputs(RE = 0.9499, ME = 0.815, rc=0.9499*3.4e6, pm=4000.0, Total_Fe_mol_fraction = 0.06, Planet_sep=0.7,albedoC=Albedo_C_range[ii], albedoH=Albedo_H_range[ii])   
        Venus_Init_conditions = Init_conditions(Init_solid_H2O=0.0, Init_fluid_H2O=init_water[ii] , Init_solid_O=0.0, Init_fluid_O=init_O[ii], Init_solid_FeO1_5 = 0.0, Init_solid_FeO=0.0, Init_solid_CO2=0.0, Init_fluid_CO2 = init_CO2[ii])   
        Sun_Stellar_inputs = Stellar_inputs(tsat_XUV=tsat_sun_ar[ii], Stellar_Mass=1.0, fsat=fsat_sun, beta0=beta_sun_ar[ii], epsilon=Epsilon_ar[ii] )
        MC_inputs_ar = MC_inputs(esc_a=imp_coef[ii], esc_b=tdc[ii],  esc_c = mult_ar[ii], esc_d = mix_epsilon_ar[ii],ccycle_a=Tefold[ii] , ccycle_b=alphaexp[ii],  supp_lim = suplim_ar[ii], interiora =offset_range[ii], interiorb=MFrac_hydrated_ar[ii],interiorc=dry_ox_frac_ac[ii],interiord = wet_oxid_eff_ar[ii],interiore = heatscale_ar[ii], interiorf = Mantle_H2O_max_ar[ii], ocean_a=ocean_Ca_ar[ii],ocean_b=ocean_Omega_ar[ii],Tstrat = Tstrat_array[ii], surface_magma_frac = surface_magma_frac_array[ii])
        inputs_for_later = [Venus_inputs,Venus_Planet_inputs,Venus_Init_conditions,Venus_Numerics,Sun_Stellar_inputs,MC_inputs_ar]
    
    sve_name = 'switch_garbage3/inputs4L%d' %ii
    np.save(sve_name,inputs_for_later)

def processInput(i):
    load_name = 'switch_garbage3/inputs4L%d.npy' %i
    try:
        if which_planet=="E": 
            print ('starting ',i)
            [Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar] = np.load(load_name,allow_pickle=True)
            outs = forward_model(Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar)
        elif which_planet =="V":  
            print ('starting ',i)
            [Venus_inputs,Venus_Planet_inputs,Venus_Init_conditions,Venus_Numerics,Sun_Stellar_inputs,MC_inputs_ar] = np.load(load_name,allow_pickle=True)
            outs = forward_model(Venus_inputs,Venus_Planet_inputs,Venus_Init_conditions,Venus_Numerics,Sun_Stellar_inputs,MC_inputs_ar) 
    except:
        try: # try again with slightly different numerical options
            if which_planet=="E": 
                [Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar] = np.load(load_name,allow_pickle=True)
                Earth_Numerics.total_steps = 9 #switches from RK23 to RK45
                outs = forward_model(Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar)
        except:
            print ('didnt work ',i)
            outs = []

    print ('done with ',i)
    return outs

Everything = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs) #Run paralllized code
input_mega=[] # Collect input parameters for saving
for kj in range(0,len(inputs)):
    print ('saving garbage',kj)
    load_name = 'switch_garbage3/inputs4L%d.npy' %kj
    input_mega.append(np.load(load_name,allow_pickle=True))

np.save('Earth_outputs',Everything)
np.save('Earth_inputs',input_mega)

shutil.rmtree('switch_garbage3')

