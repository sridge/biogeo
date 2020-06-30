import gsw
import xarray as xr
import numpy as np
import pandas as pd
import cbsyst as cb
import scipy.optimize.nnls as scinnls
# import config

'''
S:        salinity (PSU)
theta:    potential temperature (degrees C)

all biogeochemical tracers in units of umol/kg

AOU:      apparent oxygen utilization
O:        oxygen
N:        nitrate
P:        phosphate
Si:       silicate
DIC:      dissolved inorganic carbon
DIC_pre:  preformed DIC
Alk:      total alkalinity
Alk_pre:  preformed alkalinity
'''

# Redfield Ratio
R_on = 9
R_op = 135
R_np = 16
R_co = 0.69

def aou(O,theta,S):
	"""
    Apparent Oxygen Utilization (AOU). This is the difference between
    oxygen concentration at saturation and the measured oxygen 
    concentration. Origninal MATLAB script witten by Edward T. Peltzer,
    MBARI
    
    Parameters
    ----------
    O : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        Oxygen (umol/kg)
    theta : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        Potential temperature (Celsius)
    S : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        Salinity (pss-78)
        
    Returns
    -------
    AOU : `numpy.array` or `xarray.DataArray`
        An array of AOU values
    """
    
    T_1 = (theta + 273.15)/100
    
    O_sat = -177.7888 + 255.5907/T_1 + 146.4813*np.log(T_1) - 22.2040*T_1
    O_sat = O_sat + S*(-0.037362 + T_1*(0.016504 - 0.0020564*T_1))
    O_sat = np.exp(O_sat)

    O_sat = O_sat * 1000/22.392; #convert from ml/kg to umol/kg

    AOU = O_sat - O

    if (type(AOU) == pd.core.series.Series) or (type(AOU) == xr.core.dataarray):

        AOU.name = 'AOU'
    
    return AOU

def PO(P,O):
	"""
	Phosphate based conservative ocean circulation tracer (Broecker 1974)
    
    Parameters
    ----------
    P : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        Phosphate (units must be consistent with O)
    O : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        Oxygen (units must be consistent with P)
        
    Returns
    -------
    PO : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        An array of PO values
    """

    PO = R_op*P + O

    if (type(PO) == pd.core.series.Series) or (type(PO) == xr.core.dataarray):

        NO.name = 'PO'

    return PO

def NO(N,O):
	"""
	Nitrate based ocean circulation tracer (Broecker 1974). At the time
	of publication, production of nitrate by marine organisms was 
	unknown, thus this tracer is referred to as conservative. The
	interior biological source makes this tracer non-conservative
    
    Parameters
    ----------
    N : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        Nitrate (units must be consistent with O)
    O : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        Oxygen (units must be consistent with P)
        
    Returns
    -------
    NO : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        An array of NO values
    """

    NO = R_on*N + O

    if (type(NO) == pd.core.series.Series) or (type(NO) == xr.core.dataarray):

        NO.name = 'NO'

    return NO

def N_pre(N,AOU):
	"""
	Preformed Nitrate. The effects of post-subduction biologically 
	driven change in nitrate concentration have been removed.
    
    Parameters
    ----------
    N : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        Nitrate (units must be consistent with O)
    AOU : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        Apparent Oxygen Utilization (units must be consistent with N)
        
    Returns
    -------
    N_pre : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        An array of preformed nitrate values
    """

    N_pre = N-AOU/R_on

    if (type(N_pre) == pd.core.series.Series) or (type(N_pre) == xr.core.dataarray):

        N_pre.name = 'N˚'

    return N_pre

def P_pre(P,AOU):
    """
	Preformed Phosphate. The effects of post-subduction biologically 
	driven change in phosphate concentration have been removed.
    
    Parameters
    ----------
    P : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        Phosphate (units must be consistent with O)
    AOU : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        Apparent Oxygen Utilization (units must be consistent with P)
        
    Returns
    -------
    P_pre : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        An array of preformed phosphate values
    """

    P_pre = P-AOU/R_op

    if (type(P_pre) == pd.core.series.Series) or (type(P_pre) == xr.core.dataarray):

        P_pre.name = 'P˚'

    return P_pre

def Nstar(N,P):
	"""
	N* is indicative of nitrogen fixation. See Gruber and Sarmiento (1997, 2002)
    
    Parameters
    ----------
    N : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        Nitrate (umol/kg)
    P : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        Phosphate (umol/kg)
        
    Returns
    -------
    Nstar : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        An array of N* values
    """

    Nstar = N-R_np*P + 2.9

    if (type(Nstar) == pd.core.series.Series) or (type(Nstar) == xr.core.dataarray):

        Nstar.name = 'N*'

    return Nstar


def dc_dis(theta,S,N,P,O,dc_dis_eomp=np.NaN):

	"""
	N* is indicative of nitrogen fixation. See Gruber and Sarmiento (1997, 2002)
    
    Parameters
    ----------
    N : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        Nitrate (umol/kg)
    P : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        Phosphate (umol/kg)
        
    Returns
    -------
    Nstar : `numpy.array` or `xarray.DataArray` or `pd.core.series.Series`
        An array of N* values
    """
    
    a = np.ones(theta.shape)
    b = np.ones(theta.shape)
    c = np.ones(theta.shape)
    d = np.ones(theta.shape)
    e = np.ones(theta.shape)

    deep_theta_range = ((theta >= 5) & (theta < 8))
    mid_theta_range = ((theta >= 8) & (theta < 18))
    surf_theta_range = ((theta >= 18) & (theta <= 25))
    
    a[np.where(deep_theta_range)] = -7.1
    a[np.where(mid_theta_range)] = -13.4
    a[np.where(surf_theta_range)] = -38.6
    
    b[np.where(deep_theta_range)] = 1.29
    b[np.where(mid_theta_range)] = 1.18
    b[np.where(surf_theta_range)] = 1.67

    c[np.where(deep_theta_range)] = 11.1
    c[np.where(mid_theta_range)] = 0
    c[np.where(surf_theta_range)] = 16.3
    
    d[np.where(deep_theta_range)] = 0
    d[np.where(mid_theta_range)] = 0
    d[np.where(surf_theta_range)] = -0.32

    e[np.where(deep_theta_range)] = 0
    e[np.where(mid_theta_range)] = 0.17
    e[np.where(surf_theta_range)] = 0.52

    dc_dis = a + b*(theta-10) + c*(S-35) + d*(NO(N,O)-300) + e*(PO(P,O)-300)

    if (type(dc_dis) == pd.core.series.Series) or (type(dc_dis) == xr.core.dataarray):

        dc_dis.loc[theta<5] = dc_dis_eomp[np.where(theta<5)]

        dc_dis.name = '∆C_diseq'

    else:

        dc_dis[np.where(theta<5)] = dc_dis_eomp[np.where(theta<5)]
    
    return dc_dis

def dcstar(DIC,S,Alk,P,O,theta,AOU,Alk_pre_eomp=np.nan,N=np.nan,Si=np.nan,Gruber=True):
    
    if Gruber == True:

        Alk_pre_calc = 367.5 + 54.9*S + 0.074*PO(P,O)

        out = cb.Csys(fCO2=280,TA=Alk_pre_calc.ravel(),T=theta.ravel(),S=S.ravel())

        DICpi_eq = np.reshape(out.DIC,DIC.shape)

        dcstar = DIC-R_co*AOU-0.5*(Alk-Alk_pre_calc+(16/170)*(AOU))-DICpi_eq
        
    else:

        Alk_pre_calc = Alk_pre(N,O,P,Si,S,theta,AOU,Alk_pre_eomp)

        out = cb.Csys(fCO2=280,TA=Alk_pre_calc.ravel(),T=theta.ravel())

        DICpi_eq = np.reshape(out.DIC,DIC.shape)
        
        dcstar = DIC-R_co*AOU-dCa(N,O,P,Si,Alk,S,theta)-DICpi_eq 

    if (type(dcstar) == pd.core.series.Series) or (type(dcstar) == xr.core.dataarray):

        dcstar.name = '∆C*'       
         
    return dcstar

def csat_ant(S,theta):
    
    csat_ant = (S/35)*(0.85*theta+46.0)

    if (type(csat_ant) == pd.core.series.Series) or (type(csat_ant) == xr.core.dataarray):

        csat_ant.name = 'Csat_ant' 
    
    return csat_ant


def Alk_pre(N,O,P,Si,S,theta,AOU,Alk_pre_eomp=np.NaN):

    PA_pre = 587.7 + 46.2*S + 3.27*theta + 0.24*NO(N,O) + 0.73*Si

    Alk_pre = PA_pre-(N_pre(N,AOU)+P_pre(P,AOU))

    if (type(Alk_pre) == pd.core.series.Series) or (type(Alk_pre) == xr.core.dataarray):

        Alk_pre.loc[theta<5] = Alk_pre_eomp[np.where(theta<5)]
        Alk_pre.name = 'Alk˚'

    else:

        Alk_pre[np.where(theta<5)] = Alk_pre_eomp[np.where(theta<5)]

    return Alk_pre

def dCa(N,O,P,Si,Alk,S,theta):

    PA_pre = 587.7 + 46.2*S + 3.27*theta + 0.24*NO(N,O) + 0.73*Si

    PA = Alk + N + P

    dCa = 0.5*(PA-PA_pre)

    if (type(dCa) == pd.core.series.Series) or (type(dCa) == xr.core.dataarray):

        dCa.name = '∆Ca'
    
    return dCa

def DIC_pre(DIC,Alk,N,O,P,Si,S,theta,AOU):

    DIC_pre = DIC-R_co*AOU-dCa(N,O,P,Si,Alk,S,theta)

    if (type(DIC_pre) == pd.core.series.Series) or (type(DIC_pre) == xr.core.dataarray):

        dCa.name = 'DIC˚'
   
    return DIC_pre

def eOMP(theta,S,Si,O,N,P,AOU):
    
    Alk_pre_eomp = np.zeros(theta.size)
    dc_dis_eomp = np.zeros(theta.size)

    theta_A = np.array((5, -1.1, 5, -1.7, 1.5, -0.7))
    S_A = np.array((35.20, 34.88, 33.90, 34.00, 34.70, 34.65))
    Si_A = np.array((11, 6, 16, 72, 106, 160))
    O_A = np.array((307, 358, 309, 366, 336, 355))
    N_A = np.array((10.5, 9.8, 20.6, 37.0, 15.1, 20.5))
    P_A = np.array((0.63, 0.67, 1.43, 2.60, 1.07, 1.48))
    
    Alk_pre_A = np.array((2310, 2286, 2282, 2344, 2319, 2347))
    dc_dis_A = np.array((-7, -21, 0, -5, -23, -30))

    A = np.ones((7,6))
    A[0,:] = theta_A
    A[1,:] = S_A
    A[2,:] = Si_A

    nanmask = theta + S + Si + O + N + P + AOU

    for ind in range(0,theta.size):

        A[3,:] = O_A-AOU[ind]
        A[4,:] = N_A+AOU[ind]/R_on
        A[5,:] = P_A+AOU[ind]/R_op 
        
        b = np.array((theta[ind],S[ind],Si[ind],O[ind],N[ind],P[ind],1))

        W = np.array([8,3,2,1.6,1.7,1.5,100])
        W = np.sqrt(np.diag(W))
        Aw = np.dot(W,A)
        bw = np.dot(b,W)
        
        if ~np.isnan(nanmask)[ind]:
            
            x,R=scinnls(Aw,bw)

            Alk_pre_eomp[ind] = np.sum(Alk_pre_A*x)
            dc_dis_eomp[ind] = np.sum(dc_dis_A*x)
            
        else:
            
            Alk_pre_eomp[ind] = np.nan
            dc_dis_eomp[ind] = np.nan  
        
    return Alk_pre_eomp,dc_dis_eomp

def phiCT(theta,S,Si,O,N,P,Alk,DIC):

    AOU = aou(O,theta,S)
    
    Alk_pre_eomp,dc_dis_eomp = eOMP(theta,S,Si,O,N,P,AOU)

    cstar_phi = dcstar(AOU=AOU,Alk=Alk,DIC=DIC,O=O,P=P,S=S,theta=theta,N=N,Si=Si,Alk_pre_eomp=Alk_pre_eomp,Gruber=False)

    dcdis = dc_dis(theta,S,N,P,O,dc_dis_eomp)

    Cant = (cstar_phi-dcdis)/(1+0.55*(np.abs(dcdis)/csat_ant(S=S,theta=theta)))

    if (type(Cant) == pd.core.series.Series) or (type(Cant) == xr.core.dataarray):

        Cant.name = 'Cant'

    return Cant