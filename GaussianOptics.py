# %%
import numpy as np
#import matplotlib.pyplot as plt

#def champOptics(win, din, f1, f2, Lm, S):
     
     



def Gaussian_propagation2d(w0,z,Lambda):
    # confocal distance, in grasp it is denoted by 'b'
    z_c= np.pi*w0**2/Lambda
    # beam radius at distance of z
    w_z=w0*np.sqrt(1+(z/z_c)**2)
    # radius of wavefront curvature at z
    if z==0:
        R_z=10.0**100
    else:
        R_z=z+z_c**2/z
    # beam parameter
    q=z+1j*z_c
    # phasor
    phi=np.arctan(z/z_c)

    # beamsize
    theta=Lambda/(np.pi*w0)

    return z_c, w_z, R_z, q, phi

def ThinLens(win, din, f, Lambda):
    # para:  'f' or 'din'
    # para:  
    z_c, w_z, Rin, q1, phi1=Gaussian_propagation2d(win,din,Lambda)
    Mirror_para={'Lambda':Lambda,
                 'z_c_in': z_c,
                 'z_c_out':None,
                 'f':f,
                 'win':win,'din':din,'Rin':Rin,'q1':q1, 'phi1': phi1,
                 'wout':None,'dout':None,'Rout':0,'q2': 0, 'phi2': None,
                 'M':0,
                 'M_max':f/z_c,
                 'w_mirror':win*np.sqrt(1+(din/z_c)**2)}
  
    M_max=f/z_c
    factor=(din/f-1)**2+(z_c/f)**2
    dout=f*(1+(din/f-1)/factor)
    wout=win/np.sqrt(factor)
    if f==Rin:
         Rout=10.0**100
    else:
         Rout=1/(1/f-1/Rin)
    M=wout/win
    z_out_c=np.pi*wout**2/Lambda
    q2=dout+1j*z_out_c

    Mirror_para['wout']=wout
    Mirror_para['dout']=dout
    Mirror_para['Rout']=Rout
    Mirror_para['q2']=q2
    Mirror_para['phi2']=np.arctan(dout/z_out_c)
    Mirror_para['M']=M
    Mirror_para['z_c_out']=np.pi*wout**2/Lambda
    #for key in Mirror_para.keys():
    #    print(key,':',Mirror_para[key])
    return Mirror_para



def ThinLens_f_M(win, wout, f, Lambda):
        '''given focal length and magnification'''
        M=wout/win
        z_c_in = np.pi*win**2/Lambda
        z_c_out= np.pi*wout**2/Lambda
        Mirror_para1={'Lambda':Lambda,
                        'z_c_in': z_c_in,
                        'z_c_out':None,
                        'f':f,
                        'win':win,'din':None,'Rin':None,'q1':None, 'phi1':None,
                        'wout':wout,'dout':None,'Rout':None,'q2': None, 'phi2': None,
                        'M':M}
        Mirror_para2={'Lambda':Lambda,
                        'z_c_in': z_c_in,
                        'z_c_out':None,
                        'f':f,
                        'win':win,'din':None,'Rin':None,'q1':None, 'phi1':None,
                        'wout':wout,'dout':None,'Rout':None,'q2': None, 'phi2': None,
                        'M':M}
        f0=np.pi*win*wout/Lambda
        # pos
        din=f+1/M*np.sqrt(f**2-f0**2);Rin=din*(1+(np.pi*win**2/Lambda/din)**2)
        dout=f+M*np.sqrt(f**2-f0**2);Rout=1/(1/f-1/Rin)
        q1=din+1j*z_c_in
        q2=dout+1j*z_c_out
        phi1=np.arctan(din/z_c_in)
        phi2=np.arctan(dout/z_c_out)
        Mirror_para1['din']=din
        Mirror_para1['Rin']=Rin
        Mirror_para1['dout']=dout
        Mirror_para1['Rout']=Rout
        Mirror_para1['q1']=q1
        Mirror_para1['q2']=q2
        Mirror_para1['phi1']=phi1
        Mirror_para1['phi2']=phi2
        Mirror_para1['z_c_out']=np.pi*wout**2/Lambda

        # neg
        din=f-1/M*np.sqrt(f**2-f0**2);Rin=din*(1+(np.pi*win**2/Lambda/din)**2)
        dout=f-M*np.sqrt(f**2-f0**2);Rout=1/(1/f-1/Rin)
        q1=din+1j*z_c_in
        q2=dout+1j*z_c_out
        phi1=np.arctan(din/z_c_in)
        phi2=np.arctan(dout/z_c_out)
        Mirror_para2['din']=din
        Mirror_para2['Rin']=Rin
        Mirror_para2['dout']=dout
        Mirror_para2['Rout']=Rout
        Mirror_para2['q1']=q1
        Mirror_para2['q2']=q2
        Mirror_para2['phi1']=phi1
        Mirror_para2['phi2']=phi2
        Mirror_para2['din']=din
        Mirror_para2['dout']=dout
        Mirror_para2['z_c_out']=np.pi*wout**2/Lambda
        for key in Mirror_para1.keys():
             print(key,':',Mirror_para1[key])
        for key in Mirror_para2.keys():
             print(key,':',Mirror_para2[key])
        return Mirror_para1, Mirror_para2

def ThinLens_d_M(win, wout, d, Lambda):
        '''give din and Magnification'''
        M=wout/win
        z_c_in = np.pi*win**2/Lambda
        z_c_out= np.pi*wout**2/Lambda
        f0=np.pi*win*wout/Lambda
        Mirror_para1={'Lambda':Lambda,
                        'f':None,
                        'z_c_in': z_c_in,
                        'z_c_out':z_c_out,
                        'win':win,'din':None,'Rin':None,'q1':None, 'phi1':None,
                        'wout':wout,'dout':None,'Rout':None,'q2': None, 'phi2': None,
                        'M':M}
        Mirror_para2={'Lambda':Lambda,
                        'f':None,
                        'z_c_in': z_c_in,
                        'z_c_out':z_c_out,
                        'win':win,'din':None,'Rin':None,'q1':None, 'phi1':None,
                        'wout':wout,'dout':None,'Rout':None,'q2': None, 'phi2': None,
                        'M':M}
        if M!=1.0:
            f=(np.sqrt((M-1/M)**2*f0**2+d**2)*(M+1/M)-2*d)/(M-1/M)**2
            din=(d-f*(1-M**2))/(1+M**2);Rin=din*(1+(np.pi*win**2/Lambda/din)**2)
            dout=(M**2*d+f*(1-M**2))/(1+M**2);Rout=1/(1/f-1/Rin)
            q1=din+1j*z_c_in
            q2=dout+1j*z_c_out
            phi1=np.arctan(din/z_c_in)
            phi2=np.arctan(dout/z_c_out)
            Mirror_para1['f']=f
            Mirror_para1['din']=din
            Mirror_para1['Rin']=Rin
            Mirror_para1['dout']=dout
            Mirror_para1['Rout']=Rout
            Mirror_para1['q1']=q1
            Mirror_para1['q2']=q2
            Mirror_para1['phi1']=phi1
            Mirror_para1['phi2']=phi2

            f=(-np.sqrt((M-1/M)**2*f0**2+d**2)*(M+1/M)-2*d)/(M-1/M)**2
            din=(d-f*(1-M**2))/(1+M**2);Rin=din*(1+(np.pi*win**2/Lambda/din)**2)
            dout=(M**2*d+f*(1-M**2))/(1+M**2);Rout=1/(1/f-1/Rin)
            q1=din+1j*z_c_in
            q2=dout+1j*z_c_out
            phi1=np.arctan(din/z_c_in)
            phi2=np.arctan(dout/z_c_out)
            Mirror_para2['f']=f
            Mirror_para2['din']=din
            Mirror_para2['Rin']=Rin
            Mirror_para2['dout']=dout
            Mirror_para2['Rout']=Rout
            Mirror_para2['q1']=q1
            Mirror_para2['q2']=q2
            Mirror_para2['phi1']=phi1
            Mirror_para2['phi2']=phi2

        elif M==1.0:
            f=d/4+f0**2/d
            din=(d-f*(1-M**2))/(1+M**2);Rin=din*(1+(np.pi*win**2/Lambda/din)**2)
            dout=(M**2*d+f*(1-M**2))/(1+M**2);Rout=1/(1/f-1/Rin)
            q1=din+1j*z_c_in
            q2=dout+1j*z_c_out
            phi1=np.arctan(din/z_c_in)
            phi2=np.arctan(dout/z_c_out)
            Mirror_para1['f']=f
            Mirror_para1['din']=din
            Mirror_para1['Rin']=Rin
            Mirror_para1['dout']=dout
            Mirror_para1['Rout']=Rout
            Mirror_para1['q1']=q1
            Mirror_para1['q2']=q2
            Mirror_para1['phi1']=phi1
            Mirror_para1['phi2']=phi2
        for key in Mirror_para1.keys():
             print(key,':',Mirror_para1[key])
        for key in Mirror_para2.keys():
             print(key,':',Mirror_para2[key])
        return Mirror_para1, Mirror_para2

def DrawBeamCountour(Mirror,Nin=301,Nout=301,size=2.5,color='b',offset=0,**kwargs):
    """
    **kwargs includes:
    """
    z_c1 = Mirror['z_c_in']
    z_c2 = Mirror['z_c_out']
    win  = Mirror['win']
    wout = Mirror['wout']
    din  = Mirror['din']
    dout = Mirror['dout']
    zin =  np.linspace(0, np.abs(din),Nin)
    zout = np.linspace(0,np.abs(dout),Nout)+zin[-1]
    z = np.append(zin,zout)+offset
    Countour = np.zeros((2,Nin+Nout))
    Countour_in = win*np.sqrt(1+(zin/z_c1)**2)
    Countour_out = wout*np.sqrt(1+((zout-din-np.abs(dout))/z_c2)**2)

    return z, Countour_in, Countour_out
    
