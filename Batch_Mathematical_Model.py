import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Physical constants and default parameters
D,kappa,rho, Cp,Qheat = 1.0,1.0, 1.0, 1.0, 1.0
nu_O2=3.0/4.0

#Aluminum transport properties
u_s,D_Al=0.1,1e-3

#Arrhenius parameters
K0, Ea,Ru=1e3, 1.25e5,8.314
alpha,beta,nTexp = 1.0,1.0, 0.0

#Domain setup
L, H = 1.0,0.099
dx, dy = 0.25,0.033
Nx, Ny = int(L/dx), int(H/dy)

def BuildMatrices(nx, rx, ax):
    #Setting up tridiagonal matrices for implicit scheme
    Ad = np.full(nx,1.0+2.0*rx)
    Au = np.full(nx-1,-rx+ ax)
    Al = np.full(nx-1,-rx -ax)
    Bd = np.full(nx,1.0-2.0*rx)
    Bu = np.full(nx-1,rx-ax)
    Bl = np.full(nx-1,rx+ax)
    return (Al, Ad, Au), (Bl, Bd, Bu)

def SolveTDMA(Al, Ad, Au, b):
    #Thomas algorithm for tridiagonal system
    if b.ndim==1:
        b = b[:, None]
    n, m = b.shape
    a, d, c = Al.copy(), Ad.copy(), Au.copy()
    cp, dp = np.zeros_like(c), np.zeros((n, m))
    
    cp[0]=c[0]/d[0]
    dp[0]=b[0]/d[0]
    for i in range(1, n-1):
        denom =d[i]-a[i-1]*cp[i-1]
        cp[i]=c[i]/denom
        dp[i]=  (b[i]-a[i-1]*dp[i-1])/denom
    
    denom=d[-1]-a[-1]*cp[-1]
    dp[-1]=(b[-1] - a[-1]*dp[-2])/denom
    
    x = np.zeros_like(dp)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i]-cp[i]*x[i+1]
    return x.squeeze()

def ReactionRate(cAl, cO2, T, K0, Ea, Ru):
    #Arrhenius kinetics with temperature clipping
    T_safe=np.clip(T, 300.0, 4000.0)
    kT =K0*(T_safe**nTexp)*np.exp(-Ea/(Ru*T_safe))
    return kT*(cAl**alpha)*(cO2**beta)

def PlotField(field, title, L, H, xlabel='x', ylabel='y', cbar_label=None):
    f = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    plt.figure(figsize=(6, 3))
    im = plt.imshow(f.T, origin='lower', extent=[0, L, 0, H], aspect='auto', interpolation='nearest')
    plt.title(title)
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel)
    cbar = plt.colorbar(im)
    if cbar_label: 
        cbar.set_label(cbar_label)
    plt.tight_layout()

def RunSimulation(u0, T_inlet, Twall, cO2_in, cAl_in, n_residence=1, return_fields=False, verbose=False,
                  D_=D, kappa_=kappa, rho_=rho, Cp_=Cp, Qheat_=Qheat,
                  u_s_=u_s, D_Al_=D_Al, K0_=K0, Ea_=Ea, Ru_=Ru,
                  L_=L, H_=H, dx_=dx, dy_=dy):

    Nx_, Ny_ = int(L_/dx_), int(H_/dy_)
    nx = Nx_-1

    #Time step calculations
    t_end = n_residence*(L_/u0)
    dt_diff = 0.45*dy_**2/max(D_,kappa_,D_Al_)
    dt_adv = 0.95*dx_/max(abs(u0), abs(u_s_), 1e-8)
    dt = min(dt_diff, dt_adv)
    nsteps = int(np.ceil(t_end/dt))

    #Dimensionless parameters
    rx_c,ax_c = D_*dt/(2.0*dx_*dx_), u0*dt/(4.0*dx_)
    ry_c= D_*dt/(dy_*dy_)
    rx_t,ax_t=kappa_*dt/(2.0*dx_*dx_), u0*dt/(4.0*dx_)
    ry_t=kappa_*dt/(dy_*dy_)
    gamma=dt*Qheat_/(rho_*Cp_)
    rx_a,ry_a=D_Al_*dt/(2.0*dx_*dx_), D_Al_*dt/(dy_*dy_)

    A_c,B_c=BuildMatrices(nx,rx_c,ax_c)
    A_t, B_t=BuildMatrices(nx,rx_t,ax_t)
    A_a,B_a=BuildMatrices(nx,rx_a,0.0)

    #Initialize field variables
    cO2=np.full((Nx_+1,Ny_+1),cO2_in)
    cAl=np.full((Nx_+1,Ny_+1),cAl_in)
    T=np.full((Nx_+1,Ny_+1),T_inlet)
    T[:,0],T[:,Ny_] = Twall,Twall

    def ApplyBoundaryConditions(cO2,cAl,T):
        #Inlet conditions
        cO2[0,:],cAl[0, :], T[0,:]=cO2_in,cAl_in,T_inlet
        
        #Outlet conditions
        cO2[Nx_,:],T[Nx_,:]=cO2[Nx_-1,:],T[Nx_-1,:]
        
        #Wall conditions
        cO2[:,0],cO2[:,Ny_]=cO2[:, 1], cO2[:, Ny_-1]
        cAl[:,0],cAl[:,Ny_]=cAl[:, 1], cAl[:, Ny_-1]
        T[:,0],T[:,Ny_]=Twall, Twall
        return cO2,cAl,T

    #Main time loop
    for step in range(nsteps):
        cO2,cAl,T=ApplyBoundaryConditions(cO2, cAl, T)

        #Calculate reaction rates with consumption limits
        Rk=ReactionRate(cAl, cO2, T, K0_, Ea_, Ru_)
        Rcap=np.minimum(cAl/dt, cO2/(nu_O2*dt))
        Rk=np.minimum(Rk, Rcap)

        #O2 transport equation
        cO2_new=cO2.copy()
        for j in range(1, Ny_):
            yterm=ry_c*(cO2[1:Nx_, j-1] - 2.0*cO2[1:Nx_, j] + cO2[1:Nx_, j+1])
            rhs=(B_c[1]*cO2[1:Nx_, j] + np.r_[0.0, B_c[0]]*cO2[0:Nx_-1, j] + np.r_[B_c[2], 0.0]*cO2[2:Nx_+1, j])
            rhs+=yterm - dt*nu_O2*Rk[1:Nx_, j]
            rhs[0]+=(rx_c + ax_c)*cO2_in + (rx_c + ax_c)*cO2_in
            x = SolveTDMA(*A_c,rhs)
            cO2_new[1:Nx_, j]=np.maximum(0.0,x)
        cO2=cO2_new
        cO2,cAl,T=ApplyBoundaryConditions(cO2, cAl, T)

        #Al transport with settling
        cAl_new = cAl.copy()
        CFL_Al = abs(u_s_)*dt/dx_
        for j in range(1, Ny_):
            yterm = ry_a*(cAl[1:Nx_, j-1]-2.0*cAl[1:Nx_, j]+cAl[1:Nx_, j+1])
            if u_s_ >= 0.0:
                advx=-(u_s_*dt/dx_)*(cAl[1:Nx_, j]-cAl[0:Nx_-1, j])
            else:
                advx=-(u_s_*dt/dx_)*(cAl[2:Nx_+1, j]-cAl[1:Nx_, j])

            rhs=(B_a[1]*cAl[1:Nx_, j] + np.r_[0.0, B_a[0]]*cAl[0:Nx_-1, j] + np.r_[B_a[2], 0.0]*cAl[2:Nx_+1, j])
            rhs+=yterm+advx-dt*Rk[1:Nx_, j]
            rhs[0]+=2.0*rx_a*cAl_in
            x = SolveTDMA(*A_a,rhs)
            cAl_new[1:Nx_,j]=np.maximum(0.0, x)

        cAl_new[Nx_, :] = cAl_new[Nx_-1, :]-(u_s_*dt/dx_)*(cAl_new[Nx_-1, :]-cAl_new[Nx_-2, :])
        cAl=cAl_new
        cO2,cAl,T=ApplyBoundaryConditions(cO2,cAl,T)

        #Temperature equation
        T_new = T.copy()
        for j in range(1, Ny_):
            yterm=ry_t*(T[1:Nx_, j-1]-2.0*T[1:Nx_, j] + T[1:Nx_, j+1])
            rhs=(B_t[1]*T[1:Nx_, j]+np.r_[0.0, B_t[0]]*T[0:Nx_-1, j]+np.r_[B_t[2],0.0]*T[2:Nx_+1, j])
            rhs+=yterm + gamma*Rk[1:Nx_, j]
            rhs[0]+=(rx_t+ax_t)*T_inlet + (rx_t + ax_t)*T_inlet
            x = SolveTDMA(*A_t,rhs)
            T_new[1:Nx_, j]=x
        T=T_new
        cO2,cAl,T=ApplyBoundaryConditions(cO2,cAl,T)

        if verbose and (step % max(1, nsteps//5) == 0):
            print(f"Step {step:6d}/{nsteps} | CFL_Al={CFL_Al:.3f} | min cAl={cAl.min():.3e} min cO2={cO2.min():.3e} max T={T.max():.1f}")

    #Calculate outputs
    cAl_out, cO2_out, T_out = cAl[Nx_, :].mean(),cO2[Nx_, :].mean(), T[Nx_, :].mean()
    Tmax=T.max()

    #Wall heat flux approximation
    qy0= -kappa_*(T[:,1]-T[:,0])/dy_
    qyH =-kappa_*(T[:,-1]- T[:,-2])/dy_
    Qwall_proxy=(qy0.sum()+ qyH.sum())*dx_

    if return_fields:
        return dict(cAl_out=cAl_out,cO2_out=cO2_out,T_out=T_out,Tmax=Tmax,Qwall=Qwall_proxy,
                    cO2=cO2, cAl=cAl,T=T,dt=dt,nsteps=nsteps)
    else:
        return dict(cAl_out=cAl_out,cO2_out=cO2_out,T_out=T_out,Tmax=Tmax,Qwall=Qwall_proxy, dt=dt,nsteps=nsteps)

#Sample generation functions
def GenerateRandomSamples(N,rng=None,u0_range=(0.05,0.50), Twall_range=(1200.0, 2000.0),
                         cAl_in_fixed=100.0, cO2_in_fixed=130.0,T_in_fixed=1500.0, K0_fixed=K0):
    rng= np.random.default_rng(rng)
    u0 =rng.uniform(*u0_range, size=N)
    Twall= rng.uniform(*Twall_range, size=N)
    Tin =np.full(N,T_in_fixed)
    cO2 =np.full(N,cO2_in_fixed)
    cAl =np.full(N,cAl_in_fixed)
    K0vec=np.full(N,K0_fixed)
    return dict(u0=u0,Tin=Tin,Twall=Twall,cO2_in=cO2,cAl_in=cAl,K0=K0vec)

def GenerateGridSamples(Twall_values, u0_values, cAl_in_fixed=100.0, cO2_in_fixed=130.0, 
                       T_in_fixed=1500.0,K0_fixed=K0):
    Tw,U=np.meshgrid(Twall_values, u0_values, indexing="ij")
    G=Tw.size
    return dict(u0=U.ravel(),Tin=np.full(G, T_in_fixed), Twall=Tw.ravel(),
                cO2_in=np.full(G,cO2_in_fixed),cAl_in=np.full(G,cAl_in_fixed), K0=np.full(G, K0_fixed))

def BatchRunner(N=100,n_residence=1, outfile="surrogate_data.xlsx",seed=0,verbose_every=50,
               use_grid=False, Twall_values=None, u0_values=None,u0_range=(0.05, 0.50), 
               Twall_range=(1200.0, 2000.0), cAl_in_fixed=100.0,cO2_in_fixed=130.0, 
               T_in_fixed=1500.0, K0_fixed=K0):

    if use_grid:
        if Twall_values is None or u0_values is None:
            raise ValueError("Need Twall_values and u0_values for grid sampling")
        S=GenerateGridSamples(Twall_values,u0_values,cAl_in_fixed,cO2_in_fixed,T_in_fixed,K0_fixed)
    else:
        S=GenerateRandomSamples(N,rng=seed, u0_range=u0_range,Twall_range=Twall_range,
                                 cAl_in_fixed=cAl_in_fixed, cO2_in_fixed=cO2_in_fixed, 
                                 T_in_fixed=T_in_fixed, K0_fixed=K0_fixed)

    results=[]
    total=len(S['u0'])
    for i in range(total):
        u0_i, Tin_i,Twall_i = S['u0'][i], S['Tin'][i], S['Twall'][i]
        cO2_in_i,cAl_in_i,K0_i = S['cO2_in'][i], S['cAl_in'][i], S['K0'][i]

        #Derived dimensionless numbers
        tau=L/max(u0_i, 1e-9)
        Pec,PeT=u0_i*L/D,u0_i*L/kappa
        k_ref=K0_i*np.exp(-Ea/(Ru*max(Tin_i,300.0)))
        Da=k_ref*tau

        out = RunSimulation(u0_i,Tin_i, Twall_i, cO2_in_i,cAl_in_i,
                           n_residence=n_residence,return_fields=False, K0_=K0_i)

        results.append(dict(u0=u0_i,T_in=Tin_i, T_wall=Twall_i, cO2_in=cO2_in_i, cAl_in=cAl_in_i, K0=K0_i,
                           tau=tau, Pe_c=Pec,Pe_T=PeT, Da=Da, cAl_out=out['cAl_out'], cO2_out=out['cO2_out'], 
                           T_out=out['T_out'],Tmax=out['Tmax'], Qwall_proxy=out['Qwall'], 
                           dt=out['dt'], nsteps=out['nsteps']))
        
        if verbose_every and (i % verbose_every == 0):
            print(f"[{i:5d}/{total}] u0={u0_i:.3f}, T_wall={Twall_i:.0f}, T_in={Tin_i:.0f}, cO2_in={cO2_in_i:.1f} -> T_out={out['T_out']:.1f}, cAl_out={out['cAl_out']:.3f}, cO2_out={out['cO2_out']:.3f}")

    df = pd.DataFrame(results)
    df.to_excel(outfile, index=False)
    print(f"\nSaved {len(df)} rows to {outfile}")
    return df

if __name__== "__main__":
    #Single case demo
    demo = RunSimulation(u0=0.2,T_inlet=1500.0,Twall=1600.0,cO2_in=130.0,cAl_in=100.0,
                        n_residence=1,return_fields=True,verbose=False)
    
    PlotField(demo['cO2'],r"$c_{O_2}$ (final)",L,H,cbar_label="concentration")
    PlotField(demo['cAl'],r"$c_{Al}$ (final)",L,H,cbar_label="concentration") 
    PlotField(demo['T'],r"$T$ (final)",L,H,cbar_label="temperature")
    plt.show()

    #Generate grid dataset
    Twall_vals=np.linspace(1200.0,2000.0,100)
    u0_vals=np.linspace(0.01,2,20)
    dataset=BatchRunner(use_grid=True,Twall_values=Twall_vals, u0_values=u0_vals,
                         n_residence=1,outfile="surrogate_data_u0_Twall_grid.xlsx",
                         seed=1, verbose_every=10)