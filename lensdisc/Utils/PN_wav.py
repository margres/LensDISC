import numpy as np 

######## TaylorF2 wave form
## D (Mpc): the distance from source to the observer 
## chirpM (solar mass): the chirp mass
## nu (dimensionless): the symmetric mass ratio
def PN_waveFunc(f,tc, phic, D, chirpM, nu ):
    PI = np.pi

    chirpM *= 4.925616652796259e-06 # from solar mass to second
    M = chirpM*(nu**(-0.6))
    v = (PI*M*f)**(1.0/3)

    theta = 0.0
    phi = 0.0
    iota = 0.0
    psi = 0.0

    D *= 102927125037948.75 # from Mpc to seconds

    gama = 0.57721566490153286060651209008240243104215933593992

    vlso = 1.0/(6.0**0.5)

    Fp = -0.5*(1+np.cos(theta)**2)*np.cos(2*phi)*np.cos(2*psi)-np.cos(theta)*np.sin(2*phi)*np.sin(2*psi)
    Fm = 0.5*(1+np.cos(theta)**2)*np.cos(2*phi)*np.sin(2*psi)-np.cos(theta)*np.sin(2*phi)*np.cos(2*psi)
    Deff = D/((Fp**2)*(1+np.cos(iota)**2)/4.0+(Fm**2)*(np.cos(iota)**2))**0.5

    Psi = 2*PI*f*tc - phic - PI/4.0 + 3.0/(128.0*nu*(v**5))*(1+20.0/9.0*(743.0/336.0+11.0/4.0*nu)*(v**2.0)
        -16.0*PI*(v**3.0)+10.0*(3058673.0/1016064.0 + 5429.0/1008.0*nu + 617.0/144.0*(nu**2.0))*(v**4.0)
        +PI*(38645.0/756.0-65.0/9.0*nu)*(1+3*np.log(v/vlso))*(v**5.0)
        +(11583231236531.0/4694215680.0-640.0/3*(PI**2.0)-6848.0*gama/21.0-6848.0*np.log(4*v)/21.0
        +(-15737765635.0/3048192.0+2255.0*(PI**2.0)/12.0)*nu+76055.0/1728.0*(nu**2.0)-127825.0*(nu**3.0)/1296.0)*(v**6.0)
        +PI*(77096675.0/254016.0+378515.0*nu/1512.0-74045.0*(nu**2.0)/756.0)*(v**7.0))

    return (chirpM**(5.0/6.0))*f**(-7.0/6.0)*np.exp(Psi*1j)/Deff
