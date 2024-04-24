import torch
from .two_elec_two_center_int_local_frame import two_elec_two_center_int_local_frame as TETCILF
from .cal_par import *
from .constants import ev

#two electron two center integrals
def two_elec_two_center_int(const,idxi, idxj, ni, nj, xij, rij, Z, zetas, zetap, gss, gpp, gp2, hsp):
    """
    two electron two center integrals in molecule frame
    """
    dtype = xij.dtype
    device = xij.device
    #two electron two center integrals
    # ni, nj, xij, rij for each pair
    # Z, zetas, zetap, gss, gpp, gp2, hsp for each atom

    #rotate(ni,nj,xij,rij,tore,da,db, qa,qb, rho0a,rho0b, rho1a,rho1b, rho2a,rho2b, cutoff=1.0e10):
    # ni, nj, xij, rij, da, db, qa, qb, rho0a, rho0b ... rho2b, shape (napirs,)
    # tore: dictionary type tensor tore[1]=1,
    #       valence shell charge for H, tore[6]=4, valence shell charge for C

    tore = const.tore
    qn = const.qn
    hpp = 0.5*(gpp-gp2)
    qn0 = qn[Z]
    #Z==0 is for padding zero
    isH = Z==1  # Hydrogen
    isX = Z>2   # Heavy atom

    rho_0 = torch.zeros_like(qn0)
    rho_1 = torch.zeros_like(qn0)
    rho_2 = torch.zeros_like(qn0)
    dd = torch.zeros_like(qn0)
    qq = torch.zeros_like(qn0)
    rho1 = additive_term_rho1.apply
    rho2 = additive_term_rho2.apply

    dd[isX], qq[isX] = dd_qq(qn0[isX], zetas[isX], zetap[isX])
    rho_0[isH] = 0.5 * ev / gss[isH]
    rho_0[isX] = 0.5 * ev / gss[isX]
    if torch.sum(isX)>0:
        rho_1[isX] = rho1(hsp[isX], dd[isX])
        rho_2[isX] = rho2(hpp[isX], qq[isX])
    
    w, e1b, e2a = \
        rotate(ni, nj, xij, rij, tore, dd[idxi], dd[idxj], \
               qq[idxi], qq[idxj], \
               rho_0[idxi], rho_0[idxj], \
               rho_1[idxi], rho_1[idxj], \
               rho_2[idxi], rho_2[idxj])

    return w, e1b, e2a


#rotate: rotate the two electron two center integrals from local frame to molecule frame
def rotate(ni, nj, xij, rij, tore, da, db, qa, qb, rho0a, rho0b, rho1a, rho1b, rho2a, rho2b, cutoff=1.0e10):
    """
    rotate the two elecron two center integrals from local frame to molecule frame
    """
    dtype =  xij.dtype
    device = xij.device
    # in mopac, xij = xi - xj for rotate, rather than the general one, xj-xi
    # thus put minus sign on xij
    # ni, nj, xij, rij, da, db, qa, qb, rho0a, rho0b ... rho2b, shape (napirs,)
    # tore: dictionary type tensor tore[1]=1,
    #       valence shell charge for H, tore[6]=4, valence shell charge for C

    #   ROTATE CALCULATES THE TWO-PARTICLE INTERACTIONS.
    #
    #   ON INPUT  NI     = ATOMIC NUMBER OF FIRST ATOM.
    #             NJ     = ATOMIC NUMBER OF SECOND ATOM.
    #
    # ON OUTPUT W      = ARRAY OF TWO-ELECTRON REPULSION INTEGRALS.
    #           E1B,E2A= ARRAY OF ELECTRON-NUCLEAR ATTRACTION INTEGRALS,
    #                    E1B = ELECTRON ON ATOM NI ATTRACTING NUCLEUS OF NJ.
    # for the local frame, refer to TwoElectronTwoCenterIntegralsLocalFrame.py
    # repp.f
    # xij, unit vector from atom i to atom j, shape (npairs, 3)
    # ni>=nj, shape (npairs, )
    # rij, distance between atom i and atom j in atomic unit, shape (npairs,)
    # tore, da, db, qa, qb, rho0, rho1, rho2 ==> refer to TwoElectronTwoCenterIntegralsLocalFrame.py
    # cutoff, not used
    #***************************************************************************
    #the inner cutoff in rotate.f is sqrt(2e-5) Angstrom = 0.00447 Angstrom
    #default inner cutoff in data_loader is 0.001 Angstrom
    # +++++
    # in this module, xij is the unit vector for the direction of atom i -> j
    # update data_loader soon
    # +++++
    #***************************************************************************
    #xij = xij0/torch.norm(xij,dim=1,keepdim=True)

    #enuc is not computed at this moment
    HH = (ni==1) & (nj==1)
    XH = (ni>1) & (nj==1)
    XX = (ni>1) & (nj>1)

    #rij = torch.where(rij>cutoff, torch.tensor(cutoff,dtype=dtype),rij)
    #ni>=nj
    #
    #w[1] (s s/s s)
    #wHH = ri[1]

    #riHH, riXH, ri, coreHH, coreXH, core = \
    wHH, riXH, ri, coreHH, coreXH, core = \
           TETCILF(ni,nj,rij, tore, \
                da, db, qa,qb, rho0a,rho0b, rho1a,rho1b, rho2a,rho2b)
    
    #print(wHH)
    #print(riXH)
    #print(ri)
    #

    ###############################33
    # X-H hevay atom - Hydrogen
    xXH = -xij[XH]
    yXH = torch.zeros(xXH.shape[0], 2, dtype=dtype, device=device)
    zXH = torch.zeros_like(xXH)
    #cond1 = torch.abs(xXH[...,3-1])>0.99999999
    #xXH[...,3-1] = torch.where(xXH[...,3-1]>0.99999999, torch.tensor([1.0],dtype=dtype, xXH[...,3-1]))
    #xXH[...,3-1] = torch.where(xXH[...,3-1]<-0.99999999, torch.tensor([-1.0],dtype=dtype, xXH[...,3-1]))
    #zXH[...,3-1] = torch,where(cond1, torch.tensor([0.0],dtype=dtype),
    #                           torch.sqrt(1.0-xXH[...,3-1]**2))
    """
    pytorch new version doesn't support modify z and z depends on a , a depends on z
    zXH[...,3-1] =  torch.sqrt(1.0-xXH[...,3-1]**2)
    cond1XH = zXH[...,3-1]>1.0e-5
    #cond1 = torch.abs(xXH[...,3-1])>0.99999999, but z[3] is used in dominator, so change to
    #cond1 = zXH[...,3-1]>0.0
    #abs(x[3])>0.99999999 ==> z[3]<1.4142e-4
    #don't use th.where, use indexing to get rid of singularity
    aXH = 1.0/zXH[cond1XH,3-1]

    zXH[...,1-1] = 1.0
    zXH[cond1XH,1-1] = -aXH*xXH[cond1XH,1-1]*xXH[cond1XH,3-1]

    #zXH[...,2-1]=0.0
    zXH[cond1XH,2-1] = -aXH*xXH[cond1XH,2-1]*xXH[cond1XH,3-1]
    """
    #modify the code
    #xXH.register_hook(print)
    #zXH2 = torch.sqrt(1.0-xXH[...,3-1]**2)
    zXH2 = torch.zeros_like(xXH[...,2])
    cond_xXH2 = torch.abs(xXH[...,2])<1.0
    zXH2[cond_xXH2] = torch.sqrt(1.0 - xXH[cond_xXH2,2]**2)
    cond1XH = zXH2>1.0e-5
    aXH = 1.0 / zXH2[cond1XH]
    zXH0 =  torch.ones_like(zXH2)
    zXH0[cond1XH] = -aXH*xXH[cond1XH,0]*xXH[cond1XH,2]
    zXH1 = torch.zeros_like(zXH2)
    zXH1[cond1XH] = -aXH*xXH[cond1XH,1]*xXH[cond1XH,2]
    zXH = torch.stack((zXH0, zXH1, zXH2), dim=1)


    #yXH[...,1-1]=0.0
    yXH[cond1XH,0] =  aXH*xXH[cond1XH,1] * \
                      torch.where( xXH[cond1XH,1-1]>=0.0, \
                                   torch.tensor(-1.0,dtype=dtype, device=device), \
                                   torch.tensor(1.0,dtype=dtype, device=device) )
    #yXH[cond1XH,1-1] = -aXH*xXH[cond1XH,2-1]
    #yXH[xXH[...,1-1]<0.0,1-1] *= -1.0
    #yXH[xXH[...,1-1]<0.0,1-1].mul_(-1.0)

    yXH[...,1] = 1.0
    yXH[cond1XH,1] = torch.abs(aXH * xXH[cond1XH,0])
    #y[3] is not used

    xx11XH = xXH[...,0]**2
    xx21XH = xXH[...,1]*xXH[...,0]
    xx22XH = xXH[...,1]**2
    xx31XH = xXH[...,2]*xXH[...,0]
    xx32XH = xXH[...,2]*xXH[...,1]
    xx33XH = xXH[...,2]**2
    zz31XH = zXH[...,2]*zXH[...,0]
    zz32XH = zXH[...,2]*zXH[...,1]
    zz33XH = zXH[...,2]**2
    yyzz11XH = yXH[...,0]**2 + zXH[...,0]**2
    yyzz21XH = yXH[...,1]*yXH[...,0] + zXH[...,1]*zXH[...,0]
    yyzz22XH = yXH[...,1]**2 + zXH[...,1]**2

    wXH = torch.zeros(riXH.shape[0], 10, dtype=dtype, device=device)
    # (s s/s s)
    wXH[...,0] = riXH[...,0]
    ## (px s/s s)
    #w[1] = ri[1]*x[0]
    wXH[...,1] = riXH[...,1]*xXH[...,0]
    ## (px px/s s)
    #w[2] = ri[2]*xx11 + ri[3]*yyzz11
    wXH[...,2] = riXH[...,2]*xx11XH + riXH[...,3]*yyzz11XH
    ## (py s/s s)
    #w[3] = ri[1]*x[1]
    wXH[...,3] = riXH[...,1]*xXH[...,1]
    ## (py px/s s)
    #w[4] = ri[2]*xx21 + ri[3]*yyzz21
    wXH[...,4] = riXH[...,2]*xx21XH + riXH[...,3]*yyzz21XH
    ## (py py/s s)
    #w[5] = ri[2]*xx22 + ri[3]*yyzz22
    wXH[...,5] = riXH[...,2]*xx22XH + riXH[...,3]*yyzz22XH
    ## (pz s/ss)
    #w[6] = ri[1]*x[2]
    wXH[...,6] = riXH[...,1]*xXH[...,2]
    ## (pz px/s s)
    #w[7] = ri[2]*xx31 + ri[3]*zz31
    wXH[...,7] = riXH[...,2]*xx31XH + riXH[...,3]*zz31XH
    ## (pz py/s s)
    #w[8] = ri[2]*xx32 + ri[3]*zz32
    wXH[...,8] = riXH[...,2]*xx32XH + riXH[...,3]*zz32XH
    ## (pz pz/s s)
    #w[9] = ri[2]*xx33 + ri[3]*zz33
    wXH[...,9] = riXH[...,2]*xx33XH + riXH[...,3]*zz33XH

    ##############################################
    # X-X heavy atom - heavy atom
    x = -xij[XX]
    y = torch.zeros(x.shape[0], 2, dtype=dtype, device=device)
    #z=torch.zeros_like(x)
    #cond1 = torch.abs(x[...,3-1])>0.99999999
    #x[...,3-1] = torch.where(x[...,3-1]>0.99999999, torch.tensor([1.0],dtype=dtype, x[...,3-1]))
    #x[...,3-1] = torch.where(x[...,3-1]<-0.99999999, torch.tensor([-1.0],dtype=dtype, x[...,3-1]))
    #z[...,3-1] = torch,where(cond1, torch.tensor([0.0],dtype=dtype),
    #                           torch.sqrt(1.0-x[...,3-1]**2))
    """
    #not working in the new version as a depends on z
    z[...,3-1] =  torch.sqrt(1.0-x[...,3-1]**2)
    cond1XX = z[...,3-1]>1.0e-5
    #cond1 = torch.abs(x[...,3-1])>0.99999999, but z[3] is used in dominator, so change to
    #cond1 = z[...,3-1]>0.0
    #abs(x[3])>0.99999999 ==> z[3]<1.4142e-4
    #don't use th.where, use indexing to get rid of singularity
    a = 1.0/z[cond1XX,3-1]
    z[...,1-1] = 1.0
    z[cond1XX,1-1] = -a*x[cond1XX,1-1]*x[cond1XX,3-1]

    #z[...,2-1]=0.0
    z[cond1XX,2-1] = -a*x[cond1XX,2-1]*x[cond1XX,3-1]
    """
    #modify the code

    #z2 = torch.sqrt(1.0-x[...,3-1]**2)
    cond_x2 = torch.abs(x[...,2])<1.0
    z2 = torch.zeros_like(x[...,2])
    z2[cond_x2] = torch.sqrt(1.0 - x[cond_x2,2]**2)
    cond1XX = z2 > 1.0e-5
    a = 1.0 / z2[cond1XX]
    z0 = torch.ones_like(z2)
    z0[cond1XX] = -a*x[cond1XX,0]*x[cond1XX,2]
    z1 = torch.zeros_like(z2)
    z1[cond1XX] = -a*x[cond1XX,1]*x[cond1XX,2]

    z = torch.stack((z0,z1,z2),dim=1)

    #y[...,1-1]=0.0
    #y[cond1XX,0] =  a*x[cond1XX,2-1] * \
    #                  torch.where( x[cond1XX,1-1]>=0.0, \
    #                               torch.tensor(-1.0,dtype=dtype, device=device), \
    #                               torch.tensor(1.0,dtype=dtype, device=device) )
    #y[cond1XX,1-1] = -a*x[cond1XX,2-1]
    #y[x[...,1-1]<0.0,1-1] *= -1.0

    y[...,0] = 0.0
    cond1XX_X1g0 = cond1XX & ( x[...,0]>=0.0 )
    cond1XX_X1l0 = cond1XX & ( x[...,0]<0.0 )
    y[cond1XX_X1g0,0] = -(1.0/z2[cond1XX_X1g0])*x[cond1XX_X1g0,1]
    y[cond1XX_X1l0,0] = (1.0/z2[cond1XX_X1l0])*x[cond1XX_X1l0,1]



    y[...,1] = 1.0
    y[cond1XX,1] = torch.abs(a * x[cond1XX,0])
    #y[3] is not used


    xx11 = x[...,0]**2
    xx21 = x[...,1]*x[...,0]
    xx22 = x[...,1]**2
    xx31 = x[...,2]*x[...,0]
    xx32 = x[...,2]*x[...,1]
    xx33 = x[...,2]**2
    yy11 = y[...,0]**2
    yy21 = y[...,1]*y[...,0]
    yy22 = y[...,1]**2
    zz11 = z[...,0]**2
    zz21 = z[...,1]*z[...,0]
    zz22 = z[...,1]**2
    zz31 = z[...,2]*z[...,0]
    zz32 = z[...,2]*z[...,1]
    zz33 = z[...,2]**2
    yyzz11 = yy11+zz11
    yyzz21 = yy21+zz21
    yyzz22 = yy22+zz22
    xy11 = 2.0*x[...,0]*y[...,0]
    xy21 = x[...,0]*y[...,1] + x[...,1]*y[...,0]
    xy22 = 2.0*x[...,1]*y[...,1]
    xy31 = x[...,2]*y[...,0]
    xy32 = x[...,2]*y[...,1]
    xz11 = 2.0*x[...,0]*z[...,0]
    xz21 = x[...,0]*z[...,1] + x[...,1]*z[...,0]
    xz22 = 2.0*x[...,1]*z[...,1]
    xz31 = x[...,0]*z[...,2] + x[...,2]*z[...,0]
    xz32 = x[...,1]*z[...,2] + x[...,2]*z[...,1]
    xz33 = 2.0*x[...,2]*z[...,2]
    yz11 = 2.0*y[...,0]*z[...,0]
    yz21 = y[...,0]*z[...,1]+y[...,1]*z[...,0]
    yz22 = 2.0*y[...,1]*z[...,1]
    yz31 = y[...,0]*z[...,2]
    yz32 = y[...,1]*z[...,2]

    w = torch.zeros(ri.shape[0], 100, dtype=dtype, device=device)
    ##(S S/S S)
    #w[1] = ri[1]
    w[...,0] = ri[...,0]
    ## (s s/px s)
    #w[2] = ri[5]*x[1]
    w[...,1] = ri[...,4]*x[...,0]
    ## (s s/px px)
    #w[3] = ri[11]*xx11 + ri[12]*yyzz11
    w[...,2] = ri[...,10]*xx11 + ri[...,11]*yyzz11
    ## (s s/py s)
    #w[4] = ri[5]*x[2]
    w[...,3] = ri[...,4]*x[...,1]
    ## (s s/py px)
    #w[5] = ri[11]*xx21 + ri[12]*yyzz21
    w[...,4] = ri[...,10]*xx21 + ri[...,11]*yyzz21
    ## (s s/py py)
    #w[6] = ri[11]*xx22 + ri[12]*yyzz22
    w[...,5] = ri[...,10]*xx22 + ri[...,11]*yyzz22
    ## (s s/pz s)
    #w[7] = ri[5]*x[3]
    w[...,6] = ri[...,4]*x[...,2]
    ## (s s/pz px)
    #w[8] = ri[11]*xx31 + ri[12]*zz31
    w[...,7] = ri[...,10]*xx31 + ri[...,11]*zz31
    ## (s s/pz py)
    #w[9] = ri[11]*xx32 + ri[12]*zz32
    w[...,8] = ri[...,10]*xx32 + ri[...,11]*zz32
    ## (s s/pz pz)
    #w[10] = ri[11]*xx33 + ri[12]*zz33
    w[...,9] = ri[...,10]*xx33 + ri[...,11]*zz33
    ## (px s/s s)
    #w[11] = ri[2]*x[1]
    w[...,10] = ri[...,1]*x[...,0]
    # (px s/px s)
    w[...,11] = ri[...,5]*xx11 + ri[...,6]*yyzz11
    # (px s/px px)
    w[...,12] = x[...,0] * (ri[...,12]*xx11 + ri[...,13]*yyzz11) \
          + ri[...,14] * (y[...,0]*xy11 + z[...,0]*xz11)
    # (px s/py s)
    w[...,13] = ri[...,5]*xx21 + ri[...,6]*yyzz21
    # (px s/py px)
    w[...,14] = x[...,0] * (ri[...,12]*xx21 + ri[...,13]*yyzz21) \
          + ri[...,14] * (y[...,0]*xy21 + z[...,0]*xz21)
    # (px s/py py)
    w[...,15] = x[...,0] * (ri[...,12]*xx22 + ri[...,13]*yyzz22) \
          + ri[...,14] * (y[...,0]*xy22 + z[...,0]*xz22)
    # (px s/pz s)
    w[...,16] = ri[...,5]*xx31 + ri[...,6]*zz31
    # (px s/pz px)
    w[...,17] = x[...,0] * (ri[...,12]*xx31 + ri[...,13]*zz31) \
          + ri[...,14] * (y[...,0]*xy31 + z[...,0]*xz31)
    # (px s/pz py)
    w[...,18] = x[...,0] * (ri[...,12]*xx32 + ri[...,13]*zz32) \
          + ri[...,14] * (y[...,0]*xy32 + z[...,0]*xz32)
    # (px s/pz pz)
    w[...,19] = x[...,0] * (ri[...,12]*xx33 + ri[...,13]*zz33) \
          + ri[...,14] * (     z[...,0]*xz33)
    # (px px/s s)
    w[...,20] = ri[...,2]*xx11 + ri[...,3]*yyzz11
    # (px px/px s)
    w[...,21] = x[...,0] * (ri[...,7]*xx11 + ri[...,8]*yyzz11) \
          + ri[...,9] * (y[...,0]*xy11 + z[...,0]*xz11)
    # (px px/px px)
    w[...,22] =  \
       ( ri[...,15] * xx11 + ri[...,16] * yyzz11 ) * xx11 \
       + ri[...,17] * xx11*yyzz11 \
       + ri[...,18] * (yy11*yy11 + zz11*zz11) \
       + ri[...,19] * (xy11*xy11 + xz11*xz11) \
       + ri[...,20] * (yy11*zz11 + zz11*yy11) \
       + ri[...,21] * yz11*yz11
    # (px px/py s)
    w[...,23] = x[...,1] * (ri[...,7]*xx11 + ri[...,8]*yyzz11) \
          + ri[...,9] * (y[...,1]*xy11 + z[...,1]*xz11)
    # (px px/py px)
    w[...,24] =  \
       ( ri[...,15] * xx11 + ri[...,16] * yyzz11 ) * xx21 \
       + ri[...,17] * xx11*yyzz21 \
       + ri[...,18] * (yy11*yy21 + zz11*zz21) \
       + ri[...,19] * (xy11*xy21 + xz11*xz21) \
       + ri[...,20] * (yy11*zz21 + zz11*yy21) \
       + ri[...,21] * yz11*yz21
    # (px px/py py)
    w[...,25] =  \
       ( ri[...,15] * xx11 + ri[...,16] * yyzz11 ) * xx22 \
       + ri[...,17] * xx11*yyzz22 \
       + ri[...,18] * (yy11*yy22 + zz11*zz22) \
       + ri[...,19] * (xy11*xy22 + xz11*xz22) \
       + ri[...,20] * (yy11*zz22 + zz11*yy22) \
       + ri[...,21] * yz11*yz22
    # (px px/pz s)
    w[...,26] = x[...,2] * (ri[...,7]*xx11 + ri[...,8]*yyzz11) \
          + ri[...,9] * z[...,2]*xz11
    # (px px/pz px)
    w[...,27] =  \
       (ri[...,15]*xx11 + ri[...,16]*yyzz11 ) * xx31 \
       + (ri[...,17]*xx11 + ri[...,18]*zz11 + ri[...,20]*yy11 ) * zz31 \
       + ri[...,19] * (xy11*xy31 + xz11*xz31) \
       + ri[...,21]*yz11*yz31
    # (px px/pz py)
    w[...,28] =  \
       (ri[...,15]*xx11 + ri[...,16]*yyzz11 ) * xx32 \
       + (ri[...,17]*xx11 + ri[...,18]*zz11 + ri[...,20]*yy11 ) * zz32 \
       + ri[...,19] * (xy11*xy32 + xz11*xz32) \
       + ri[...,21]*yz11*yz32
    # (px px/pz pz)
    w[...,29] =  \
       (ri[...,15]*xx11 + ri[...,16]*yyzz11 ) * xx33 \
       + (ri[...,17]*xx11 + ri[...,18]*zz11 + ri[...,20]*yy11 ) * zz33 \
       + ri[...,19]*xz11*xz33
    # (py s/s s)
    w[...,30] = ri[...,1]*x[...,1]
    # (py s/px s)
    w[...,31] = ri[...,5]*xx21 + ri[...,6]*yyzz21
    # (py s/px px)
    w[...,32] = x[...,1] * (ri[...,12]*xx11 + ri[...,13]*yyzz11) \
          + ri[...,14] * (y[...,1]*xy11 + z[...,1]*xz11)
    # (py s/py s)
    w[...,33] = ri[...,5]*xx22 + ri[...,6]*yyzz22
    # (py s/py px)
    w[...,34] = x[...,1] * (ri[...,12]*xx21 + ri[...,13]*yyzz21) \
          + ri[...,14] * (y[...,1]*xy21 + z[...,1]*xz21)
    # (py s/py py)
    w[...,35] = x[...,1] * (ri[...,12]*xx22 + ri[...,13]*yyzz22) \
          + ri[...,14] * (y[...,1]*xy22 + z[...,1]*xz22)
    # (py s/pz s)
    w[...,36] = ri[...,5]*xx32 + ri[...,6]*zz32
    # (py s/pz px)
    w[...,37] = x[...,1] * (ri[...,12]*xx31 + ri[...,13]*zz31) \
          + ri[...,14] * (y[...,1]*xy31 + z[...,1]*xz31)
    # (py s/pz py)
    w[...,38] = x[...,1] * (ri[...,12]*xx32 + ri[...,13]*zz32) \
          + ri[...,14] * (y[...,1]*xy32 + z[...,1]*xz32)
    # (py s/pz pz)
    w[...,39] = x[...,1] * (ri[...,12]*xx33 + ri[...,13]*zz33) \
          + ri[...,14] * z[...,1]*xz33
    # (py px/s s)
    w[...,40] = ri[...,2]*xx21 + ri[...,3]*yyzz21
    # (py px/px s)
    w[...,41] = x[...,0] * (ri[...,7]*xx21 + ri[...,8]*yyzz21) \
          + ri[...,9] * (y[...,0]*xy21 + z[...,0]*xz21)
    # (py px/px px)
    w[...,42] =  \
       ( ri[...,15] * xx21 + ri[...,16]*yyzz21 ) * xx11 \
       + ri[...,17] * xx21*yyzz11 \
       + ri[...,18] * (yy21*yy11 + zz21*zz11) \
       + ri[...,19] * (xy21*xy11 + xz21*xz11) \
       + ri[...,20] * (yy21*zz11 + zz21*yy11) \
       + ri[...,21] * yz21*yz11
    # (py px/py s)
    w[...,43] = x[...,1] * (ri[...,7]*xx21 + ri[...,8]*yyzz21) \
          + ri[...,9] * (y[...,1]*xy21 + z[...,1]*xz21)
    # (py px/py px)
    w[...,44] =  \
       ( ri[...,15] * xx21 + ri[...,16]*yyzz21 ) * xx21 \
       + ri[...,17] * xx21*yyzz21 \
       + ri[...,18] * (yy21*yy21 + zz21*zz21) \
       + ri[...,19] * (xy21*xy21 + xz21*xz21) \
       + ri[...,20] * (yy21*zz21 + zz21*yy21) \
       + ri[...,21] * yz21*yz21
    # (py px/py py)
    w[...,45] =  \
       ( ri[...,15] * xx21 + ri[...,16]*yyzz21 ) * xx22 \
       + ri[...,17] * xx21*yyzz22 \
       + ri[...,18] * (yy21*yy22 + zz21*zz22) \
       + ri[...,19] * (xy21*xy22 + xz21*xz22) \
       + ri[...,20] * (yy21*zz22 + zz21*yy22) \
       + ri[...,21] * yz21*yz22
    # (py px/pz s)
    w[...,46] = x[...,2] * (ri[...,7]*xx21 + ri[...,8]*yyzz21) \
          + ri[...,9] * z[...,2]*xz21
    # (py px/pz px)
    w[...,47] =  \
       ( ri[...,15]*xx21 + ri[...,16]*yyzz21 ) * xx31 \
       + (ri[...,17]*xx21 + ri[...,18]*zz21 + ri[...,20]*yy21 ) * zz31 \
       + ri[...,19] * (xy21*xy31 + xz21*xz31) \
       + ri[...,21]*yz21*yz31
    # (py px/pz py)
    w[...,48] =  \
       (ri[...,15]*xx21 + ri[...,16]*yyzz21 ) * xx32 \
       + (ri[...,17]*xx21 + ri[...,18]*zz21 + ri[...,20]*yy21 ) * zz32 \
       + ri[...,19] * (xy21*xy32 + xz21*xz32) \
       + ri[...,21]*yz21*yz32
    # (py px/pz pz)
    w[...,49] =  \
       (ri[...,15]*xx21 + ri[...,16]*yyzz21 ) * xx33 \
       + (ri[...,17]*xx21 + ri[...,18]*zz21 + ri[...,20]*yy21 ) * zz33 \
       + ri[...,19]*xz21*xz33
    # (py py/s s)
    w[...,50] = ri[...,2]*xx22 + ri[...,3]*yyzz22
    # (py py/px s)
    w[...,51] = x[...,0] * (ri[...,7]*xx22 + ri[...,8]*yyzz22) \
          + ri[...,9] * (y[...,0]*xy22 + z[...,0]*xz22)
    # (py py/px px)
    w[...,52] =  \
       ( ri[...,15] * xx22 + ri[...,16]*yyzz22 ) * xx11 \
       + ri[...,17] * xx22*yyzz11 \
       + ri[...,18] * (yy22*yy11 + zz22*zz11) \
       + ri[...,19] * (xy22*xy11 + xz22*xz11) \
       + ri[...,20] * (yy22*zz11 + zz22*yy11) \
       + ri[...,21] * yz22*yz11
    # (py py/py s)
    w[...,53] = x[...,1] * (ri[...,7]*xx22 + ri[...,8]*yyzz22) \
          + ri[...,9] * (y[...,1]*xy22 + z[...,1]*xz22)
    # (py py/py px)
    w[...,54] =  \
       ( ri[...,15] * xx22 + ri[...,16]*yyzz22 ) * xx21 \
       + ri[...,17] * xx22*yyzz21 \
       + ri[...,18] * (yy22*yy21 + zz22*zz21) \
       + ri[...,19] * (xy22*xy21 + xz22*xz21) \
       + ri[...,20] * (yy22*zz21 + zz22*yy21) \
       + ri[...,21] * yz22*yz21
    # (py py/py py)
    w[...,55] =  \
       ( ri[...,15] * xx22 + ri[...,16]*yyzz22 ) * xx22 \
       + ri[...,17] * xx22*yyzz22 \
       + ri[...,18] * (yy22*yy22 + zz22*zz22) \
       + ri[...,19] * (xy22*xy22 + xz22*xz22) \
       + ri[...,20] * (yy22*zz22 + zz22*yy22) \
       + ri[...,21] * yz22*yz22
    # (py py/pz s)
    w[...,56] = x[...,2] * (ri[...,7]*xx22 + ri[...,8]*yyzz22) \
          + ri[...,9] *  z[...,2]*xz22
    # (py py/pz px)
    w[...,57] =  \
       ( ri[...,15] * xx22 + ri[...,16]*yyzz22 ) * xx31 \
       + (ri[...,17]*xx22 + ri[...,18]*zz22 + ri[...,20]*yy22 ) * zz31 \
       + ri[...,19] * (xy22*xy31 + xz22*xz31) \
       + ri[...,21]*yz22*yz31
    # (py py/pz py)
    w[...,58] =  \
       (ri[...,15]*xx22 + ri[...,16]*yyzz22 ) * xx32 \
       + (ri[...,17]*xx22 + ri[...,18]*zz22 + ri[...,20]*yy22 ) * zz32 \
       + ri[...,19] * (xy22*xy32 + xz22*xz32) \
       + ri[...,21]*yz22*yz32
    # (py py/pz pz)
    w[...,59] =  \
       (ri[...,15]*xx22 + ri[...,16]*yyzz22 ) * xx33 \
       + (ri[...,17]*xx22 + ri[...,18]*zz22 + ri[...,20]*yy22 ) * zz33 \
       + ri[...,19]*xz22*xz33
    # (pz s/ss)
    w[...,60] = ri[...,1]*x[...,2]
    # (pz s/px s)
    w[...,61] = ri[...,5]*xx31 + ri[...,6]*zz31
    # (pz s/px px)
    w[...,62] = x[...,2] * (ri[...,12]*xx11 + ri[...,13]*yyzz11) \
          + ri[...,14] * z[...,2]*xz11
    # (pz s/py s)
    w[...,63] = ri[...,5]*xx32 + ri[...,6]*zz32
    # (pz s/py px)
    w[...,64] = x[...,2] * (ri[...,12]*xx21 + ri[...,13]*yyzz21) \
          + ri[...,14] * z[...,2]*xz21
    # (pz s/py py)
    w[...,65] = x[...,2] * (ri[...,12]*xx22 + ri[...,13]*yyzz22) \
          + ri[...,14] * z[...,2]*xz22
    # (pz s/pz s)
    w[...,66] = ri[...,5]*xx33 + ri[...,6]*zz33
    # (pz s/pz px)
    w[...,67] = x[...,2] * (ri[...,12]*xx31 + ri[...,13]*zz31) \
          + ri[...,14] *  z[...,2]*xz31
    # (pz s/pz py)
    w[...,68] = x[...,2] * (ri[...,12]*xx32 + ri[...,13]*zz32) \
          + ri[...,14] * z[...,2]*xz32
    # (pz s/pz pz)
    w[...,69] = x[...,2] * (ri[...,12]*xx33 + ri[...,13]*zz33) \
          + ri[...,14] * z[...,2]*xz33
    # (pz px/s s)
    w[...,70] = ri[...,2]*xx31 + ri[...,3]*zz31
    # (pz px/px s)
    w[...,71] = x[...,0] * (ri[...,7]*xx31 + ri[...,8]*zz31) \
          + ri[...,9] * (y[...,0]*xy31 + z[...,0]*xz31)
    # (pz px/px px)
    w[...,72] =  \
       ( ri[...,15] * xx31 + ri[...,16]*zz31 ) * xx11 \
       + ri[...,17] * xx31*yyzz11 \
       + ri[...,18] * zz31*zz11 \
       + ri[...,19] * (xy31*xy11 + xz31*xz11) \
       + ri[...,20] * zz31*yy11 \
       + ri[...,21] * yz31*yz11
    # (pz px/py s)
    w[...,73] = x[...,1] * (ri[...,7]*xx31 + ri[...,8]*zz31) \
          + ri[...,9] * (y[...,1]*xy31 + z[...,1]*xz31)
    # (pz px/py px)
    w[...,74] =  \
       ( ri[...,15] * xx31 + ri[...,16]*zz31 ) * xx21 \
       + ri[...,17] * xx31*yyzz21 \
       + ri[...,18] * zz31*zz21 \
       + ri[...,19] * (xy31*xy21 + xz31*xz21) \
       + ri[...,20] * zz31*yy21 \
       + ri[...,21] * yz31*yz21
    # (pz px/py py)
    w[...,75] =  \
       ( ri[...,15] * xx31 + ri[...,16]*zz31 ) * xx22 \
       + ri[...,17] * xx31*yyzz22 \
       + ri[...,18] * zz31*zz22 \
       + ri[...,19] * (xy31*xy22 + xz31*xz22) \
       + ri[...,20] * zz31*yy22 \
       + ri[...,21] * yz31*yz22
    # (pz px/pz s)
    w[...,76] = x[...,2] * (ri[...,7]*xx31 + ri[...,8]*zz31) \
          + ri[...,9] * z[...,2]*xz31
    # (pz px/pz px)
    w[...,77] =  \
       (ri[...,15]*xx31 + ri[...,16]*zz31 ) * xx31 \
       + (ri[...,17]*xx31 + ri[...,18]*zz31 ) * zz31 \
       + ri[...,19] * (xy31*xy31 + xz31*xz31) \
       + ri[...,21]*yz31*yz31
    # (pz px/pz py)
    w[...,78] =  \
       (ri[...,15]*xx31 + ri[...,16]*zz31 ) * xx32 \
       + (ri[...,17]*xx31 + ri[...,18]*zz31 ) * zz32 \
       + ri[...,19] * (xy31*xy32 + xz31*xz32) \
       + ri[...,21]*yz31*yz32
    # (pz px/pz pz)
    w[...,79] =  \
       (ri[...,15]*xx31 + ri[...,16]*zz31 ) * xx33 \
       + (ri[...,17]*xx31 + ri[...,18]*zz31 ) * zz33 \
       + ri[...,19]*xz31*xz33
    # (pz py/s s)
    w[...,80] = ri[...,2]*xx32 + ri[...,3]*zz32
    # (pz py/px s)
    w[...,81] = x[...,0] * (ri[...,7]*xx32 + ri[...,8]*zz32) \
          + ri[...,9] * (y[...,0]*xy32 + z[...,0]*xz32)
    # (pz py/px px)
    w[...,82] =  \
       ( ri[...,15] * xx32 + ri[...,16]*zz32 ) * xx11 \
       + ri[...,17] * xx32*yyzz11 \
       + ri[...,18] * zz32*zz11 \
       + ri[...,19] * (xy32*xy11 + xz32*xz11) \
       + ri[...,20] * zz32*yy11 \
       + ri[...,21] * yz32*yz11
    # (pz py/py s)
    w[...,83] = x[...,1] * (ri[...,7]*xx32 + ri[...,8]*zz32) \
          + ri[...,9] * (y[...,1]*xy32 + z[...,1]*xz32)
    # (pz py/py px)
    w[...,84] =  \
       ( ri[...,15] * xx32 + ri[...,16]*zz32 ) * xx21 \
       + ri[...,17] * xx32*yyzz21 \
       + ri[...,18] * zz32*zz21 \
       + ri[...,19] * (xy32*xy21 + xz32*xz21) \
       + ri[...,20] * zz32*yy21 \
       + ri[...,21] * yz32*yz21
    # (pz py/py py)
    w[...,85] =  \
       ( ri[...,15] * xx32 + ri[...,16]*zz32 ) * xx22 \
       + ri[...,17] * xx32*yyzz22 \
       + ri[...,18] * zz32*zz22 \
       + ri[...,19] * (xy32*xy22 + xz32*xz22) \
       + ri[...,20] * zz32*yy22 \
       + ri[...,21] * yz32*yz22
    # (pz py/pz s)
    w[...,86] = x[...,2] * (ri[...,7]*xx32 + ri[...,8]*zz32) \
          + ri[...,9] * z[...,2]*xz32
    # (pz py/pz px)
    w[...,87] =  \
       (ri[...,15]*xx32 + ri[...,16]*zz32 ) * xx31 \
       + (ri[...,17]*xx32 + ri[...,18]*zz32 ) * zz31 \
       + ri[...,19] * (xy32*xy31 + xz32*xz31) \
       + ri[...,21]*yz32*yz31
    # (pz py/pz py)
    w[...,88] =  \
       (ri[...,15]*xx32 + ri[...,16]*zz32 ) * xx32 \
       + (ri[...,17]*xx32 + ri[...,18]*zz32 ) * zz32 \
       + ri[...,19] * (xy32*xy32 + xz32*xz32) \
       + ri[...,21]*yz32*yz32
    # (pz py/pz pz)
    w[...,89] =  \
       (ri[...,15]*xx32 + ri[...,16]*zz32 ) * xx33 \
       + (ri[...,17]*xx32 + ri[...,18]*zz32 ) * zz33 \
       + ri[...,19]*xz32*xz33
    # (pz pz/s s)
    w[...,90] = ri[...,2]*xx33 + ri[...,3]*zz33
    # (pz pz/px s)
    w[...,91] = x[...,0] * (ri[...,7]*xx33 + ri[...,8]*zz33) \
          + ri[...,9] * z[...,0]*xz33
    # (pz pz/px px)
    w[...,92] =  \
       ( ri[...,15] * xx33 + ri[...,16]*zz33 ) * xx11 \
       + ri[...,17] * xx33*yyzz11 \
       + ri[...,18] * zz33*zz11 \
       + ri[...,19] * xz33*xz11 \
       + ri[...,20] * zz33*yy11
    # (pz pz/py s)
    w[...,93] = x[...,1] * (ri[...,7]*xx33 + ri[...,8]*zz33) \
          + ri[...,9] * z[...,1]*xz33
    # (pz pz/py px)
    w[...,94] =  \
       ( ri[...,15] * xx33 + ri[...,16]*zz33 ) * xx21 \
       + ri[...,17] * xx33*yyzz21 \
       + ri[...,18] * zz33*zz21 \
       + ri[...,19] * xz33*xz21 \
       + ri[...,20] * zz33*yy21
    # (pz pz/py py)
    w[...,95] =  \
       ( ri[...,15] * xx33 + ri[...,16]*zz33 ) * xx22 \
       + ri[...,17] * xx33*yyzz22 \
       + ri[...,18] * zz33*zz22 \
       + ri[...,19] * xz33*xz22 \
       + ri[...,20] * zz33*yy22
    # (pz pz/pz s)
    w[...,96] = x[...,2] * (ri[...,7]*xx33 + ri[...,8]*zz33) \
          + ri[...,9] * z[...,2]*xz33
    # (pz pz/pz px)
    w[...,97] =  \
       (ri[...,15]*xx33 + ri[...,16]*zz33 ) * xx31 \
       + (ri[...,17]*xx33 + ri[...,18]*zz33 ) * zz31 \
       + ri[...,19]*xz33*xz31
    # (pz pz/pz py)
    w[...,98] =  \
       (ri[...,15]*xx33 + ri[...,16]*zz33 ) * xx32 \
       + (ri[...,17]*xx33 + ri[...,18]*zz33 ) * zz32 \
       + ri[...,19]*xz33*xz32
    # (pz pz/pz pz)
    w[...,99] =  \
       (ri[...,15]*xx33 + ri[...,16]*zz33 ) * xx33 \
       + (ri[...,17]*xx33 + ri[...,18]*zz33 ) * zz33 \
       + ri[...,19]*xz33*xz33
    #
    """
    #
    css1,csp1,cpps1,cppp1 = core[1:,1]
    css2,csp2,cpps2,cppp2 = core[1:,2]
    e1b[1] = -css1
    #if(natorb(ni).eq.4) then
    if ni>1:
        # currently only s and p orbitals
        e1b[2] = -csp1 *x[1]
        e1b[3] = -cpps1*xx11-cppp1*yyzz11
        e1b[4] = -csp1 *x[2]
        e1b[5] = -cpps1*xx21-cppp1*yyzz21
        e1b[6] = -cpps1*xx22-cppp1*yyzz22
        e1b[7] = -csp1 *x[3]
        e1b[8] = -cpps1*xx31-cppp1*zz31
        e1b[9] = -cpps1*xx32-cppp1*zz32
        e1b[10] = -cpps1*xx33-cppp1*zz33

    e2a[1] = -css2
    #if(natorb(nj).eq.4) then
    if nj>1:
        e2a[2] = -csp2 *x[1]
        e2a[3] = -cpps2*xx11-cppp2*yyzz11
        e2a[4] = -csp2 *x[2]
        e2a[5] = -cpps2*xx21-cppp2*yyzz21
        e2a[6] = -cpps2*xx22-cppp2*yyzz22
        e2a[7] = -csp2 *x[3]
        e2a[8] = -cpps2*xx31-cppp2*zz31
        e2a[9] = -cpps2*xx32-cppp2*zz32
        e2a[10] = -cpps2*xx33-cppp2*zz33
    """
    """
    e1bHH = -coreHH[...,0]
    e2aHH = -coreHH[...,1]

    e1bXH = torch.zeros(coreXH.shape[0],10,dtype=dtype, device=device)
    e1bXH[...,1-1] = -coreXH[...,0]
    e1bXH[...,2-1] = -coreXH[...,1]*xXH[...,1-1]
    e1bXH[...,3-1] = -coreXH[...,2]*xx11XH - coreXH[...,3]*yyzz11XH
    e1bXH[...,4-1] = -coreXH[...,1]*xXH[...,2-1]
    e1bXH[...,5-1] = -coreXH[...,2]*xx21XH - coreXH[...,3]*yyzz21XH
    e1bXH[...,6-1] = -coreXH[...,2]*xx22XH - coreXH[...,3]*yyzz22XH
    e1bXH[...,7-1] = -coreXH[...,1]*xXH[...,3-1]
    e1bXH[...,8-1] = -coreXH[...,2]*xx31XH - coreXH[...,3]*zz31XH
    e1bXH[...,9-1] = -coreXH[...,2]*xx32XH - coreXH[...,3]*zz32XH
    e1bXH[...,10-1] = -coreXH[...,2]*xx33XH - coreXH[...,3]*zz33XH
    e2aXH = -coreXH[...,4]

    e1b = torch.zeros(core.shape[0],10,dtype=dtype, device=device)
    e2a = torch.zeros_like(e1b)
    e1b[...,1-1] = -core[...,0]
    e1b[...,2-1] = -core[...,1]*x[...,1-1]
    e1b[...,3-1] = -core[...,2]*xx11 - core[...,3]*yyzz11
    e1b[...,4-1] = -core[...,1]*x[...,2-1]
    e1b[...,5-1] = -core[...,2]*xx21 - core[...,3]*yyzz21
    e1b[...,6-1] = -core[...,2]*xx22 - core[...,3]*yyzz22
    e1b[...,7-1] = -core[...,1]*x[...,3-1]
    e1b[...,8-1] = -core[...,2]*xx31 - core[...,3]*zz31
    e1b[...,9-1] = -core[...,2]*xx32 - core[...,3]*zz32
    e1b[...,10-1] = -core[...,2]*xx33 - core[...,3]*zz33

    e2a[...,1-1] = -core[...,4]
    e2a[...,2-1] = -core[...,5]*x[...,1-1]
    e2a[...,3-1] = -core[...,6]*xx11 - core[...,7]*yyzz11
    e2a[...,4-1] = -core[...,5]*x[...,2-1]
    e2a[...,5-1] = -core[...,6]*xx21 - core[...,7]*yyzz21
    e2a[...,6-1] = -core[...,6]*xx22 - core[...,7]*yyzz22
    e2a[...,7-1] = -core[...,5]*x[...,3-1]
    e2a[...,8-1] = -core[...,6]*xx31 - core[...,7]*zz31
    e2a[...,9-1] = -core[...,6]*xx32 - core[...,7]*zz32
    e2a[...,10-1] = -core[...,6]*xx33 - core[...,7]*zz33

    return wHH, e1bHH, e2aHH, wXH, e1bXH, e2aXH, w, e1b, e2a
    """
    #combine w, e1b, e2a




    # as index_add_ is used later, which is slow, so
    # change e1b, e2a to shape (npairs, 4,4), only need to do index_add once
    """
    e1b = torch.zeros(rij.shape[0],10,dtype=dtype, device=device)
    e2a = torch.zeros_like(e1b)

    e1b[HH,0] = -coreHH[...,0]
    e2a[HH,0] = -coreHH[...,1]

    #e1bXH = torch.zeros(coreXH.shape[0],10,dtype=dtype, device=device)
    e1b[XH,1-1] = -coreXH[...,0]
    e1b[XH,2-1] = -coreXH[...,1]*xXH[...,1-1]
    e1b[XH,3-1] = -coreXH[...,2]*xx11XH - coreXH[...,3]*yyzz11XH
    e1b[XH,4-1] = -coreXH[...,1]*xXH[...,2-1]
    e1b[XH,5-1] = -coreXH[...,2]*xx21XH - coreXH[...,3]*yyzz21XH
    e1b[XH,6-1] = -coreXH[...,2]*xx22XH - coreXH[...,3]*yyzz22XH
    e1b[XH,7-1] = -coreXH[...,1]*xXH[...,3-1]
    e1b[XH,8-1] = -coreXH[...,2]*xx31XH - coreXH[...,3]*zz31XH
    e1b[XH,9-1] = -coreXH[...,2]*xx32XH - coreXH[...,3]*zz32XH
    e1b[XH,10-1] = -coreXH[...,2]*xx33XH - coreXH[...,3]*zz33XH
    e2a[XH,0] = -coreXH[...,4]

    #e1b = torch.zeros(core.shape[0],10,dtype=dtype, device=device)
    #e2a = torch.zeros_like(e1b)
    e1b[XX,1-1] = -core[...,0]
    e1b[XX,2-1] = -core[...,1]*x[...,1-1]
    e1b[XX,3-1] = -core[...,2]*xx11 - core[...,3]*yyzz11
    e1b[XX,4-1] = -core[...,1]*x[...,2-1]
    e1b[XX,5-1] = -core[...,2]*xx21 - core[...,3]*yyzz21
    e1b[XX,6-1] = -core[...,2]*xx22 - core[...,3]*yyzz22
    e1b[XX,7-1] = -core[...,1]*x[...,3-1]
    e1b[XX,8-1] = -core[...,2]*xx31 - core[...,3]*zz31
    e1b[XX,9-1] = -core[...,2]*xx32 - core[...,3]*zz32
    e1b[XX,10-1] = -core[...,2]*xx33 - core[...,3]*zz33

    e2a[XX,1-1] = -core[...,4]
    e2a[XX,2-1] = -core[...,5]*x[...,1-1]
    e2a[XX,3-1] = -core[...,6]*xx11 - core[...,7]*yyzz11
    e2a[XX,4-1] = -core[...,5]*x[...,2-1]
    e2a[XX,5-1] = -core[...,6]*xx21 - core[...,7]*yyzz21
    e2a[XX,6-1] = -core[...,6]*xx22 - core[...,7]*yyzz22
    e2a[XX,7-1] = -core[...,5]*x[...,3-1]
    e2a[XX,8-1] = -core[...,6]*xx31 - core[...,7]*zz31
    e2a[XX,9-1] = -core[...,6]*xx32 - core[...,7]*zz32
    e2a[XX,10-1] = -core[...,6]*xx33 - core[...,7]*zz33
    """

    e1b = torch.zeros((rij.shape[0], 4, 4),dtype=dtype, device=device)
    e2a = torch.zeros_like(e1b)

    e1b[HH,0,0] = -coreHH[...,0]
    e2a[HH,0,0] = -coreHH[...,1]

    #e1bXH = torch.zeros(coreXH.shape[0],10,dtype=dtype, device=device)
    e1b[XH,0,0] = -coreXH[...,0]
    e1b[XH,0,1] = -coreXH[...,1]*xXH[...,0]
    e1b[XH,1,1] = -coreXH[...,2]*xx11XH - coreXH[...,3]*yyzz11XH
    e1b[XH,0,2] = -coreXH[...,1]*xXH[...,1]
    e1b[XH,1,2] = -coreXH[...,2]*xx21XH - coreXH[...,3]*yyzz21XH
    e1b[XH,2,2] = -coreXH[...,2]*xx22XH - coreXH[...,3]*yyzz22XH
    e1b[XH,0,3] = -coreXH[...,1]*xXH[...,2]
    e1b[XH,1,3] = -coreXH[...,2]*xx31XH - coreXH[...,3]*zz31XH
    e1b[XH,2,3] = -coreXH[...,2]*xx32XH - coreXH[...,3]*zz32XH
    e1b[XH,3,3] = -coreXH[...,2]*xx33XH - coreXH[...,3]*zz33XH
    e2a[XH,0,0] = -coreXH[...,4]

    #e1b = torch.zeros(core.shape[0],10,dtype=dtype, device=device)
    #e2a = torch.zeros_like(e1b)
    e1b[XX,0,0] = -core[...,0]
    e1b[XX,0,1] = -core[...,1]*x[...,0]
    e1b[XX,1,1] = -core[...,2]*xx11 - core[...,3]*yyzz11
    e1b[XX,0,2] = -core[...,1]*x[...,1]
    e1b[XX,1,2] = -core[...,2]*xx21 - core[...,3]*yyzz21
    e1b[XX,2,2] = -core[...,2]*xx22 - core[...,3]*yyzz22
    e1b[XX,0,3] = -core[...,1]*x[...,2]
    e1b[XX,1,3] = -core[...,2]*xx31 - core[...,3]*zz31
    e1b[XX,2,3] = -core[...,2]*xx32 - core[...,3]*zz32
    e1b[XX,3,3] = -core[...,2]*xx33 - core[...,3]*zz33

    e2a[XX,0,0] = -core[...,4]
    e2a[XX,0,1] = -core[...,5]*x[...,0]
    e2a[XX,1,1] = -core[...,6]*xx11 - core[...,7]*yyzz11
    e2a[XX,0,2] = -core[...,5]*x[...,1]
    e2a[XX,1,2] = -core[...,6]*xx21 - core[...,7]*yyzz21
    e2a[XX,2,2] = -core[...,6]*xx22 - core[...,7]*yyzz22
    e2a[XX,0,3] = -core[...,5]*x[...,2]
    e2a[XX,1,3] = -core[...,6]*xx31 - core[...,7]*zz31
    e2a[XX,2,3] = -core[...,6]*xx32 - core[...,7]*zz32
    e2a[XX,3,3] = -core[...,6]*xx33 - core[...,7]*zz33


    wc  = torch.zeros(rij.shape[0], 10, 10, dtype=dtype, device=device)
    wc[HH,0,0] = wHH
    wc[XH,:,0] = wXH
    wc[XX] = w.reshape((-1,10,10))
    #print('w: ', wc)
    return wc, e1b, e2a
