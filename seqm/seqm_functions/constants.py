import torch

ev = 27.21 #used in mopac7
#1 hatree = 27.211386 eV
#ev =  27.211386

a0=0.529167  #used in mopac7
#a0=0.5291772109
ev_kcalpmol = 23.061  # 1 eV = 23.061 kcal/mol

"""
in mopac the cutoff for overlap is 10 Angstrom
Atomic units is used here
set cutoff = 20.0 Angstrom and 20/0.529167 = 37.8
"""
overlap_cutoff = 40.0

class Constants(torch.nn.Module):
    """
    Constants used in seqm
    """

    def __init__(self, length_conversion_factor=(1.0/a0), energy_conversion_factor=1.0, do_timing=False):
        """
        Constructor
        length_conversion_factor : atomic unit is used for length inside seqm
            convert the length by  oldlength*length_conversion_factor  to atomic units
            default value assume Angstrom used outside, and times 1.0/bohr_radius
        energy_conversion_factor : eV usedfor energy inside sqem
            convert by multiply energy_conversion_factor
            default value assumes eV used outside
        """

        super().__init__()

        # atomic unit for length is used in seqm
        # 1.8897261246364832 = 1.0/0.5291772109  1.0/bohr radius
        # factor convert length to atomic unit (default is from Angstrom to atomic unit)
        self.length_conversion_factor = length_conversion_factor
        #self.a0 = 0.529167  #used in mopac7
        #self.a0=0.5291772109

        # factor converting energy to eV (default is eV)
        self.energy_conversion_factor = energy_conversion_factor
        #self.ev = 27.21 #used in mopac7
        #1 hatree = 27.211386 eV
        #self.ev =  27.211386
        #
        # valence shell charge for each atom type
        #tore[1]=1, Hydrogen has 1.0 charge on valence shell
        self.label=['0',
               'H', 'He',
               'Li','Be',' B',' C',' N',' O',' F','Ne',
               'Na','Mg','Al','Si',' P',' S','Cl','Ar',]
        tore=torch.as_tensor([0.0,
                              1.0,                        0.0,
                              1.0,2.0,3.0,4.0,5.0,6.0,7.0,0.0,
                              1.0,2.0,3.0,4.0,5.0,6.0,7.0,0.0,])
        #
        #principal quantum number for valence shell
        # qn[1] = 1, principal quantum number for the valence shell of Hydrogen is 1
        qn = torch.as_tensor([0.0,
                              1.0,                        0.0,
                              2.0,2.0,2.0,2.0,2.0,2.0,2.0,0.0,
                              3.0,3.0,3.0,3.0,3.0,3.0,3.0,0.0])
        #
        qn_int = qn.type(torch.int64)
        #number of s electrons for each element
        ussc=torch.as_tensor([0.0,
                              1.0,                        0.0,
                              1.0,2.0,2.0,2.0,2.0,2.0,2.0,0.0,
                              1.0,2.0,2.0,2.0,2.0,2.0,2.0,0.0,])
        #
        #number of p electrons for each element
        uppc=torch.as_tensor([0.0,
                              0.0,                        0.0,
                              0.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,
                              0.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,])
        #
        #
        gssc=torch.as_tensor([0.0,
                              0.0,                        0.0,
                              0.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,
                              0.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,])
        #
        #
        gspc=torch.as_tensor([0.0,
                              0.0,                         0.0,
                              0.0,0.0,2.0,4.0,6.0,8.0,10.0,0.0,
                              0.0,0.0,2.0,4.0,6.0,8.0,10.0,0.0,])
        #
        hspc=torch.as_tensor([0.0,
                              0.0,                             0.0,
                              0.0,0.0,-1.0,-2.0,-3.0,-4.0,-5.0,0.0,
                              0.0,0.0,-1.0,-2.0,-3.0,-4.0,-5.0,0.0,])
        #
        gp2c=torch.as_tensor([0.0,
                              0.0,                         0.0,
                              0.0,0.0,0.0,1.5,4.5,6.5,10.0,0.0,
                              0.0,0.0,0.0,1.5,4.5,6.5,10.0,0.0,])
        #
        gppc=torch.as_tensor([0.0,
                              0.0,                           0.0,
                              0.0,0.0,0.0,-0.5,-1.5,-0.5,0.0,0.0,
                              0.0,0.0,0.0,-0.5,-1.5,-0.5,0.0,0.0,])
        #
        #heat of formation for each individual atom
        #experimental value, taken from block.f
        #unit kcal/mol
        eheat=torch.as_tensor([ 0.000,
                               52.102,                                                    0.0,
                               38.410, 76.960, 135.700, 170.890, 113.000, 59.559, 18.890, 0.0,
                               25.850, 35.000,  79.490, 108.390,  75.570, 66.400, 28.990, 0.0,])
        #
        #mass of atom
        mass=torch.as_tensor([ 0.00000,
                               1.00790,                                                              4.00260,
                               6.94000,  9.01218, 10.81000, 12.01100, 14.00670, 15.99940, 18.99840, 20.17900,
                              22.98977, 24.30500, 26.98154, 28.08550, 30.97376, 32.06000, 35.45300, 39.94800, ])
        
        
        # electronegativity (for EEQ charges) from Caldeweyher et al. J. Chem. Phys. 150, 154122 (2019).
        chi = torch.tensor([
            0.00000000,
            1.23695041,                                                                         1.26590957, 
            0.54341808, 0.99666991, 1.26691604, 1.40028282, 1.55819364, 1.56866440, 1.57540015, 1.15056627,
            0.55936220, 0.72373742, 1.12910844, 1.12306840, 1.52672442, 1.40768172, 1.48154584, 1.31062963,])
#            0.40374140, 0.75442607, 
#                    0.76482096, 0.98457281, 0.96702598, 1.05266584, 0.93274875,
#                    1.04025281, 0.92738624, 1.07419210, 1.07900668, 1.04712861,
#                                    1.15018618, 1.15388455, 1.36313743, 1.36485106, 1.39801837, 1.18695346,
#            0.36273870, 0.58797255, 
#                    0.71961946, 0.96158233, 0.89585296, 0.81360499, 1.00794665,
#                    0.92613682, 1.09152285, 1.14907070, 1.13508911, 1.08853785,
#                                    1.11005982, 1.12452195, 1.21642129, 1.36507125, 1.40340000, 1.16653482,])
        
        # chemical hardness (for EEQ charges) from Caldeweyher et al. J. Chem. Phys. 150, 154122 (2019).
        eta = torch.tensor([
            0.00000000,
           -0.35015861,                                                                         1.04121227,
            0.09281243, 0.09412380, 0.26629137, 0.19408787, 0.05317918, 0.03151644, 0.32275132, 1.30996037,
            0.24206510, 0.04147733, 0.11634126, 0.13155266, 0.15350650, 0.15250997, 0.17523529, 0.28774450,])
#            0.42937314, 0.01896455,
#                    0.07179178, -0.01121381, -0.03093370,  0.02716319, -0.01843812,
#                   -0.15270393, -0.09192645, -0.13418723, -0.09861139,  0.18338109,
#                                   0.08299615, 0.11370033, 0.19005278, 0.10980677, 0.12327841, 0.25345554,
#            0.58615231, 0.16093861, 
#                    0.04548530, -0.02478645, 0.01909943, 0.01402541, -0.03595279,
#                    0.01137752, -0.03697213, 0.08009416, 0.02274892,  0.12801822,
#                                  -0.02078702, 0.05284319,  0.07581190, 0.09663758, 0.09547417, 0.07803344,])
        
        self.tore   = torch.nn.Parameter(tore,   requires_grad=False)
        self.qn     = torch.nn.Parameter(qn,     requires_grad=False)
        self.qn_int = torch.nn.Parameter(qn_int, requires_grad=False)
        self.ussc   = torch.nn.Parameter(ussc,   requires_grad=False)
        self.uppc   = torch.nn.Parameter(uppc,   requires_grad=False)
        self.gssc   = torch.nn.Parameter(gssc,   requires_grad=False)
        self.gspc   = torch.nn.Parameter(gspc,   requires_grad=False)
        self.hspc   = torch.nn.Parameter(hspc,   requires_grad=False)
        self.gp2c   = torch.nn.Parameter(gp2c,   requires_grad=False)
        self.gppc   = torch.nn.Parameter(gppc,   requires_grad=False)
        self.eheat  = torch.nn.Parameter(eheat/ev_kcalpmol,  requires_grad=False)
        self.mass   = torch.nn.Parameter(mass,   requires_grad=False)
        self.do_timing = do_timing
        if self.do_timing:
            self.timing = {"Hcore + STO Integrals" : [],
                           "SCF"                   : [],
                           "Force"                 : [],
                           "MD"                    : [],
                           "D*"                    : []
                          }


    def forward(self):
        pass
        """
        return self.length_conversion_factor, \
               self.energy_conversion_factor, \
               self.tore, \
               self.qn, \
               self.qn_int
        """
    
