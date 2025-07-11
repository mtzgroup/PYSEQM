import torch


def params(method='MNDO', elements=[1,6,7,8],
           parameters=['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
                       'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'alpha'],
           root_dir='./params/MOPAC/'):
    """
    load parameters from AM1 PM3 MNDO
    """
    # method=MNDO, AM1, PM3
    # load the parameters taken from MOPAC
    # elements: elements needed, not checking on the type, but > 0 and <= 107
    # parameters: parameter lists
    # root_dir : directory for these parameter files
    fn=root_dir+"parameters_"+method+"_MOPAC.csv"
    #will directly use atomic number as array index
    #elements.sort()
    m=max(elements)
    n=len(parameters)
    p=torch.zeros((m+1,n))
    f=open(fn)
    header=f.readline().strip().replace(' ', '').split(',')
    idx = [header.index(item) for item in parameters]
    for l in f:
        t=l.strip().replace(' ', '').split(',')
        z=int(t[0])
        if z in elements:
            p[z,:] = torch.tensor([float(t[x]) for x in idx])
        if z > max(elements): break
    f.close()
    return torch.nn.Parameter(p, requires_grad=False)


def atom_params(method='MNDO', elements=[1,6,7,8],
                parameters=['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
                            'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'alpha'],
                root_dir='./params/MOPAC/'):
    """
    load parameters from AM1 PM3 MNDO
    """
    p_dict = {p_i:{} for p_i in parameters}
    f = open(root_dir + "parameters_" + method + "_MOPAC.csv")
    header = f.readline().strip().replace(' ', '').split(',')
    idx = [header.index(item) for item in parameters]
    for l in f:
        t = l.strip().replace(' ', '').split(',')
        z = int(t[0])
        if z in elements:
            for i_p, p_i in enumerate(parameters):
                p_dict[p_i][z] = float(t[idx[i_p]])
        if z > max(elements): break
    f.close()
    return p_dict


def pair_params(method='AM1_PDREP', elements=[1,6,7,8],
                parameters=['alpha', 'chi'], root_dir='./params/MOPAC/'):

    """
    load diatomic core-core parameters and returns them as dictionary
    """
    #will directly use atomic number as array index just as in the params method
    p_dict = {p_i:{int(e):{} for e in elements if e>0} for p_i in parameters}
    if method in ['PM6', 'PM6_SP', 'AM1_PDREP']:
        f = open(root_dir+"PWCCT_"+method+"_MOPAC.csv")
        for l in f:
            t = l.strip().replace(' ', '').split(',')
            z1, z2 = int(t[0]), int(t[1])
            my_par = {'alpha':float(t[2]), 'chi':float(t[3]), 'mu':float(t[4]), 'nu':float(t[5])}
            if z1 in elements and z2 in elements:
                for p_i in parameters:
                    p_dict[p_i][z1][z2] = p_dict[p_i][z2][z1] = my_par[p_i]
            if z1 > max(elements) and z2 > max(elements): break
        f.close()
    return p_dict

