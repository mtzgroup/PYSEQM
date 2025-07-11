import torch
from torch.nested import as_nested_tensor


def tuple2nested(x): return as_nested_tensor(list(x))

def nestedmap_unary(f, x):
    ## update once complex supported in `as_nested_tensor`
    return torch.nested.nested_tensor(list(map(f, x)))

def nestedmap_binary(f, x, y):
    ## update once complex supported in `as_nested_tensor`
    return torch.nested.nested_tensor(list(map(f, x, y)))

def eckart(masses, coordinates):
    M_mol = masses.sum(dim=(1,2))
    COMs = (masses * coordinates).sum(dim=1) / M_mol
    R0 = coordinates - COMs
    I = torch.einsum("bij,bik->bjk", masses * R0, R0)
    _, X = torch.linalg.eigh(I / M_mol)
    R_Eckart = R0.bmm(X)
    return X, R_Eckart

def vibrational_basis(masses, coordinates):
    # get Eckart frame
    X, R_Eck = eckart(masses, coordinates)
    # Eigenbasis for translations and rotations
    T1 = torch.eye(3, device=coordinates.device, dtype=coordinates.dtype)
    G_Eck = R_Eck * masses.sqrt()
    R = torch.stack([torch.cat([torch.stack([torch.linalg.cross(gij, xi[i]) for i in range(3)])
                     for gij in gi]) for (gi, xi) in zip(G_Eck, X)])
    TR = torch.cat((T1.repeat(*coordinates.shape[:2], 1), R), 2)
    # Eigenspace of global TR = net translations and rotations
    # -> remove subspace corresponding to the 6 (5) non-zero
    #    singular values for non-linear (linear) molecules
    U, S, _ = torch.linalg.svd(TR, full_matrices=True)
    nn0 = S.round_(decimals=15).count_nonzero(dim=-1)
    B = as_nested_tensor([Ui[:,n:] for (Ui,n) in zip(U, nn0)])
    return B

def harmonic_analysis(masses, coordinates, hessian):
    im = masses.pow(-0.5).repeat_interleave(3, dim=1).squeeze(-1)
#    Hm = torch.einsum('bi,bj->bij', (m, m)) * hessian
    Hm = as_nested_tensor([im_i.unsqueeze(-1) * Hi * im_i for (im_i,Hi) in zip(im, hessian)])
    B = vibrational_basis(masses, coordinates)
    BTHB = B.transpose(-1,-2).bmm(Hm.bmm(B))
#    e, v = torch.linalg.eigh(BTHB)
    # update once `eigh` supported by nested
    e, v = map(tuple2nested, zip(*map(torch.linalg.eigh, BTHB)))
    U = B.bmm(v)
    M = (im_i.unsqueeze(-1) for im_i in im)
    Q = nestedmap_binary(torch.mul, U, M)
#    Q = U * torch.einsum('bi,bj->bij', (im, torch.ones_like(U[:,0,:])))
    return e, Q.transpose(-1,-2)

def curve2freq(x):
    # update once `type` and `sqrt` supported by nested
    freq = x.type(torch.complex64).sqrt()
    ## ignore imaginary freqencies small than 1 cm^-1
    if (freq.imag.abs() < 1.9176525463471034e-3).all(): freq = freq.real
    return freq * 0.06465415105180661 ## curvature to frequency

