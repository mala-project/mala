import random
import scipy.integrate as integrate
import numpy as np
import torch

__author__  = "Joshua A. Rackers"

def parse_basisfile(basisfile):
    widthO = []
    widthH = []
    readnextline = False
    with open(basisfile,'r') as f:
        for line in f.readlines():
            if line.strip():
                tag = line.split()[0]
                if readnextline:
                    scinotation = line.split()[0].replace("D", "E")
                    if (mode == 'O'):
                        widthO.append([float(scinotation), orbital_type])
                    if (mode == 'H'):
                        widthH.append([float(scinotation), orbital_type])
                    readnextline = False
                if (tag == 'S') or (tag == 'SP'):
                    readnextline = True
                    orbital_type = tag
                if (tag == 'O'):
                    mode = 'O'
                if (tag == 'H'):
                    mode = 'H'
    return widthO, widthH


def parse_whole_basisfile(basisfile):
    widthO = []
    widthH = []
    readnextline = False
    with open(basisfile,'r') as f:
        for line in f.readlines():
            if line.strip():
                tag = line.split()[0]
                if readnextline:
                    scinotation = line.split()[0].replace("D", "E")
                    if (mode == 'O'):
                        widthO.append([float(scinotation), orbital_type])
                    if (mode == 'H'):
                        widthH.append([float(scinotation), orbital_type])
                    readnextline = False
                if tag in ('S','SP','P','D'):
                    readnextline = True
                    orbital_type = tag
                if (tag == 'O'):
                    mode = 'O'
                if (tag == 'H'):
                    mode = 'H'
    return widthO, widthH

def parse_whole_normfile(normfile):
    normO = []
    normH = []
    readnextline = False
    with open(normfile,'r') as f:
        for line in f.readlines():
            if line.strip():
                tag = line.split()[0]
                if (tag == 'O'):
                    mode = 'O'
                    readnextline = True
                elif (tag == 'H'):
                    mode = 'H'
                    readnextline = True
                else:
                    if readnextline:
                        if (mode == 'O'):
                            normO.append(float(line.split()[0]))
                        if (mode == 'H'):
                            normH.append(float(line.split()[0]))
    return normO, normH

def integrate_yo_stuff(basis,coeffs):
    total = 0.0
    wO, wH = basis
    widths = []
    for a in coeffs:
        # based on the number of S orbitals (assumes a2 basis)
        if len(a) == 8:
            widths.append(wO)
        elif len(a) == 4:
            widths.append(wH)
    for i, atom in enumerate(widths):
        for j, value in enumerate(atom):
            w = value[0]
            orbital_type = value[1]
            c = coeffs[i][j]

            ## not necessary. there are no 2s GTOs!
            #if orbital_type == 'S':
            #    integrand = lambda r:gaussian(c,w)(r)*r**2
            #elif orbital_type == 'SP':
            #    integrand = lambda r:gaussian(c,w)(r)*r**4

            #integrand = lambda r:gaussian(c,w)(r)*r**2
            #integral, error = integrate.quad(integrand,0.0,np.inf)
            
            # integral(r^2*exp(-ar^2)) = 1/(4w) * sqrt(pi/w)
            
            normalization = (2*w/(np.pi))**(0.75)
            integral = c*normalization*(1/(4*w))*np.sqrt(np.pi/w)
            
            # not needed, y00 = 1
            #y00 = 0.5*np.sqrt(1/np.pi)
            #solidharmonic = np.sqrt(4*np.pi)
            
            space = 4*np.pi
            num_ele = integral*space
            total += num_ele
            #print (j,num_ele)
        #print (total)
    return total


def testnumelectrons(net,device,n_dimers,basisset,dataset_onehot,dataset_geom,dataset_typemap,coeff_by_type):
    print('\nNow testing number of electrons on {0} randomly selected dimers'.format(n_dimers))

    basisfuncs = parse_basisfile(basisset)
    for i in range(n_dimers):
        j = random.randint(3000, len(dataset_geom) - 1)
        onehot = dataset_onehot[j]
        points = dataset_geom[j]
        atom_type_map = dataset_typemap[j]
        outputO, outputH = net(onehot.to(device),points.to(device),atom_type_map)
        outputO = outputO.squeeze().tolist()
        outputH = outputH.squeeze().tolist()
        newoutputO = []
        for item in outputO:
            newitem = item[:8]
            newoutputO.append(newitem)
        newoutputH = []
        for item in outputH:
            newitem = item[:4]
            newoutputH.append(newitem)
        newoutputO.extend(newoutputH)
        ml_coeffs = newoutputO

        #with open('ml_coeffs_' + str(i) + '_.dat','wb') as f:
        #    pickle.dump(ml_coeffs, f)

        coeffs = coeff_by_type[j]
        s_coeffs = []
        for atom in coeffs:
            atom_s_coeffs = []
            for shell in atom:
                if len(shell) == 1:
                    atom_s_coeffs.append(shell[0])
            s_coeffs.append(atom_s_coeffs)

        #with open('real_coeffs_' + str(i) + '_.dat','wb') as f:
        #    pickle.dump(s_coeffs, f)

        true_tot = integrate_yo_stuff(basisfuncs,s_coeffs)
        ml_tot = integrate_yo_stuff(basisfuncs,ml_coeffs)

        print('\nNumber of electrons for structure {0}:'.format(j))
        print('   True: {0}'.format(true_tot*2))
        print('     ML: {0}'.format(ml_tot*2))


def get_exponents(basisfile):
    alphaO, alphaH = parse_whole_basisfile(basisfile)
    ## add in the duplicate entries for the SP basis functions
    # oxygen
    sp_duplicates = [item for item in alphaO if 'SP' in item]
    last_index = max(idx for idx, val in enumerate(alphaO) if 'SP' in val)
    alphaO[last_index+1:last_index+1] = sp_duplicates
    alphaO = torch.FloatTensor([item[0] for item in alphaO])
    # hydrogen
    sp_duplicates = [item for item in alphaH if 'SP' in item]
    last_index = max(idx for idx, val in enumerate(alphaH) if 'SP' in val)
    alphaH[last_index+1:last_index+1] = sp_duplicates
    alphaH = torch.FloatTensor([item[0] for item in alphaH])
    return alphaO, alphaH

def get_spherical_harmonic_norms(Rs_out_O,Rs_out_H):
    # copied from e3nn spherical harmonics cuda kernel
    def spherical_harmonic_norm(l,m):
        rsqrt_pi = 1/np.sqrt(np.pi)
        if l == 0:
            norm = rsqrt_pi/2
        if l == 1:
            norm = rsqrt_pi*np.sqrt(3)/2
        if l == 2:
            if (m == -2 or m == 2):
                norm = rsqrt_pi*np.sqrt(15)/4
            elif (m == -1 or m == 1):
                norm = rsqrt_pi*np.sqrt(15)/2
            elif m == 0:
                norm = (-rsqrt_pi*np.sqrt(5)/4)*(rsqrt_pi*np.sqrt(5)*3/4)
        return norm

    sph_normsO = np.empty(0)
    for mul, L in Rs_out_O:
        for i in range(mul):
            for j in range(2*L+1):
                sph_normsO = np.append(sph_normsO,(spherical_harmonic_norm(L,j-L)))

    sph_normsH = np.empty(0)
    for mul, L in Rs_out_H:
        for i in range(mul):
            for j in range(2*L+1):
                sph_normsH = np.append(sph_normsH,(spherical_harmonic_norm(L,j-L)))
    
    return sph_normsO,sph_normsH