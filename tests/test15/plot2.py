import numpy as np
import matplotlib.pyplot as plt
import sys
try:
    fn=sys.argv[1]
except:
    fn='log.dat'
d=np.loadtxt(fn)
##index, dx, fy, dfydx

t=np.argmin(d[:,2])
print(d[t,1])


t=np.argmin(np.abs(d[:,3]))
print(d[t,1])

dx = d[1,1]-d[0,1]

# 0         1             2          3                  4                     5                      6
#index, dx (Angstrom), Etot (eV), Fz_bf (eV/Angstrom), Fz_in (eV/Angstrom), Hzz_bf (eV/Angstrom^2), Hzz_in 
z = d[:,1]
E = d[:,2]
F_fd = -(E[2:] - E[:-2])/dx/2.
F_adE = d[:,3]
F_cal = d[:,4]
Hzz_fd = (F_fd[2:] - F_fd[:-2])/dx/2.
Hzz_fd_adE = (F_adE[2:] - F_adE[:-2])/dx/2.
Hzz_fd_cal = (F_cal[2:] - F_cal[:-2])/dx/2.
Hzz_ad2E = d[:,5]
Hzz_adF = d[:,6]
z_fd = z[1:-1]
z_fd2 = z_fd[1:-1]

plt.figure(0)
plt.plot(z_fd, F_fd, c='0.75', lw=4, label='FD')
plt.plot(z, F_adE, 'r', label='AD Energy')
plt.plot(z, F_cal, 'b--', label='Force calc')
plt.plot(z, np.zeros_like(z))
plt.legend()
plt.ylabel('Fz = dE/dz (eV/Angstrom)')
plt.xlabel('z (Angstrom)')
#plt.xlim([0.0,10])

plt.figure(1)
dF_adE = F_adE[1:-1] - F_fd
dF_cal = F_cal[1:-1] - F_fd
plt.plot(z_fd, np.zeros_like(z_fd), c='0.5', ls=':')
plt.plot(z_fd, dF_adE, 'r', label='AD Energy')
plt.plot(z_fd, dF_cal, 'g--', label='Force calc')
plt.legend()
plt.ylabel('Fz error (eV/Angstrom)')
plt.xlabel('z (Angstrom)')
#plt.xlim([0.0,10])


plt.figure(2)
plt.plot(z_fd2, Hzz_fd, c='0.75', lw=4, label='double FD')
plt.plot(z_fd, Hzz_fd_adE, 'r', label='FD AD Energy')
plt.plot(z_fd, Hzz_fd_cal, 'b--', label='FD Force calc')
plt.plot(z, Hzz_ad2E, 'g-.', label='double AD')
plt.plot(z, Hzz_adF, 'm:', marker='^', mfc='none', label='AD Force calc')
plt.legend()
plt.ylabel('Hzz = d^2 E / dz^2  (eV/Angstrom)')
plt.xlabel('z (Angstrom)')
#plt.xlim([0.0,10])

plt.figure(3)
dH_fd_adE = Hzz_fd_adE[1:-1] - Hzz_fd
dH_fd_cal = Hzz_fd_cal[1:-1] - Hzz_fd
dH_ad2E = Hzz_ad2E[2:-2] - Hzz_fd
dH_adF = Hzz_adF[2:-2] - Hzz_fd
plt.plot(z_fd2, dH_fd_adE, 'r', label='FD AD Energy')
plt.plot(z_fd2, dH_fd_cal, 'b--', label='FD Force calc')
plt.plot(z_fd2, dH_ad2E, 'g-.', label='double AD')
plt.plot(z_fd2, dH_adF, 'm:', marker='^', mfc='none', label='AD Force calc')
plt.legend()
plt.ylabel('error in Hzz = d^2 E / dz^2  (eV/Angstrom)')
plt.xlabel('z (Angstrom)')

plt.show()
