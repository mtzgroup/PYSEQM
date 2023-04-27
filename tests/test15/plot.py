import numpy as np
import matplotlib.pyplot as plt
import sys
try:
    fn0=sys.argv[1]
    fn1=sys.argv[2]
    fn2=sys.argv[3]
except:
    fn0='log0.dat'
    fn1='log1.dat'
    fn2='log2.dat'

d=np.loadtxt(fn0)
d1=np.loadtxt(fn1)
d2=np.loadtxt(fn2)
##index, dx, fy, dfydx

t=np.argmin(d[:,2])
print(d[t,1])


t=np.argmin(np.abs(d[:,3]))
print(d[t,1])

#-0.122

dx = d[1,1]-d[0,1]

x1 = d[1:-1,1]
# FD force
f0 = -(d[2:,4]-d[:-2,4])/dx/2.0
# FD Hessian
f1 = (d[2:,2]-d[:-2,2])/dx/2.0
plt.figure(0)
plt.plot(d[:,1],d[:,3],'r',label='no SCFbackward')
plt.plot(d1[:,1],d1[:,3],'b',label='recursive formula')
plt.plot(d2[:,1],d2[:,3],'g',label='direct backprop')
plt.plot(x1, f1, 'k--', label='numerical diff')
plt.plot(d[:,1], np.zeros_like(d[:,1]), c='0.5', ls=':')
plt.legend()
plt.ylabel('dfz_dz = d^2E/dz^2 (eV/Angstrom^2)')
plt.xlabel('z (Angstrom)')
#plt.xlim([0.0,10])

plt.figure(1)
e = d[1:-1,3]-f1
e1 = d1[1:-1,3]-f1
e2 = d2[1:-1,3]-f1
plt.plot(x1, e, 'r',label='no SCFbackward')
plt.plot(x1, e1, 'b',label='recursive formula')
plt.plot(x1, e2, 'g',label='direct backprop')
plt.ylabel('|dfz_dz error| (eV/Angstrom^2)')
plt.xlabel('z (Angstrom)')
plt.legend()
#plt.xlim([0.0,10])

plt.figure(2)
plt.plot(x1, e/f1, 'r',label='no SCFbackward')
plt.plot(x1, e1/f1, 'b',label='recursive formula')
plt.plot(x1, e2/f1, 'g',label='direct backprop')
plt.ylabel('dfz_dz rel. error (eV/Angstrom^2)')
plt.xlabel('z (Angstrom)')
plt.legend()
#plt.xlim([0.0,10])


plt.figure(3)
plt.plot(d[:,1], d[:,2], 'r')
plt.plot(d1[:,1], d1[:,2], 'b')
plt.plot(d2[:,1], d2[:,2], 'g')
plt.plot(d[1:-1,1], f0, 'k--')
plt.ylabel('fz (eV/Angstrom)')
plt.xlabel('z (Angstrom)')
#plt.xlim([0.0,10])

plt.figure(4)
e = d[1:-1,2]-f0
e1 = d1[1:-1,2]-f0
e2 = d2[1:-1,2]-f0
plt.plot(x1, e, 'r',label='no SCFbackward')
plt.plot(x1, e1, 'b:',label='recursive formula')
plt.plot(x1, e2, 'g--',label='direct backprop')
plt.ylabel('|fz error| (eV/Angstrom)')
plt.xlabel('z (Angstrom)')
plt.legend()
#plt.xlim([0.0,10])

plt.show()
