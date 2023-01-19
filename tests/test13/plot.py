import numpy as np
import matplotlib.pyplot as plt
import sys
#try:
#    fn = sys.argv[1]
#except:
#    fn = 'log.dat'

systems = ['nondegen','degen']
solvers = ['old','new']
colors = ['tab:red','tab:green','darkorange','tab:blue']
legpos = [['lower left','lower left'],['upper left','center right'],['lower left','center right']]

for m in systems:
    for s in solvers:
        l = m+'_'+s
        data = np.loadtxt(l+'.dat')
        ##index, par, e(orb), ADgrad
        i, p, e, g = data.T
        exec('p'+l+' = p[::-1]')
        exec('e'+l+' = e[::-1]')
        exec('g'+l+' = g[::-1]')
        exec('dp'+l+' = p'+l+'[1] - p'+l+'[0]')
        exec('fd'+l+' = (e'+l+'[2:] - e'+l+'[:-2]) / (2.*dp'+l+')')
        exec('df'+l+' = g'+l+'[1:-1] - fd'+l)

for i, m in enumerate(systems):
    ax = plt.subplot2grid((4,2),(0,i))
    for j, s in enumerate(solvers):
        my_p, my_e = eval('p'+m+'_'+s), eval('e'+m+'_'+s)
        ax.plot(my_p, my_e, lw=1.5, ls='-', c=colors[2*j], 
                label=m+'_'+s)
    ax.set_ylabel(r'$\varepsilon\mathsf{(LUMO)\,\,[eV]}$')
    ax.set_xlabel(r'$\mathsf{p\,\,[arb.u.]}$')
    ax.set_xlim(my_p[0],my_p[-1])
    ax.legend(loc=legpos[0][i], prop={'size':11})

ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")


for i, m in enumerate(systems):
    ax = plt.subplot2grid((4,2),(1,i), rowspan=2)
    for j, s in enumerate(solvers):
        l = m+'_'+s
        my_p, my_g, my_fd = eval('p'+l), eval('g'+l), eval('fd'+l)
        ax.plot(my_p, 100*my_g, lw=1.5, ls='-', c=colors[2*j], 
                label='AD('+l+')')
        ax.plot(my_p[1:-1], 100*my_fd, lw=1.5, ls='--', c=colors[2*j+1], 
                label='FD('+l+')')
    ax.set_ylabel(r'$100\cdot\,\partial \varepsilon\mathsf{(LUMO)\,/\,\partial p\,\,[arb.u.]}$')
    ax.set_xlabel(r'$\mathsf{p\,\,[arb.u.]}$')
    ax.set_xlim(my_p[0],my_p[-1])
    ax.legend(loc=legpos[1][i], prop={'size':11})

ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")


for i, m in enumerate(systems):
    ax = plt.subplot2grid((4,2),(3,i))
    for j, s in enumerate(solvers):
        my_p, my_df = eval('p'+m+'_'+s), eval('df'+m+'_'+s)
        ax.plot(my_p[1:-1], 100*my_df, lw=1.5, ls='-', c=colors[2*j], 
        label=m+'_'+s)
    ax.set_ylabel(r'$\mathsf{100\cdot(AD - FD)\,\,[arb.u.]}$')
    ax.set_xlabel(r'$\mathsf{p\,\,[arb.u.]}$')
    ax.set_xlim(my_p[0],my_p[-1])
    ax.legend(loc=legpos[2][i], prop={'size':11})

ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")


fig = plt.gcf()
fig.set_size_inches(7.5,7)
fig.subplots_adjust(left=0.115, top=0.99, right=0.915, bottom=0.07,
                    hspace=0., wspace=0.025)
fig.savefig('xitorch-solver-test.pdf')
plt.close()
