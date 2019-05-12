import numpy as np
from scipy import integrate
import matplotlib.pyplot as pl
import tensors as ts
from mpl_toolkits.mplot3d import Axes3D

# the energy density at a given omega to each direction

def return_angle(om,n):
    de = []
    for i in range(n+1):
        de.append(ts.grav_wave_energy_simps((np.pi/(2*n)*i),om))
        if ((i+1) % 5 == 0):
            print('omega = '+str(om)+': '+str(i+1)+'/'+str(n)+' points')
    return de


def test_antenna(om,n,**plot):
    de = return_angle(om,n)
    f = open(str(om)+".dat",'w')
    f.write(str(de))
    f.close()
    t = []
    for i in range(n+1):
        t.append(np.pi/(2*n)*i)

    for i in range(n+1):
        de.append(de[n-i])
        t.append(np.pi-t[n-i])

    x = []
    y = []
    for i in range(2*n+1):
        x.append(de[i]*np.cos(t[i]))
        y.append(de[i]*np.sin(t[i]))

    if ('line' in plot):
        pl.plot(x,y,plot['line'])
    else:
      pl.plot(x,y)

def totalrad(om):
    integrand = lambda xi : ts.grav_wave_energy_simps(xi,om)
    x = []
    y = []
    for i in range(20):
        x.append(np.pi*i/100)
        y.append(integrand(x[i]))
    return (integrate.simps(y,x),0.1)


def spectrum():
    oms = []
    rads = []
    errlow = []
    errhigh = []
    f = open("spectrum.dat",'w')
    for i in range(50):
        oms.append(0.003*i)
        print "step "+str(i)
        y,err = totalrad(oms[i])
        rads.append(y)
        errlow.append(y-err)
        errhigh.append(y+err)
        f.write(str(oms[i])+"\t"+str(y)+"\t"+str(err)+'\n')
    pl.plot(oms,rads)
    pl.show()
    f.close()

#

def test_antenna_n(minom,maxom,n,m):
    diff = (minom-maxom)/n
    for i in range(m):
        test_antenna(minom+i*diff,n)
        print(str(i+1)+"/"+str(m)+" plots done\n")
    pl.show()

# 3D plot of the waves

def plot_3d(om,n):
    de = []
    th = []
    for i in range(n):
        de.append(ts.grav_wave_energy(np.pi/n*i,om))
        th.append(np.pi/n*i)

    x = []
    y = []
    z = []

    for i in range(2*n+1):
        t = th[i]
        r = de[i]
        zi = np.cos(t)*r
        psi = np.linspace(-np.pi,np.pi,n)
        for p in psi:
            y.append(r*np.sin(p)*np.sin(t))
            x.append(r*np.cos(p)*np.sin(t))
            z.append(zi)

    fig = pl.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_trisurf(x,y,z)
    pl.show()


# plot antenna pattern for 3 values of omega

def plot_3om(omt1,omt2,omt3):
    om1 = omt1/ts.d()
    om2 = omt2/ts.d()
    om3 = omt3/ts.d()
    test_antenna(om1,50,line='k')
    test_antenna(om2,50,line='k:')
    test_antenna(om3,50,line='k--')
    pl.legend([r'$\omega\tau$='+str(omt1),r'$\omega\tau$='+str(omt2),r'$\omega\tau$='+str(omt3)])
    pl.title(r'Cutoff function: $C_1(t)$')
    pl.axis('off')
    pl.savefig('latest.png')
    pl.show()

spectrum()
# plot_3om(2.2,5.0,11.0)
