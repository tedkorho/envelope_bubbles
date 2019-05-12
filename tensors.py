import numpy as np
from scipy import integrate
from scipy import special
import pylab as pl

# initial distance between the bubbles

def d():
    return 60.0

# completion time tau of phase transition

def tau():
    return 1.2*d()


# cutoff function

def cutoff(t):
    #if (t < 0.9*tau()):
    #    return 1.0
    #else:
    #    return 0.0
    if (t < 0.9*tau()):
        return 1.0
    else:
        return np.exp(-(t-0.9*tau())**2/(0.025*tau())**2)


# max. degree of Chebyshev polynomials used in upcoming
# integrals

def maxp():
    return 25


# error tolerance in integration

def relt():
    return 0.01


# limit to which

def alpha(t):
    if (t<=0.5*d()):
        return 0.
    else:
        return np.arccos(d()/(2.*t))


# Creates a single variable integrand for integration over theta

def tensor_integrand(xi,om,t,selection):
    kx = om*np.sin(xi)
    ky = 0
    kz = om*np.cos(xi)

    j0 = lambda x : special.jn(0,kx*t*np.sin(x))
    j1 = lambda x : special.jn(1,kx*t*np.sin(x))
    j2 = lambda x : special.jn(2,kx*t*np.sin(x))

    if (selection == "txx"):
        return lambda th : (np.sin(th))**3*(np.cos(kz*t*np.cos(th)+0.5*kz*d()))*(j0(th)-j2(th))
    if (selection == "tyy"):
        return lambda th : (np.sin(th))**3*(np.cos(kz*t*np.cos(th)+0.5*kz*d()))*(j0(th)+j2(th))
    if (selection == "tzz"):
        return lambda th : (np.sin(th)*np.cos(th)**2*(np.cos(kz*t*np.cos(th)+0.5*kz*d()))*j0(th))
    if (selection == "txz"):
        return lambda th : (np.sin(th)**2*np.cos(th)*(np.cos(kz*t*np.cos(th)+0.5*kz*d()))*(j1(th)))
    return 0


def tensor_integrand_simpson(xi,om,t,selection):
    N = 20
    h = (np.pi-alpha(t))/N
    theta = np.empty(N)
    y = np.empty(N)
    f = tensor_integrand(xi,om,t,selection)
    for i in range(N):
        theta[i] = h*i
        y[i] = f(theta[i])

    return integrate.simps(y,theta)*t**3*cutoff(t)

def tensor_simpson(xi,om,selection):
    N = 60
    h = (tau()/N)
    t = np.empty(N)
    yr = np.empty(N)
    yim = np.empty(N)
    for i in range(N):
        t[i] = h*i
        yr[i] = tensor_integrand_simpson(xi,om,t[i],selection)*np.cos(om*t[i])
        yim[i] = tensor_integrand_simpson(xi,om,t[i],selection)*np.sin(om*t[i])

    return 1.j*integrate.simps(yim,t)+integrate.simps(yr,t)

def grav_wave_energy_simps(xi,om):
    txx = tensor_simpson(xi,om,"txx")
    tyy = tensor_simpson(xi,om,"tyy")
    tzz = tensor_simpson(xi,om,"tzz")
    txz = tensor_simpson(xi,om,"txz")
    return om**2*np.abs(( \
            tzz*np.sin(xi)**2+txx*np.cos(xi)**2-tyy-2*txz*np.sin(xi)*np.cos(xi)))**2




# the gravitational wave energy is given by Gw^2|Txx+Tyy+Tzz+Txz|^2, with Tij
# multiplied by various trigonometric functions.
# integrating over t, this is the imaginary part of the integrand

def tensor_integrand_2_im(xi,om,selection):
    #int1 = lambda t : integrate.quad(tensor_integrand(xi,om,t,selection),0,np.pi-alpha(t),maxp1=maxp(),epsrel=relt())
    int1 = lambda t : integrate.romberg(tensor_integrand(xi,om,t,selection),0,np.pi-alpha(t),rtol=relt())

    return lambda t : t**3*np.sin(om*t)*cutoff(t)*int1(t)

# ... and this is the real part

def tensor_integrand_2_r(xi,om,selection):
    #int1 = lambda t : integrate.quad(tensor_integrand(xi,om,t,selection),0,np.pi-alpha(t),maxp1=maxp(),epsrel=relt())
    int1 = lambda t : integrate.romberg(tensor_integrand(xi,om,t,selection),0,np.pi-alpha(t),rtol=relt())
    return lambda t : t**3*np.cos(om*t)*cutoff(t)*int1(t)




def grav_wave_energy(xi,om):
    txx_r = integrate.quad(tensor_integrand_2_r(xi,om,"txx"),0.,tau(),maxp1=maxp(),epsrel=relt())[0]
    txx_im = integrate.quad(tensor_integrand_2_im(xi,om,"txx"),0.,tau(),maxp1=maxp(),epsrel=relt())[0]
    txx = 1.j*txx_im+txx_r
    tyy_r = integrate.quad(tensor_integrand_2_r(xi,om,"tyy"),0.,tau(),maxp1=maxp(),epsrel=relt())[0]
    tyy_im = integrate.quad(tensor_integrand_2_im(xi,om,"tyy"),0.,tau(),maxp1=maxp(),epsrel=relt())[0]
    tyy = 1.j*tyy_im+tyy_r
    tzz_r = integrate.quad(tensor_integrand_2_r(xi,om,"tzz"),0.,tau(),maxp1=maxp(),epsrel=relt())[0]
    tzz_im = integrate.quad(tensor_integrand_2_im(xi,om,"tzz"),0.,tau(),maxp1=maxp(),epsrel=relt())[0]
    tzz = 1.j *tzz_im+tzz_r
    txz_r = integrate.quad(tensor_integrand_2_r(xi,om,"txz"),0.,tau(),maxp1=maxp(),epsrel=relt())[0]
    txz_im = integrate.quad(tensor_integrand_2_im(xi,om,"txz"),0.,tau(),maxp1=maxp(),epsrel=relt())[0]
    txz = 1.j*txz_im+txz_r
    return om**2*np.abs(( \
            tzz*np.sin(xi)**2+txx*np.cos(xi)**2-tyy-2*txz*np.sin(xi)*np.cos(xi)))**2

def test_tensor_integrand():

    funcx = tensor_integrand(0.5,1.,1.,"txx")
    funcy = tensor_integrand(0.5,1.,1.,"tyy")
    funcz = tensor_integrand(0.5,1.,1.,"tzz")
    funcxz = tensor_integrand(0.5,1.,1.,"txz")

    x = []
    y = []
    z = []
    a = []
    b = []

    for i in range(1,110):
        x.append(0.03*i)
        y.append(funcx(0.03*i))
        z.append(funcy(0.03*i))
        a.append(funcz(0.03*i))
        b.append(funcxz(0.03*i))

    pl.plot(x,y)
    pl.plot(x,z)
    pl.plot(x,a)
    pl.plot(x,b)
    pl.show()

def test_tensor_integrand_2():

    xi = np.pi/2
    om = 1.0/d()

    t = []
    txi = []
    tyi = []
    tzi = []
    txzi = []
    cutoff_pl = []
    igx = tensor_integrand_2_r(xi,om,"txx")
    igxi = tensor_integrand_2_im(xi,om,"txx")
    #igy = tensor_integrand_2_r(xi,om,"tyy")
    #igyi = tensor_integrand_2_im(xi,om,"tyy")
    #igz = tensor_integrand_2_r(xi,om,"tzz")
    #igzi = tensor_integrand_2_im(xi,om,"tzz")
    #igxz = tensor_integrand_2_r(xi,om,"txz")
    #igxzi = tensor_integrand_2_im(xi,om,"txz")

    for i in range(400):
        t.append(.2*i)
        #txi.append(np.abs(1.j*igxi(t[i])+igx(t[i])))
        txi.append(np.abs(tensor_integrand_simpson(xi,om,t[i],"txx")))
        tyi.append(np.abs(tensor_integrand_simpson(xi,om,t[i],"tyy")))
        tzi.append(np.abs(tensor_integrand_simpson(xi,om,t[i],"tzz")))
        txzi.append(np.abs(tensor_integrand_simpson(xi,om,t[i],"txz")))
        cutoff_pl.append(cutoff(t[i])*120000)
        #tyi.append(np(igy(1*i)**2+igyi(1*i)**2))
        #tzi.append((igz(1*i)**2+igyi(1*i)**2))
        #txzi.append((igxz(1*i)**2+igxzi(1*i)**2))

    pl.plot(t,txi)
    pl.plot(t,tyi)
    pl.plot(t,tzi)
    pl.plot(t,txzi)
    pl.plot(t,cutoff_pl)
    pl.axvline(x=d()/2)
    pl.axvline(x=0.9*tau())
    pl.axvline(x=tau())
    pl.show()

#test_tensor_integrand_2()
