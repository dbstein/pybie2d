import numpy as np

def squished_circle(N, x=0.0, y=0.0, r=1.0, b=1.0, rot=0.0, imag=True):
    """
    Function defining a squished circle
    Parameters:
        N:   number of points
        x:   x coordinate of center
        y:   y coordinate of center
        r:   nominal radius
        b:   how squished the circle is, 0<b<1, smaller b is more pinched
        rot: angle of rotation
    """
    t = np.linspace(0.0, 2.0*np.pi, N, endpoint=False)
    c = (x+1j*y) + r*np.exp(1j*rot)*(np.exp(1j*t)-1j*(1-b)*np.sin(t)**3)
    if imag:
        return c
    else:
        return c.real, c.imag

def star(N, x=0.0, y=0.0, r=1.0, a=0.5, f=3, rot=0.0, imag=True):
    """
    Function defining a star shaped object
    Parameters:
        N:   number of points
        x:   x coordinate of center
        y:   y coordinate of center
        r:   nominal radius
        a:   amplitude of wobble, 0<a<1, smaller a is less wobbly
        f:   frequency - how many lobes are in the star
        rot: angle of rotation
    """
    t = np.linspace(0.0, 2.0*np.pi, N, endpoint=False)
    c = (x+1j*y) + (r + r*a*np.cos(f*(t-rot)))*np.exp(1j*t)
    if imag:
        return c
    else:
        return c.real, c.imag
    