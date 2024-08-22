import matplotlib.pyplot as plt
import numpy as np

## Первый вариант системы ОДУ
class parametr:

    def __init__(self, m, L, x0, y0, v0x, v0y, tau, tBEG, tEND):

        self.g = 9.81

        self.m = m
        self.L = L
        self.x0 = x0
        self.y0 = y0
        self.u0 = np.asarray( [ x0, y0, v0x, v0y, -self.F(0) / L * y0 ] ) #u0 = [ x0, y0, T0, vx, vy ]
        self.tau = tau
        self.tBEG = tBEG
        self.tEND = tEND

    def F(self, t):

        return (self.g + 0.05 * np.sin( 2. * np.pi * t )) * self.m

        #return self.g

    def f(self, t ):

        return (0.05 * np.cos( 2. * np.pi * t ) * 2. * np.pi) * self.m

        #return 0.0

    def fun(self, t, u ):

        f_out = np.zeros( 5 )

        f_out[ 0 ] = u[ 2 ]

        f_out[ 1 ] = u[ 3 ]

        f_out[ 2 ] = - u[ 0 ] / (self.L * self.m) * u[ 4 ]

        f_out[ 3 ] =  -u[ 1 ] / (self.L * self.m) * u[ 4 ] - self.F( t )

        f_out[ 4 ] = 2. * self.m / self.L * ( u[ 2 ] * f_out[ 2 ] + u[ 3 ] * f_out[ 3 ] ) - u[ 3 ] * self.F( t ) / self.L - u[ 1 ] * self.f( t ) / self.L

        return f_out

## Второй вариант системы ОДУ
class parametr2:

    def __init__(self, m, L, x0, y0, v0x, v0y, tau, tBEG, tEND):

        self.g = 9.81

        self.m = m
        self.L = L
        self.x0 = x0
        self.y0 = y0
        self.u0 = np.asarray( [ x0, v0x, y0, v0y ] ) #u0 = [ x0, vx, y0, vy, ]
        self.tau = tau
        self.tBEG = tBEG
        self.tEND = tEND

    def F(self, t):

        return (self.g + 0.05 * np.sin( 2. * np.pi * t )) * self.m

        #return self.g

    def f(self, t ):

        return (0.05 * np.cos( 2. * np.pi * t ) * 2. * np.pi) * self.m

        #return 0.0

    def fun(self, t, u ):

        T = (self.m * (u[1] ** 2.0 + u[3] ** 2.0) - u[2] * self.F(t)) / self.L

        f_out = np.zeros( 4 )

        f_out[ 0 ] = u[ 1 ]

        f_out[ 1 ] = - u[ 0 ] / (self.L * self.m) * T

        f_out[ 2 ] = u[ 3 ]

        f_out[ 3 ] =  -u[ 2 ] / (self.L * self.m) * T - self.F( t ) / self.m

        #f_out[ 4 ] = 2. / alpha * ( u[ 2 ] * f_out[ 2 ] + u[ 3 ] * f_out[ 3 ] ) - u[ 3 ] * self.F( t ) / alpha - u[ 1 ] * self.f( t ) / alpha

        return f_out

## РЕАЛИЗАЦИЯ ЯВНОГО МЕТОДА РУНГЕ-КУТТЫ 4 ПОРЯДКА

def k( Params, t, u ):

    k = np.zeros( ( 4, u.size) )

    k[ 0 ] = Params.fun( t, u )

    for i in np.arange( 1, 3, 1 ):

        k[ i ] = Params.fun( t + Params.tau / 2., u + Params.tau * k[ i - 1 ] / 2. )

    k[ -1 ] = Params.fun( t + Params.tau, u + Params.tau * k[ -2 ] )

    return k

def runge_kutta( Params ):

    global k

    t = np.arange( Params.tBEG, Params.tEND, Params.tau )

    u = np.zeros( ( t.shape[ 0 ], Params.u0.size )  )

    u[ 0 ] = Params.u0

    for i in np.arange( 1, t.shape[ 0 ], 1 ):

        kk = k( Params, t[ i - 1 ], u[ i - 1 ] )

        C = Params.tau / 6. * ( kk[ 0 ] + 2.0 * kk[ 1 ] + 2.0 * kk[ 2 ] + kk[ 3 ] )

        u[ i ] = u[ i - 1 ] + C

    return u, t

## РЕАЛИЗАЦИЯ НЕЯВНОГО МЕТОДА АДАМСА 2 ПОРЯДКА

def jacobian( f, x ):

    h = 1.0e-4

    J = np.zeros( ( x.size, x.size ) )

    fn = f(x)

    for i in np.arange( 0, x.size, 1 ):

        x_old = np.copy( x )

        x_old[ i ] = x_old[ i ] + h

        fn_1 = f(x_old)

        J[:,i] = ( fn_1 - fn ) / h

    return J, fn

def newthon_method( f, x, eps = 1.0e-4 ):

    max_iter = 10000

    for i in np.arange( 0, max_iter, 1 ):

        J, fn = jacobian( f, x )

        if np.sqrt( np.dot( fn, fn ) / x.size ) < eps:

            return x, i

        dx = np.linalg.solve( J, fn )

        x = x - dx

def adams(Params, alpha ):

    def F(y_next):

        return y_next - Params.tau * alpha * Params.fun(t[i+1], y_next) - y[i] - Params.tau * ( 1. - alpha) * Params.fun(t[i], y[i] )

    t = np.arange( Params.tBEG, Params.tEND, Params.tau )

    y = np.zeros( ( t.shape[ 0 ], Params.u0.size )  )

    y[0] = Params.u0

    for i in np.arange(0, t.size-1, 1):

        y_next = y[ i ] + Params.tau * Params.fun( t[ i ], y[ i ] )

        y[ i + 1 ], iter = newthon_method( F, y_next )

    return y, t

def plot_graph( height: int, width: int, x: list, y: list, line_type: list, color: list, legend_list: list, ylab: str ):

    plt.figure( figsize = ( height, width ) )
    plt.rc('font', **{'size' : 32})
    for arr, obb, lin_style, col, leg_lab in zip( x, y, line_type, color, legend_list ):

        plt.plot( arr, obb, linestyle = lin_style,  color = col, label = leg_lab )

    plt.ylabel( ylab )
    plt.xlabel( 't' )
    plt.grid()
    plt.legend()
    plt.show()



