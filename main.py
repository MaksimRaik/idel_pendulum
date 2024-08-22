import matplotlib.pyplot as plt
import numpy as np
from function import *


## РЕШЕНИЕ СИСТЕМЫ ОДУ МЕТОДОМ РУНГЕ-КУТТЫ 4 ПОРЯДКА

## Первый вариант системы ОДУ
Params1 = parametr( 1., 5., 3., -4., -0.8, -0.6, 0.01, 0.0, 2. )

Params2 = parametr( 1., 5., 3., -4., -0.8, -0.6, 0.01, 0.0, 100. )

## Второй вариант системы ОДУ
Params12 = parametr2( 1., 5., 3., -4., -0.8, -0.6, 0.01, 0.0, 4 )

Params22 = parametr2( 1., 5., 3., -4., -0.8, -0.6, 0.01, 0.0, 100. )

uRk1, tRk1 = runge_kutta( Params12 )

uRk2, tRk2 = runge_kutta( Params22 )

uAd1, tAd1 = adams(Params12, 0.5)

uAd2, tAd2 = adams(Params22, 0.5)

print(np.sqrt( uAd2[ :,0 ]**2 + uAd2[ :,2 ]**2 ))



plot_graph( 25, 22, [tRk1, tAd1 ],
            [np.sqrt( uRk1[ :,0 ]**2 + uRk1[ :,2 ]**2 ), np.sqrt( uAd1[ :,0 ]**2 + uAd1[ :,2 ]**2 ) ],
            ['-','--'], ['blue', 'red'], [ 'Рунге-Кутта', 'метод Адамса' ], r'$\sqrt{x^2+y^2}$' )

plot_graph( 25, 22, [tRk2, tAd2 ],
            [np.sqrt( uRk2[ :,0 ]**2 + uRk2[ :,2 ]**2 ), np.sqrt( uAd2[ :,0 ]**2 + uAd2[ :,2 ]**2 ) ],
            ['-','--'], ['blue', 'red'], [ 'Рунге-Кутта', 'метод Адамса' ], r'$\sqrt{x^2+y^2}$' )

plot_graph( 20, 12, [tRk1, tRk1, tAd1, tAd1 ],
            [uRk1[ :,0 ], uRk1[ :,2 ], uAd1[ :, 0], uAd1[ :, 2] ],
            ['-', '-', '--', '--' ], ['blue', 'green', 'red', 'black' ], [ r'метод Рунге-Кутты Ox', r'метод Рунге-Кутты Oy', r'метод Адамса Ox', r'метод Адамса Oy' ], r'$u$' )

plot_graph( 25, 12, [tAd2, ],
           [np.sqrt( uAd2[ :,0 ]**2 + uAd2[ :,2 ]**2 ) - Params1.L, ],
           ['-',], ['blue', ], [ 'Рунге-Кутта',  ], r'$\sqrt{x^2+y^2}$' )

