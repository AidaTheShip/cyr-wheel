import numpy as np
import sympy as sym
import casadi
from sympy.utilities.lambdify import lambdify
from sympy.parsing.mathematica import parse_mathematica

# angles

psi0 = sym.symbols('psi0')
psi1 = sym.symbols('psi1')
psi2 = sym.symbols('psi2')

theta0 = sym.symbols('theta0')
theta1 = sym.symbols('theta1')
theta2 = sym.symbols('theta2')

phi0 = sym.symbols('phi0')
phi1 = sym.symbols('phi1')
phi2 = sym.symbols('phi2')

# point mass

c0 = sym.symbols('c0')
c1 = sym.symbols('c1')
c2 = sym.symbols('c2')
mm = sym.symbols('m')

# force term

fSPar, fS3 = sym.symbols('fSPar fS3')
fAPar, fAPer= sym.symbols('fAPar fAPer')
l = sym.symbols('l')

# xy term

xx = sym.symbols('x')
yy = sym.symbols('y')

# time

tt = sym.symbols('t')

# Make circle
_a = np.linspace(0, np.pi * 2, 100)
BaseCircle = np.zeros((3, len(_a)))
BaseCircle[0] = np.cos(_a)
BaseCircle[1] = np.sin(_a)

BaseZ = np.zeros((3, 1))
BaseZ[2] = np.array([1])

# Euler angle convention: angles = [psi, theta, phi]
# Draw the circle rotated according to the Euler angles

def RotMatrices(angles):
    
    L1 = np.array([[ np.cos(angles[0]), np.sin(angles[0]), 0], 
                   [-np.sin(angles[0]), np.cos(angles[0]), 0], 
                   [0, 0, 1]])

    L2 = np.array([[1, 0, 0],
                   [0,  np.cos(angles[1]), np.sin(angles[1])], 
                   [0, -np.sin(angles[1]), np.cos(angles[1])]])

    L3 = np.array([[ np.cos(angles[2]), np.sin(angles[2]), 0], 
                   [-np.sin(angles[2]), np.cos(angles[2]), 0], 
                   [0, 0, 1]])
    
    return L1, L2, L3

def DrawCircle(ax, state):
    
    angles = Angles(state)
    XY = GC(state)
    L1, L2, L3 = RotMatrices(angles)
    L = np.matmul(L3, np.matmul(L2, L1))
    
    LL = np.matmul(np.linalg.inv(L), BaseCircle)
    LC = np.matmul(np.linalg.inv(L), BaseZ) * state[8]
    
    displacements = XY
    LL = LL + np.tile(displacements, (len(_a), 1)).T
    LC = LC.reshape(-1) + displacements
    
    ax.plot(LL[0], LL[1], LL[2], color = 'black', zorder = 1)
    ax.plot([displacements[0], LC[0]],
            [displacements[1], LC[1]],
            [displacements[2], LC[2]], color = 'blue')
    ax.scatter(LL[0][0], LL[1][0], LL[2][0], edgecolor = 'black', facecolor = 'white', lw = 1.5, zorder = 2)
    ax.scatter(LC[0], LC[1], LC[2], edgecolor = 'blue', facecolor = 'white', lw = 1.5, zorder = -1)
    
def GC(state):
    return np.array([state[6], state[7], np.sin(state[1])])

def Angles(state):
    return np.array(state[0:3])

def CT(state):
    L1, L2, L3 = RotMatrices(Angles(state))
    return GC(state) - np.matmul(np.linalg.inv(np.matmul(L2, L1)), np.array([0, 1, 0]))

def sympy2casadi(sympy_expr, sympy_var, casadi_var):
    assert casadi_var.is_vector()
    if casadi_var.shape[1]>1:
        casadi_var = casadi_var.T
    casadi_var = casadi.vertsplit(casadi_var)
    
    mapping = {'ImmutableDenseMatrix': casadi.blockcat,
               'MutableDenseMatrix': casadi.blockcat,
               'Abs':casadi.fabs
              }
    
    f = lambdify(sympy_var, sympy_expr, modules=[mapping, casadi])

    return f(*casadi_var)

def get_casadi_EOM(m):

    sympy_state = [psi0, theta0, phi0, psi1, theta1, phi1, xx, yy, c0, c1, c2]

    sympy_EOM = [psi1, theta1, phi1]
    
    # EOM without force terms
    psi2_expression = parse_mathematica('(-8 c0 c1 m psi1 + 2 theta1 (-c0 m psi1 (1 + 4 c0 Cot[theta0]) + (2 + m) phi1 Csc[theta0]))/(2 + m + 4 c0^2 m)')
    theta2_expression = parse_mathematica('-(1/(3 + 2 m + 2 c0^2 m))(2 c2 m + 2 Cos[theta0] + 2 m Cos[theta0] + 2 c0 m (2 c1 theta1 + phi1 psi1 Cos[theta0] + psi1^2 Cos[2 theta0] - Sin[theta0]) + 2 (2 + m) phi1 psi1 Sin[theta0] + 3 psi1^2 Cos[theta0] Sin[theta0] + m psi1^2 Sin[2 theta0] - c0^2 m psi1^2 Sin[2 theta0])')
    phi2_expression = parse_mathematica('(1/(2 (2 + m + 4 c0^2 m)))(4 c1 m psi1 (4 c0 Cos[theta0] + Sin[theta0]) + theta1 Csc[theta0] (4 phi1 (-(2 + m) Cos[theta0] + c0 m Sin[theta0]) + 2 psi1 (c0^2 m (7 + Cos[2 theta0]) + (3 + 2 m) Sin[theta0]^2 + 2 c0 m Sin[2 theta0])))')
    
    x1_expression = parse_mathematica('-Cos[psi0] (phi1 + psi1 Cos[theta0]) + theta1 Sin[psi0] Sin[theta0]')
    y1_expression = parse_mathematica('-(phi1 + psi1 Cos[theta0]) Sin[psi0] - theta1 Cos[psi0] Sin[theta0]')

    sympy_EOM.append(psi2_expression.subs([(mm, m)]))
    sympy_EOM.append(theta2_expression.subs([(mm, m)]))
    sympy_EOM.append(phi2_expression.subs([(mm, m)]))
    sympy_EOM.append(x1_expression.subs([(mm, m)]))
    sympy_EOM.append(y1_expression.subs([(mm, m)]))
    
    sympy_EOM.append(c1)
    sympy_EOM.append(c2)

    x = casadi.MX.sym('s', 10) 
    u = casadi.MX.sym('u')

    casadi_EOM = sympy2casadi(sympy_EOM, sympy_state, casadi.vertcat(x, u))

    return casadi_EOM, x, u

def get_stationary_phi1():

    mapping = {'ImmutableDenseMatrix': casadi.blockcat,
               'MutableDenseMatrix': casadi.blockcat,
               'Abs':casadi.fabs
              }

    _phi1 = parse_mathematica('((-2 (1 + m) Cos[theta0] + 2 c0 m Sin[theta0] + psi1^2 (-2 c0 m Cos[2 theta0] + (-3 + 2 (-1 + c0^2) m) Cos[theta0] Sin[theta0]))/(2 psi1 (c0 m Cos[theta0] + (2 + m) Sin[theta0])))')
    _phi1 = lambdify([theta0, psi1, mm, c0], _phi1, modules=[mapping, casadi])
    # _phi1 = sympy2casadi(_phi1, [theta0, psi1, mm, c0], casadi.vertcat(_theta0, _psi1, _m, _c))
    return _phi1

def get_stable_mode(_theta0, _psi1, _m, _c):

    _phi1 = parse_mathematica('((-2 (1 + m) Cos[theta0] + 2 c0 m Sin[theta0] + psi1^2 (-2 c0 m Cos[2 theta0] + (-3 + 2 (-1 + c0^2) m) Cos[theta0] Sin[theta0]))/(2 psi1 (c0 m Cos[theta0] + (2 + m) Sin[theta0])))')
    _phi1 = float(_phi1 .subs([(theta0, _theta0), (mm, _m), (psi1, _psi1), (c0, _c)]))

    return [0, _theta0, 0, _psi1, 0, _phi1, 0, 0, _c, 0]

def energy(m):

    state = [psi0, theta0, phi0, psi1, theta1, phi1, xx, yy, c0, c1]

    __ = parse_mathematica('1/8 (4 c1^2 m + 8 phi1^2 + 4 m phi1^2 + 5 psi1^2 + 2 m psi1^2 + 8 c1 m theta1 + 6 theta1^2 + 4 m theta1^2 + 16 phi1 psi1 Cos[theta0] + 8 m phi1 psi1 Cos[theta0] + 3 psi1^2 Cos[2 theta0] + 2 m psi1^2 Cos[2 theta0] + 8 Sin[theta0] + 8 m Sin[theta0] + 4 c0^2 m (theta1^2 + psi1^2 Sin[theta0]^2) - 4 c0 m (2 phi1 psi1 Sin[theta0] +    2 Cos[theta0] (-1 + psi1^2 Sin[theta0])))')
    __ = __.subs([(mm, m)])
    __ = lambdify(state, __)

    return __
