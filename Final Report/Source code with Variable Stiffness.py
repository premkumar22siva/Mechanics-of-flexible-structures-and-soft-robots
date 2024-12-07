!pip install import-ipynb --quiet

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

#####################################################################################
# Miscellaneous Functions: signedAngle, rotateAxisAngle, parallel_transport, crossMat
#####################################################################################

def rotate_points_3d(points, angle_deg, axis='z'):
    """Rotate points around a specified axis ('x', 'y', or 'z') by a given angle in degrees."""
    angle_rad = np.radians(angle_deg)
    if axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0,                 0,                 1]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0,                 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'x':
        rotation_matrix = np.array([
            [1, 0,                 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad),  np.cos(angle_rad)]
        ])
    else:
        raise ValueError("Invalid rotation axis. Use 'x', 'y', or 'z'.")

    return rotation_matrix @ points

def signedAngle(u = None,v = None,n = None):
    # This function calculates the signed angle between two vectors, "u" and "v",
    # using an optional axis vector "n" to determine the direction of the angle.

    w = np.cross(u,v)
    angle = np.arctan2( np.linalg.norm(w), np.dot(u,v) )
    if (np.dot(n,w) < 0):
        angle = - angle

    return angle

def rotateAxisAngle(v = None,z = None,theta = None):
    # This function rotates a vector "v" around a specified axis "z" by an angle "theta".

    if (theta == 0):
        vNew = v
    else:
        c = np.cos(theta)
        s = np.sin(theta)
        vNew = c * v + s * np.cross(z,v) + np.dot(z,v) * (1.0 - c) * z

    return vNew

def parallel_transport(u = None,t1 = None,t2 = None):

    # This function parallel transports a vector u from tangent t1 to t2
    # Input:
    # t1 - vector denoting the first tangent
    # t2 - vector denoting the second tangent
    # u - vector that needs to be parallel transported
    # Output:
    # d - vector after parallel transport

    b = np.cross(t1,t2)
    if (np.linalg.norm(b) == 0):
        d = u
    else:
        b = b / np.linalg.norm(b)
        b = b - np.dot(b,t1) * t1
        b = b / np.linalg.norm(b)
        b = b - np.dot(b,t2) * t2
        b = b / np.linalg.norm(b)
        n1 = np.cross(t1,b)
        n2 = np.cross(t2,b)
        d = np.dot(u,t1) * t2 + np.dot(u,n1) * n2 + np.dot(u,b) * b

    return d

def crossMat(a):
    A=np.matrix([[0,- a[2],a[1]],[a[2],0,- a[0]],[- a[1],a[0],0]])
    return A

##################################################################################################
# Functions to calculate tangent, material frame, and reference frame (by time parallel transport)
##################################################################################################

def computeTangent(q):
  ne = int((len(q)+1)/4 - 1)
  tangent = np.zeros((ne, 3))
  for c in range(ne):
    dx = q[4*c+4:4*c+7] - q[4*c:4*c+3] # edge vector
    tangent[c,:] = dx / np.linalg.norm(dx)
  return tangent

def computeSpaceParallel(d1_first, q):
  ne = int((len(q)+1)/4 - 1)
  tangent = computeTangent(q)

  d1 = np.zeros((ne, 3))
  d2 = np.zeros((ne, 3))

  # First edge
  d1[0,:] = d1_first # Given
  t0 = tangent[0,:] # Tangent on first edge
  d2[0,:] = np.cross(t0, d1_first)

  # Parallel transport from previous edge to the next
  for c in range(1, ne):
    t = tangent[c,:]
    d1_first = parallel_transport(d1_first, t0, t)
    # d1_first should be perpendicular to t
    d1_first = d1_first - np.dot(d1_first, t) * t
    d1_first = d1_first / np.linalg.norm(d1_first)

    # Store d1 and d2 vectors for c-th edge
    d1[c,:] = d1_first
    d2[c,:] = np.cross(t, d1_first)

    t0 = t.copy() # New tangent now becomes old tangent
  return d1, d2

def computeMaterialFrame(a1, a2, theta):
  ne = len(theta)
  m1 = np.zeros((ne, 3))
  m2 = np.zeros((ne, 3))
  for c in range(ne): # loop over every edge
    m1[c,:] = a1[c,:] * np.cos(theta[c]) + a2[c,:] * np.sin(theta[c])
    m2[c,:] = - a1[c,:] * np.sin(theta[c]) + a2[c,:] * np.cos(theta[c])
  return m1, m2

def computeTimeParallel(a1_old, q0, q):
  # a1_old is (ne,3) ndarray representing old reference frame
  # q0 is the old DOF vector from where reference frame should be transported
  # q is the new DOF vector where reference frame should be transported to
  ne = int((len(q)+1)/4 - 1)
  tangent0 = computeTangent(q0) # Old tangents
  tangent = computeTangent(q) # New tangents

  a1 = np.zeros((ne, 3))
  a2 = np.zeros((ne, 3))
  for c in range(ne):
    t0 = tangent0[c,:]
    t = tangent[c,:]
    a1_tmp = parallel_transport(a1_old[c,:], t0, t)
    a1[c,:] = a1_tmp - np.dot(a1_tmp, t) * t
    a1[c,:] = a1[c,:] / np.linalg.norm(a1[c,:])
    a2[c,:] = np.cross(t, a1[c,:])

  return a1, a2

######################################################
# Functions to calculate reference twist and curvature
######################################################

def computeReferenceTwist(u1, u2, t1, t2, refTwist = None):
    # This function computes the reference twist angle between two vectors "u1" and "u2",
    # given two tangent directions "t1" and "t2", and an optional initial guess for the twist.
    # It adjusts the guess to align "u1" with "u2" when parallel transported along "t1" and "t2".

    if refTwist is None:
      refTwist = 0
    ut = parallel_transport(u1, t1, t2)
    ut = rotateAxisAngle(ut, t2, refTwist)
    refTwist = refTwist + signedAngle(ut, u2, t2)
    return refTwist

def computekappa(node0, node1, node2, m1e, m2e, m1f, m2f):
    # This function computes the curvature "kappa" at a "turning" node in a discrete elastic rod model.
    # The curvature is calculated using the positions of three consecutive nodes and the material
    # directors of the edges before and after the turning point.

    t0 = (node1 - node0) / np.linalg.norm(node1 - node0)
    t1 = (node2 - node1) / np.linalg.norm(node2 - node1)

    kb = 2.0 * np.cross(t0,t1) / (1.0 + np.dot(t0,t1))
    kappa1 = 0.5 * np.dot(kb,m2e + m2f)
    kappa2 = - 0.5 * np.dot(kb,m1e + m1f)

    kappa = np.zeros(2)
    kappa[0] = kappa1
    kappa[1] = kappa2

    return kappa

def getRefTwist(a1, tangent, refTwist):
  ne = a1.shape[0] # Shape of a1 is (ne,3) - returns number of rows in a1 which is ne
  for c in np.arange(1,ne):
    u0 = a1[c-1,0:3] # reference frame vector of previous edge
    u1 = a1[c,0:3] # reference frame vector of current edge
    t0 = tangent[c-1,0:3] # tangent of previous edge
    t1 = tangent[c,0:3] # tangent of current edge
    refTwist[c] = computeReferenceTwist(u0, u1, t0, t1, refTwist[c])
  return refTwist

def getKappa(q0, m1, m2):
  ne = m1.shape[0] # Shape of m1 is (ne,3)
  nv = ne + 1

  kappa = np.zeros((nv,2))

  for c in np.arange(1,ne):
    node0 = q0[4*c-4:4*c-1]
    node1 = q0[4*c+0:4*c+3]
    node2 = q0[4*c+4:4*c+7]

    m1e = m1[c-1,0:3].flatten() # Material frame of previous edge
    m2e = m2[c-1,0:3].flatten()
    m1f = m1[c,0:3].flatten() # Material frame of current edge
    m2f = m2[c,0:3].flatten()

    kappa_local = computekappa(node0, node1, node2, m1e, m2e, m1f, m2f)

    # Store the values
    kappa[c,0] = kappa_local[0]
    kappa[c,1] = kappa_local[1]

  return kappa

############################################
# Gradients and Hessians of Elastic Energies
############################################

def gradEs_hessEs(node0 = None,node1 = None,l_k = None,EA = None):

# Inputs:
# node0: 1x3 vector - position of the first node
# node1: 1x3 vector - position of the last node

# l_k: reference length (undeformed) of the edge
# EA: scalar - stretching stiffness - Young's modulus times area

# Outputs:
# dF: 6x1  vector - gradient of the stretching energy between node0 and node 1.
# dJ: 6x6 vector - hessian of the stretching energy between node0 and node 1.

    ## Gradient of Es
    edge = node1 - node0

    edgeLen = np.linalg.norm(edge)
    tangent = edge / edgeLen
    epsX = edgeLen / l_k - 1
    dF_unit = EA * tangent * epsX
    dF = np.zeros((6))
    dF[0:3] = - dF_unit
    dF[3:6] = dF_unit

    ## Hessian of Es
    Id3 = np.eye(3)
    M = EA * ((1 / l_k - 1 / edgeLen) * Id3 + 1 / edgeLen * ( np.outer( edge, edge ) ) / edgeLen ** 2)

    dJ = np.zeros((6,6))
    dJ[0:3,0:3] = M
    dJ[3:6,3:6] = M
    dJ[0:3,3:6] = - M
    dJ[3:6,0:3] = - M
    return dF,dJ

def gradEb_hessEb(node0 = None,node1 = None,node2 = None,m1e = None,m2e = None,m1f = None,m2f = None,kappaBar = None,l_k = None, EI1 = None, EI2 = None):

# This function follows the formulation by Panetta et al. 2019

# Inputs:
# node0: 1x3 vector - position of the node prior to the "turning" node
# node1: 1x3 vector - position of the "turning" node
# node2: 1x3 vector - position of the node after the "turning" node

# m1e: 1x3 vector - material director 1 of the edge prior to turning
# m2e: 1x3 vector - material director 2 of the edge prior to turning
# m1f: 1x3 vector - material director 1 of the edge after turning
# m2f: 1x3 vector - material director 2 of the edge after turning

# kappaBar: 1x2 vector - natural curvature at the turning node
# l_k: voronoi length (undeformed) of the turning node
# EI1: scalar - bending stiffness for kappa1
# EI2: scalar - bending stiffness for kappa2

# Outputs:
# dF: 11x1  vector - gradient of the bending energy at node1.
# dJ: 11x11 vector - hessian of the bending energy at node1.

    # If EI2 is not specified, set it equal to EI1
    if EI2 == None:
        EI2 = EI1

    #
    ## Computation of gradient of the two curvatures
    #
    gradKappa = np.zeros((11,2))

    ee = node1 - node0
    ef = node2 - node1
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)
    te = ee / norm_e
    tf = ef / norm_f

    # Curvature binormal
    kb = 2.0 * np.cross(te,tf) / (1.0 + np.dot(te,tf))
    chi = 1.0 + np.dot(te,tf)
    tilde_t = (te + tf) / chi
    tilde_d1 = (m1e + m1f) / chi
    tilde_d2 = (m2e + m2f) / chi

    # Curvatures
    kappa1 = 0.5 * np.dot(kb,m2e + m2f)
    kappa2 = - 0.5 * np.dot(kb,m1e + m1f)

    Dkappa1De = 1.0 / norm_e * (- kappa1 * tilde_t + np.cross(tf,tilde_d2))
    Dkappa1Df = 1.0 / norm_f * (- kappa1 * tilde_t - np.cross(te,tilde_d2))
    Dkappa2De = 1.0 / norm_e * (- kappa2 * tilde_t - np.cross(tf,tilde_d1))
    Dkappa2Df = 1.0 / norm_f * (- kappa2 * tilde_t + np.cross(te,tilde_d1))
    gradKappa[0:3,1-1] = - Dkappa1De
    gradKappa[4:7,1-1] = Dkappa1De - Dkappa1Df
    gradKappa[8:11,1-1] = Dkappa1Df
    gradKappa[0:3,2-1] = - Dkappa2De
    gradKappa[4:7,2-1] = Dkappa2De - Dkappa2Df
    gradKappa[8:11,2-1] = Dkappa2Df
    gradKappa[4-1,1-1] = - 0.5 * np.dot(kb,m1e)
    gradKappa[8-1,1-1] = - 0.5 * np.dot(kb,m1f)
    gradKappa[4-1,2-1] = - 0.5 * np.dot(kb,m2e)
    gradKappa[8-1,2-1] = - 0.5 * np.dot(kb,m2f)

    #
    ## Computation of hessian of the two curvatures
    #
    DDkappa1 = np.zeros((11,11)) # Hessian of kappa1
    DDkappa2 = np.zeros((11,11)) # Hessian of kappa2

    norm2_e = norm_e ** 2
    norm2_f = norm_f ** 2

    tt_o_tt = np.outer(tilde_t, tilde_t)
    tmp = np.cross(tf,tilde_d2)
    tf_c_d2t_o_tt = np.outer(tmp, tilde_t)
    tt_o_tf_c_d2t = np.transpose(tf_c_d2t_o_tt)
    kb_o_d2e = np.outer(kb, m2e)
    d2e_o_kb = np.transpose(kb_o_d2e) # Not used in Panetta 2019
    te_o_te = np.outer(te, te)
    Id3 = np.eye(3)

    D2kappa1De2 = 1.0 / norm2_e * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tt_o_tf_c_d2t) - kappa1 / (chi * norm2_e) * (Id3 - te_o_te ) + 1.0 / (2.0 * norm2_e) * (kb_o_d2e)


    tmp = np.cross(te,tilde_d2)
    te_c_d2t_o_tt = np.outer(tmp, tilde_t)
    tt_o_te_c_d2t = np.transpose(te_c_d2t_o_tt)
    kb_o_d2f = np.outer(kb, m2f)
    d2f_o_kb = np.transpose(kb_o_d2f) # Not used in Panetta 2019
    tf_o_tf = np.outer( tf, tf )

    D2kappa1Df2 = 1.0 / norm2_f * (2 * kappa1 * tt_o_tt + te_c_d2t_o_tt + tt_o_te_c_d2t) - kappa1 / (chi * norm2_f) * (Id3 - tf_o_tf) + 1.0 / (2.0 * norm2_f) * (kb_o_d2f)


    te_o_tf = np.outer(te, tf)
    D2kappa1DfDe = - kappa1 / (chi * norm_e * norm_f) * (Id3 + te_o_tf) + 1.0 / (norm_e * norm_f) * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt + tt_o_te_c_d2t - crossMat(tilde_d2))
    D2kappa1DeDf = np.transpose(D2kappa1DfDe)

    tmp = np.cross(tf,tilde_d1)
    tf_c_d1t_o_tt = np.outer(tmp, tilde_t)
    tt_o_tf_c_d1t = np.transpose(tf_c_d1t_o_tt)
    kb_o_d1e = np.outer(kb, m1e)
    d1e_o_kb = np.transpose(kb_o_d1e) # Not used in Panetta 2019

    D2kappa2De2 = 1.0 / norm2_e * (2.0 * kappa2 * tt_o_tt + tf_c_d1t_o_tt + tt_o_tf_c_d1t) - kappa2 / (chi * norm2_e) * (Id3 - te_o_te) - 1.0 / (2.0 * norm2_e) * (kb_o_d1e)

    tmp = np.cross(te,tilde_d1)
    te_c_d1t_o_tt = np.outer(tmp, tilde_t)
    tt_o_te_c_d1t = np.transpose(te_c_d1t_o_tt)
    kb_o_d1f = np.outer(kb, m1f)
    d1f_o_kb = np.transpose(kb_o_d1f) # Not used in Panetta 2019

    D2kappa2Df2 = 1.0 / norm2_f * (2 * kappa2 * tt_o_tt - te_c_d1t_o_tt - tt_o_te_c_d1t) - kappa2 / (chi * norm2_f) * (Id3 - tf_o_tf) - 1.0 / (2.0 * norm2_f) * (kb_o_d1f)

    D2kappa2DfDe = - kappa2 / (chi * norm_e * norm_f) * (Id3 + te_o_tf) + 1.0 / (norm_e * norm_f) * (2 * kappa2 * tt_o_tt + tf_c_d1t_o_tt - tt_o_te_c_d1t + crossMat(tilde_d1))
    D2kappa2DeDf = np.transpose(D2kappa2DfDe)

    D2kappa1Dthetae2 = - 0.5 * np.dot(kb,m2e)
    D2kappa1Dthetaf2 = - 0.5 * np.dot(kb,m2f)
    D2kappa2Dthetae2 = 0.5 * np.dot(kb,m1e)
    D2kappa2Dthetaf2 = 0.5 * np.dot(kb,m1f)

    D2kappa1DeDthetae = 1.0 / norm_e * (0.5 * np.dot(kb,m1e) * tilde_t - 1.0 / chi * np.cross(tf,m1e))
    D2kappa1DeDthetaf = 1.0 / norm_e * (0.5 * np.dot(kb,m1f) * tilde_t - 1.0 / chi * np.cross(tf,m1f))
    D2kappa1DfDthetae = 1.0 / norm_f * (0.5 * np.dot(kb,m1e) * tilde_t + 1.0 / chi * np.cross(te,m1e))
    D2kappa1DfDthetaf = 1.0 / norm_f * (0.5 * np.dot(kb,m1f) * tilde_t + 1.0 / chi * np.cross(te,m1f))
    D2kappa2DeDthetae = 1.0 / norm_e * (0.5 * np.dot(kb,m2e) * tilde_t - 1.0 / chi * np.cross(tf,m2e))
    D2kappa2DeDthetaf = 1.0 / norm_e * (0.5 * np.dot(kb,m2f) * tilde_t - 1.0 / chi * np.cross(tf,m2f))
    D2kappa2DfDthetae = 1.0 / norm_f * (0.5 * np.dot(kb,m2e) * tilde_t + 1.0 / chi * np.cross(te,m2e))
    D2kappa2DfDthetaf = 1.0 / norm_f * (0.5 * np.dot(kb,m2f) * tilde_t + 1.0 / chi * np.cross(te,m2f))

    # Curvature terms
    DDkappa1[0:3,0:3] = D2kappa1De2
    DDkappa1[0:3,4:7] = - D2kappa1De2 + D2kappa1DfDe
    DDkappa1[0:3,8:11] = - D2kappa1DfDe
    DDkappa1[4:7,0:3] = - D2kappa1De2 + D2kappa1DeDf
    DDkappa1[4:7,4:7] = D2kappa1De2 - D2kappa1DeDf - D2kappa1DfDe + D2kappa1Df2
    DDkappa1[4:7,8:11] = D2kappa1DfDe - D2kappa1Df2
    DDkappa1[8:11,0:3] = - D2kappa1DeDf
    DDkappa1[8:11,4:7] = D2kappa1DeDf - D2kappa1Df2
    DDkappa1[8:11,8:11] = D2kappa1Df2

    # Twist terms
    DDkappa1[4-1,4-1] = D2kappa1Dthetae2
    DDkappa1[8-1,8-1] = D2kappa1Dthetaf2

    # Curvature-twist coupled terms
    DDkappa1[0:3,4-1] = - D2kappa1DeDthetae
    DDkappa1[4:7,4-1] = D2kappa1DeDthetae - D2kappa1DfDthetae
    DDkappa1[8:11,4-1] = D2kappa1DfDthetae
    DDkappa1[4-1,0:3] = np.transpose(DDkappa1[0:3,4-1])
    DDkappa1[4-1,4:7] = np.transpose(DDkappa1[4:7,4-1])
    DDkappa1[4-1,8:11] = np.transpose(DDkappa1[8:11,4-1])

    # Curvature-twist coupled terms
    DDkappa1[0:3,8-1] = - D2kappa1DeDthetaf
    DDkappa1[4:7,8-1] = D2kappa1DeDthetaf - D2kappa1DfDthetaf
    DDkappa1[8:11,8-1] = D2kappa1DfDthetaf
    DDkappa1[8-1,0:3] = np.transpose(DDkappa1[0:3,8-1])
    DDkappa1[8-1,4:7] = np.transpose(DDkappa1[4:7,8-1])
    DDkappa1[8-1,8:11] = np.transpose(DDkappa1[8:11,8-1])

    # Curvature terms
    DDkappa2[0:3,0:3] = D2kappa2De2
    DDkappa2[0:3,4:7] = - D2kappa2De2 + D2kappa2DfDe
    DDkappa2[0:3,8:11] = - D2kappa2DfDe
    DDkappa2[4:7,0:3] = - D2kappa2De2 + D2kappa2DeDf
    DDkappa2[4:7,4:7] = D2kappa2De2 - D2kappa2DeDf - D2kappa2DfDe + D2kappa2Df2
    DDkappa2[4:7,8:11] = D2kappa2DfDe - D2kappa2Df2
    DDkappa2[8:11,0:3] = - D2kappa2DeDf
    DDkappa2[8:11,4:7] = D2kappa2DeDf - D2kappa2Df2
    DDkappa2[8:11,8:11] = D2kappa2Df2

    # Twist terms
    DDkappa2[4-1,4-1] = D2kappa2Dthetae2
    DDkappa2[8-1,8-1] = D2kappa2Dthetaf2

    # Curvature-twist coupled terms
    DDkappa2[0:3,4-1] = - D2kappa2DeDthetae
    DDkappa2[4:7,4-1] = D2kappa2DeDthetae - D2kappa2DfDthetae
    DDkappa2[8:11,4-1] = D2kappa2DfDthetae
    DDkappa2[4-1,0:3] = np.transpose(DDkappa2[0:3,4-1])
    DDkappa2[4-1,4:7] = np.transpose(DDkappa2[4:7,4-1])
    DDkappa2[4-1,8:11] = np.transpose(DDkappa2[8:11,4-1])

    # Curvature-twist coupled terms
    DDkappa2[0:3,8-1] = - D2kappa2DeDthetaf
    DDkappa2[4:7,8-1] = D2kappa2DeDthetaf - D2kappa2DfDthetaf
    DDkappa2[8:11,8-1] = D2kappa2DfDthetaf
    DDkappa2[8-1,0:3] = np.transpose(DDkappa2[0:3,8-1])
    DDkappa2[8-1,4:7] = np.transpose(DDkappa2[4:7,8-1])
    DDkappa2[8-1,8:11] = np.transpose(DDkappa2[8:11,8-1])

    #
    ## Gradient of Eb
    #
    EIMat = np.array([[EI1, 0], [0, EI2]])
    kappaVector = np.array([kappa1, kappa2])
    dkappaVector = kappaVector - kappaBar
    gradKappa_1 = gradKappa[:,0]
    gradKappa_2 = gradKappa[:,1]
    dE_dKappa1 = EI1 / l_k * dkappaVector[0] # Gradient of Eb wrt kappa1
    dE_dKappa2 = EI2 / l_k * dkappaVector[1] # Gradient of Eb wrt kappa2
    d2E_dKappa11 = EI1 / l_k # Second gradient of Eb wrt kappa1
    d2E_dKappa22 = EI2 / l_k # Second gradient of Eb wrt kappa2

    # dF is the gradient of Eb wrt DOFs
    dF = dE_dKappa1 * gradKappa_1 + dE_dKappa2 * gradKappa_2

    # Hessian of Eb
    gradKappa1_o_gradKappa1 = np.outer(gradKappa_1, gradKappa_1)
    gradKappa2_o_gradKappa2 = np.outer(gradKappa_2, gradKappa_2)
    dJ = dE_dKappa1 * DDkappa1 + dE_dKappa2 * DDkappa2 + d2E_dKappa11 * gradKappa1_o_gradKappa1 + d2E_dKappa22 * gradKappa2_o_gradKappa2

    return dF,dJ

def gradEt_hessEt(node0 = None,node1 = None,node2 = None,theta_e = None,
    theta_f = None,refTwist = None,twistBar = None,l_k = None,GJ = None):

# Formulation due to Panetta 2019

# Inputs:
# node0: 1x3 vector - position of the node prior to the "twisting" node
# node1: 1x3 vector - position of the "twisting" node
# node2: 1x3 vector - position of the node after the "twisting" node

# theta_e: scalar - twist angle of the first edge
# theta_f: scalar - twist angle of the second (last) edge

# l_k: voronoi length (undeformed) of the turning node
# refTwist: reference twist (unit: radian) at the node
# twistBar: undeformed twist (unit: radian) at the node
# GJ: scalar - twisting stiffness

# Outputs:
# dF: 11x1  vector - gradient of the twisting energy at node1.
# dJ: 11x11 vector - hessian of the twisting energy at node1.

    gradTwist = np.zeros(11)
    ee = node1 - node0
    ef = node2 - node1
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)
    norm2_e = norm_e ** 2
    norm2_f = norm_f ** 2
    te = ee / norm_e
    tf = ef / norm_f

    # Curvature binormal
    kb = 2.0 * np.cross(te,tf) / (1.0 + np.dot(te,tf))

    # Gradient of twist wrt DOFs
    gradTwist[0:3] = - 0.5 / norm_e * kb
    gradTwist[8:11] = 0.5 / norm_f * kb
    gradTwist[4:7] = - (gradTwist[0:3] + gradTwist[8:11])
    gradTwist[4-1] = - 1
    gradTwist[8-1] = 1

    chi = 1.0 + np.dot(te,tf)
    tilde_t = (te + tf) / chi
    te_plus_tilde_t = te + tilde_t;
    kb_o_te = np.outer(kb, te_plus_tilde_t)
    te_o_kb = np.outer(te_plus_tilde_t, kb)
    tf_plus_tilde_t = tf + tilde_t
    kb_o_tf = np.outer(kb, tf_plus_tilde_t)
    tf_o_kb = np.outer(tf_plus_tilde_t, kb)
    kb_o_tilde_t = np.outer(kb, tilde_t)

    ## Hessian of twist wrt DOFs
    DDtwist = np.zeros((11,11))
    # Panetta 2019 formulation
    D2mDe2 = -0.5 / norm2_e * (np.outer(kb, (te + tilde_t)) + 2.0 / chi * crossMat(tf))
    D2mDf2 = -0.5 / norm2_f * (np.outer(kb, (tf + tilde_t)) - 2.0 / chi * crossMat(te))
    D2mDfDe = 0.5 / (norm_e * norm_f) * (2.0 / chi * crossMat(te) - np.outer(kb, tilde_t)) # CAREFUL: D2mDfDe means \partial^2 m/\partial e^i \partial e^{i-1}
    D2mDeDf = 0.5 / (norm_e * norm_f) * (-2.0 / chi * crossMat(tf) - np.outer(kb, tilde_t))

    # See Line 1145 of https://github.com/jpanetta/ElasticRods/blob/master/ElasticRod.cc
    DDtwist[0:3,0:3] = D2mDe2
    DDtwist[0:3,4:7] = - D2mDe2 + D2mDfDe
    DDtwist[4:7,0:3] = - D2mDe2 + D2mDeDf
    DDtwist[4:7,4:7] = D2mDe2 - (D2mDeDf + D2mDfDe) + D2mDf2
    DDtwist[0:3,8:11] = - D2mDfDe
    DDtwist[8:11,0:3] = - D2mDeDf
    DDtwist[8:11,4:7] = D2mDeDf - D2mDf2
    DDtwist[4:7,8:11] = D2mDfDe - D2mDf2
    DDtwist[8:11,8:11] = D2mDf2

    ## Gradients and Hessians of energy with respect to twist
    integratedTwist = theta_f - theta_e + refTwist - twistBar
    dE_dTau = GJ / l_k * integratedTwist
    d2E_dTau2 = GJ / l_k

    ## Gradient of Et
    dF = dE_dTau * gradTwist
    ## Hessian of Eb
    gradTwist_o_gradTwist = np.outer( gradTwist, gradTwist )
    dJ = dE_dTau * DDtwist + d2E_dTau2 * gradTwist_o_gradTwist
    return dF,dJ

##########################################################
# Functions to evaluate elastic forces along the arclength
##########################################################

def getFs(q, EA, refLen):
  ndof = len(q)
  nv = int((ndof + 1) / 4 ) # Number of vertices
  ne = nv - 1 # Number of edges

  Fs = np.zeros(ndof)
  Js = np.zeros((ndof,ndof))

  for c in range(ne):
    node0 = q[4 * c : 4 * c + 3]
    node1 = q[4 * c + 4 : 4 * c + 7]
    ind = np.array([4 * c, 4 * c + 1, 4 * c + 2, 4 * c + 4, 4 * c + 5, 4 * c + 6])

    l_k = refLen[c]

    dF, dJ = gradEs_hessEs(node0, node1, l_k, EA[c]) #index this

    Fs[ind] -= dF
    Js[np.ix_(ind, ind)] -= dJ

  return Fs, Js

def getFb(q, m1, m2, nat_curvature, EI, voronoiRefLen):
  ndof = len(q)
  nv = int((ndof + 1) / 4 ) # Number of vertices
  ne = nv - 1 # Number of edges

  Fb = np.zeros(ndof)
  Jb = np.zeros((ndof,ndof))

  for c in range(1,ne): # Loop over all the internal nodes
    node0 = q[4 * c -4 : 4 * c - 1] # (c-1) th node
    node1 = q[4 * c : 4 * c + 3] # c-th node
    node2 = q[4 * c + 4: 4 * c + 7] # (c+1)-th node
    l_k = voronoiRefLen[c]

    m1e = m1[c-1, 0:3]
    m2e = m2[c-1, 0:3]
    m1f = m1[c, 0:3]
    m2f = m2[c, 0:3]

    ind = np.arange(4*c - 4, 4 * c + 7) # 11 elements (3 nodes, 2 edges/theta angles)

    dF, dJ = gradEb_hessEb(node0, node1, node2, m1e, m2e, m1f, m2f, nat_curvature, l_k, EI[c]) # index here

    Fb[ind] -= dF
    Jb[np.ix_(ind, ind)] -= dJ

  return Fb, Jb

def getFt(q, refTwist, twistBar, GJ, voronoiRefLen):
  ndof = len(q)
  nv = int((ndof + 1) / 4 ) # Number of vertices
  ne = nv - 1 # Number of edges

  Ft = np.zeros(ndof)
  Jt = np.zeros((ndof,ndof))

  for c in range(1,ne): # Loop over all the internal nodes
    node0 = q[4 * c -4 : 4 * c - 1] # (c-1) th node
    node1 = q[4 * c : 4 * c + 3] # c-th node
    node2 = q[4 * c + 4: 4 * c + 7] # (c+1)-th node

    theta_e = q[4 * c - 1]
    theta_f = q[4 * c + 3]

    l_k = voronoiRefLen[c]
    refTwist_c = refTwist[c]
    twistBar_c = twistBar[c]

    ind = np.arange(4*c - 4, 4 * c + 7) # 11 elements (3 nodes, 2 edges/theta angles)

    dF, dJ = gradEt_hessEt(node0, node1, node2, theta_e, theta_f, refTwist_c, twistBar_c, l_k, GJ[c]) # index here

    Ft[ind] -= dF
    Jt[np.ix_(ind, ind)] -= dJ

  return Ft, Jt

##############
# Plot the rod
##############

# Function to set equal aspect ratio for 3D plots
def set_axes_equal(ax):
    """
    Set equal aspect ratio for a 3D plot in Matplotlib.
    This function adjusts the limits of the plot to make sure
    that the scale is equal along all three axes.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

def plotrod_simple(q, ctime):
    """
    Function to plot the rod with the position and directors.

    Parameters:
    - q: Position vector (DOF vector).
    - ctime: Current time for title.
    """

    x1 = q[0::4]  # x-coordinates of the nodes
    x2 = q[1::4]  # y-coordinates of the nodes
    x3 = q[2::4]  # z-coordinates of the nodes

    fig = plt.figure(1)
    clear_output()
    plt.clf()  # Clear the figure
    ax = fig.add_subplot(111, projection='3d') # Creates xyz axes

    # Plot the rod as black circles connected by lines
    ax.plot3D(x1, x2, x3, 'ko-')

    # Plot the first node with a red triangle
    ax.plot3D([x1[0]], [x2[0]], [x3[0]], 'r^')

    # Set the title with current time
    ax.set_title(f'Position of the actuator at t={ctime:.2f}')

    # Set axes labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Set equal scaling and a 3D view
    set_axes_equal(ax)
    plt.draw()  # Force a redraw of the figure

    plt.show()

def animate_three_rods(
    rod_positions1, rod_positions2, rod_positions3,
    filename="three_rods_simulation.mp4", fps=30):
    """
    Create an animation of three rods' positions over time with uniform axis ranges and tick increments.

    Parameters:
    - rod_positions1, rod_positions2, rod_positions3: Lists of 3xN numpy arrays.
      Each list represents the position of a rod over time, with each array being the rod's position at a timestep.
    - filename: Output filename for the video (e.g., 'three_rods_simulation.mp4').
    - fps: Frames per second for the video.
    """
    # Combine all rod positions to determine global bounds for the plot
    all_positions = np.hstack([
        np.hstack(rod_positions1),
        np.hstack(rod_positions2),
        np.hstack(rod_positions3),
    ])
    x_min, x_max = np.min(all_positions[0]), np.max(all_positions[0])
    y_min, y_max = np.min(all_positions[1]), np.max(all_positions[1])
    z_min, z_max = np.min(all_positions[2]), np.max(all_positions[2])

    # Set a uniform axis range
    global_min = min(x_min, y_min, z_min)
    global_max = max(x_max, y_max, z_max)

    # Define tick intervals for consistent increments
    tick_interval = (global_max - global_min) / 5  # 5 ticks as an example
    ticks = np.arange(global_min, global_max + tick_interval, tick_interval)

    # Set up the figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(global_min, global_max)
    ax.set_ylim(global_min, global_max)
    ax.set_zlim(global_min, global_max)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Initialize the plot for the three rods
    rod1_line, = ax.plot([], [], [], lw=2, color="red", label="Finger 1")
    rod2_line, = ax.plot([], [], [], lw=2, color="blue", label="Finger 2")
    rod3_line, = ax.plot([], [], [], lw=2, color="green", label="Finger 3")
    ax.legend()

    def init():
        """Initialize all rods as empty."""
        rod1_line.set_data([], [])
        rod1_line.set_3d_properties([])
        rod2_line.set_data([], [])
        rod2_line.set_3d_properties([])
        rod3_line.set_data([], [])
        rod3_line.set_3d_properties([])
        return rod1_line, rod2_line, rod3_line

    def update(frame):
        """Update function for each frame."""
        # Update Rod 1
        pos1 = rod_positions1[frame]
        rod1_line.set_data(pos1[0], pos1[1])
        rod1_line.set_3d_properties(pos1[2])

        # Update Rod 2
        pos2 = rod_positions2[frame]
        rod2_line.set_data(pos2[0], pos2[1])
        rod2_line.set_3d_properties(pos2[2])

        # Update Rod 3
        pos3 = rod_positions3[frame]
        rod3_line.set_data(pos3[0], pos3[1])
        rod3_line.set_3d_properties(pos3[2])

        return rod1_line, rod2_line, rod3_line

    # Create the animation
    ani = FuncAnimation(
        fig,
        update,
        frames=len(rod_positions1),  # Assumes all three rods have the same number of timesteps
        init_func=init,
        interval=1000 / fps,
        blit=True,
    )

    # Save the animation as a video
    ani.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.close(fig)
    print(f"Animation saved as {filename}")

def animate_three_rods_underwater(
    rod_positions1, rod_positions2, rod_positions3,
    filename="three_rods_underwater.mp4", fps=30
):
    """
    Create an animation of three rods' positions over time with an underwater effect.

    Parameters:
    - rod_positions1, rod_positions2, rod_positions3: Lists of 3xN numpy arrays.
      Each list represents the position of a rod over time, with each array being the rod's position at a timestep.
    - filename: Output filename for the video (e.g., 'three_rods_underwater.mp4').
    - fps: Frames per second for the video.
    """
    # Combine all rod positions to determine global bounds for the plot
    all_positions = np.hstack([
        np.hstack(rod_positions1),
        np.hstack(rod_positions2),
        np.hstack(rod_positions3),
    ])
    x_min, x_max = np.min(all_positions[0]), np.max(all_positions[0])
    y_min, y_max = np.min(all_positions[1]), np.max(all_positions[1])
    z_min, z_max = np.min(all_positions[2]), np.max(all_positions[2])

    # Set a uniform axis range
    global_min = min(x_min, y_min, z_min)
    global_max = max(x_max, y_max, z_max)

    # Set up the figure and 3D axes with a blue background
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(global_min, global_max)
    ax.set_ylim(global_min, global_max)
    ax.set_zlim(global_min, global_max)
    ax.set_facecolor("lightblue")  # Simulate water background
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Initialize the plot for the three rods with underwater-inspired colors
    rod1_line, = ax.plot([], [], [], lw=2, color="cyan", alpha=0.8, label="Finger 1")
    rod2_line, = ax.plot([], [], [], lw=2, color="lime", alpha=0.8, label="Finger 2")
    rod3_line, = ax.plot([], [], [], lw=2, color="magenta", alpha=0.8, label="Finger 3")
    ax.legend()

    # Add wave-like distortion (optional)
    def add_wave_effect(positions, t):
        """Apply a wave effect to simulate water motion."""
        wave = 0.01 * np.sin(2 * np.pi * positions[0] + t)
        positions[2] += wave  # Add the wave effect to the z-axis
        return positions

    # Add bubbles (optional)
    def add_bubbles(ax, num_bubbles=50):
        """Add bubbles at random positions that rise over time."""
        bubble_x = np.random.uniform(global_min, global_max, num_bubbles)
        bubble_y = np.random.uniform(global_min, global_max, num_bubbles)
        bubble_z = np.random.uniform(global_min, global_max, num_bubbles)
        bubbles = ax.scatter(bubble_x, bubble_y, bubble_z, color="white", alpha=0.6, s=10)
        return bubbles

    bubbles = add_bubbles(ax)

    def update_bubbles(bubbles):
        """Update bubble positions to make them rise."""
        offsets = bubbles._offsets3d
        z = offsets[2] + 0.01  # Make bubbles rise
        z[z > global_max] = global_min  # Reset bubbles that go too high
        bubbles._offsets3d = (offsets[0], offsets[1], z)

    def init():
        """Initialize all rods as empty."""
        rod1_line.set_data([], [])
        rod1_line.set_3d_properties([])
        rod2_line.set_data([], [])
        rod2_line.set_3d_properties([])
        rod3_line.set_data([], [])
        rod3_line.set_3d_properties([])
        return rod1_line, rod2_line, rod3_line, bubbles

    def update(frame):
        """Update function for each frame."""
        # Update Rod 1
        pos1 = add_wave_effect(rod_positions1[frame].copy(), frame / fps)
        rod1_line.set_data(pos1[0], pos1[1])
        rod1_line.set_3d_properties(pos1[2])

        # Update Rod 2
        pos2 = add_wave_effect(rod_positions2[frame].copy(), frame / fps)
        rod2_line.set_data(pos2[0], pos2[1])
        rod2_line.set_3d_properties(pos2[2])

        # Update Rod 3
        pos3 = add_wave_effect(rod_positions3[frame].copy(), frame / fps)
        rod3_line.set_data(pos3[0], pos3[1])
        rod3_line.set_3d_properties(pos3[2])

        # Update bubbles
        update_bubbles(bubbles)

        return rod1_line, rod2_line, rod3_line, bubbles

    # Create the animation
    ani = FuncAnimation(
        fig,
        update,
        frames=len(rod_positions1),  # Assumes all three rods have the same number of timesteps
        init_func=init,
        interval=1000 / fps,
        blit=False,  # Cannot use blit=True with 3D plots
    )

    # Save the animation as a video
    ani.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.close(fig)
    print(f"Animation saved as {filename}")

from matplotlib.patches import Patch

def plot_three_rods_timeline(rod_positions1, rod_positions2, rod_positions3, filename="gripper_timeline.png"):
    """
    Create a static time-lapse plot showing the motion of three rods over time.

    Parameters:
    - rod_positions1, rod_positions2, rod_positions3: Lists of 3xN numpy arrays.
      Each list represents the position of a rod over time, with each array being the rod's position at a timestep.
    - filename: Output filename for the static image (e.g., 'gripper_timeline.png').
    """
    timesteps = len(rod_positions1)  # Assumes all three rods have the same number of timesteps
    cmap = plt.cm.viridis  # Color map for timesteps

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection="3d")

    # Combine all positions to determine global bounds
    all_positions = np.hstack([
        np.hstack(rod_positions1),
        np.hstack(rod_positions2),
        np.hstack(rod_positions3),
    ])
    x_min, x_max = np.min(all_positions[0]), np.max(all_positions[0])
    y_min, y_max = np.min(all_positions[1]), np.max(all_positions[1])
    z_min, z_max = np.min(all_positions[2]), np.max(all_positions[2])
    global_min = min(x_min, y_min, z_min)
    global_max = max(x_max, y_max, z_max)

    ax.set_xlim(global_min, global_max)
    ax.set_ylim(global_min, global_max)
    ax.set_zlim(global_min, global_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Plot the rods over time with varying colors and transparency
    for t, (pos1, pos2, pos3) in enumerate(zip(rod_positions1, rod_positions2, rod_positions3)):
        color = cmap(t / timesteps)  # Color for this timestep
        alpha = 0.2 + 0.8 * (t / timesteps)  # Gradually increase opacity over time

        ax.plot(pos1[0], pos1[1], pos1[2], color=color, alpha=alpha, lw=2)
        ax.plot(pos2[0], pos2[1], pos2[2], color=color, alpha=alpha, lw=2)
        ax.plot(pos3[0], pos3[1], pos3[2], color=color, alpha=alpha, lw=2)

    # Add a legend with accurate colors for start and end of motion
    legend_elements = [
        Patch(facecolor=cmap(0), edgecolor="k", label="Start of motion"),
        Patch(facecolor='#FDE725', edgecolor="k", label="End of motion"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", title="Legend")

    plt.title("Motion of Gripper Fingers Over Time")

    # Save and show the plot
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Time-lapse plot saved as {filename}")

import numpy as np
import plotly.graph_objects as go

def plot_gripper_plotly(rod_positions1, rod_positions2, rod_positions3):
    """
    Visualize the gripper's motion over time in a cooler way using Plotly.

    Parameters:
    - rod_positions1, rod_positions2, rod_positions3: Lists of 3xN numpy arrays for the three rods over time.
    """
    timesteps = len(rod_positions1)
    fig = go.Figure()

    for t in range(timesteps):
        color = f"rgba({255 - int(255 * t / timesteps)}, {int(255 * t / timesteps)}, 255, 0.7)"

        # Add traces for each rod
        for rod, name in zip(
            [rod_positions1[t], rod_positions2[t], rod_positions3[t]],
            ["Finger 1", "Finger 2", "Finger 3"]
        ):
            fig.add_trace(
                go.Scatter3d(
                    x=rod[0], y=rod[1], z=rod[2],
                    mode="lines+markers",
                    line=dict(color=color, width=4),
                    name=name,
                    marker=dict(size=3)
                )
            )

    # Update layout for cool effects
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", backgroundcolor="lightblue"),
            yaxis=dict(title="Y", backgroundcolor="lightblue"),
            zaxis=dict(title="Z", backgroundcolor="lightblue"),
        ),
        title="Gripper Motion in 3D (Interactive)"

    )

    fig.show()

####################
# Objective Function
####################

def objfun(qGuess, q0, u, a1, a2,
           freeIndex, # Boundary conditions
           dt, tol, # time stepping parameters
           refTwist, # guess refTwist to compute the new refTwist
           massVector, mMat, # Mass vector and mass matrix
           EA, refLen, # reference length
           EI, GJ, voronoiRefLen, twistBar, # stiffness
           Fg, C, # external forces
           ctime): # current time

  q = qGuess # Guess
  iter = 0
  error = 10 * tol
  nat_curvature = np.zeros(2)
  nat_curvature = [0, -0.15*ctime] # actuation

  while error > tol:
    a1Iterate, a2Iterate = computeTimeParallel(a1, q0, q) # Reference frame
    tangent = computeTangent(q)
    refTwist_iterate = getRefTwist(a1Iterate, tangent, refTwist) # Reference twist

    # Material frame
    theta = q[3::4] # twist angles
    m1Iterate, m2Iterate = computeMaterialFrame(a1Iterate, a2Iterate, theta)

    # Compute my elastic forces
    # Bending
    Fb, Jb = getFb(q, m1Iterate, m2Iterate, nat_curvature, EI, voronoiRefLen)
    # Twisting
    Ft, Jt = getFt(q, refTwist_iterate, twistBar, GJ, voronoiRefLen)
    # Stretching
    Fs, Js = getFs(q, EA, refLen)

    # Viscous force
    Fv = -C @ (q-q0) / dt
    Jv = -C / dt

    # Set up EOMs
    Forces = Fb + Ft + Fs + Fg + Fv
    Jforces = Jb + Jt + Js + Jv
    f = massVector/dt * ( (q-q0)/dt - u ) - Forces
    J = mMat / dt**2 - Jforces
    # Free components of f and J to impose BCs
    f_free = f[freeIndex]
    J_free = J[np.ix_(freeIndex, freeIndex)]

    # Update
    dq_free = np.linalg.solve(J_free, f_free)

    q[freeIndex] = q[freeIndex] - dq_free # Update free DOFs
    error = np.sum(np.abs(f_free))

    iter += 1

  u = (q - q0) / dt # velocity vector

  return q, u, a1Iterate, a2Iterate

######
# MAIN
######

nv = 20 # nodes (chosen based on sensitivity analysis)
ne = nv - 1 # edges
ndof = 4 * nv - 1 # degrees of freedom: 3*nv + ne

RodLength = 0.2 # meter
natR = 0 # natural radius of curvature

r0 = np.zeros((nv)) # cross-sectional radius (a circular cross-section is assumed for the simulation)

for c in range(nv):
  r0[c] = 0.001*(nv-c) # linear decrease in the radius of the gripper

# Matrix (numpy ndarray) for the nodes at t=0
nodes = np.zeros((nv, 3))
if natR == 0: # straight rod
  for c in range(nv):
    nodes[c, 0] = c * RodLength / (nv - 1) # x coordinate of c-th node
    nodes[c, 1] = 0                        # y coordinate of c-th node
    nodes[c, 2] = 0                        # z coordinate of c-th node
else: # rod with circular shape (ring)
  dTheta = (RodLength / natR) * (1.0 / ne) # If the angle formed by two consecutive nodes at the centre of the circle is dTheta
  for c in range(nv):
    nodes[c, 0] = natR * np.cos(c * dTheta) # x coordinate of c-th node
    nodes[c, 1] = natR * np.sin(c * dTheta) # y coordinate of c-th node
    nodes[c, 2] = 0.0                       # z coordinate of c-th node

# Material parameters
Y = 386.66e3 # Pascals, modulus of elasticity of Elastosil MA 4601 
#Y = 169.56e3 # Pascals, effective modulus of elasticity for actuator made of Soft Translucent Silicone
G = Y / 3 # shear modulus (this is corresponding to an incompressible material)

# Stiffness parameters
EI = Y * np.pi * r0**4 / 4 # Bending stiffness
GJ = G * np.pi * r0**4 / 2 # Twisting stiffness
EA = Y * np.pi * r0**2     # Stretching stiffness

# Time stepping parameters
totalTime = 1 # second
dt = 0.01 # second (chosen based on sensitivity analysis)

# Tolerance
tol = EI / RodLength**2 * 1e-3
tol = tol[-1]

# Mass matrix
rho = 1130 # Density of Elastosil in water (kg/m^3)
#rho = 1100 # Density of Soft Translucent Silicone in water (kg/m^3)
totalM = np.zeros((nv))
dm = np.zeros((nv))

for c in range(nv):
  totalM[c] = (np.pi * r0[c]**2 * RodLength) * rho # total mass in kg
  dm[c] = totalM[c] / ne # mass per edge

massVector = np.zeros(ndof)
# mass at the nodes
for c in range(nv): # 0, 1, 2, ..., nv-1
  ind = [4*c, 4*c+1, 4*c+2]
  if c == 0: # first node
    massVector[ind] = dm[c]/2 # Since dm is mass per edge, at the first and last node the mass will be dm/2
  elif c == nv-1: # last node
    massVector[ind] = dm[c]/2 # Since dm is mass per edge, at the first and last node the mass will be dm/2
  else: # internal nodes
    massVector[ind] = dm[c]

# mass at the edges
for c in range(ne):
  massVector[4*c + 3] = 1/2 * dm[c] * r0[c]**2

# Diagonal matrix rerpesentation of mass vector
mMat = np.diag(massVector)

# External Force
# Gravity
g = np.array([0, 0, -9.81])
Fg = np.zeros(ndof) # External force vector for gravity
for c in range(nv):
  ind = [4*c, 4*c+1, 4*c+2]
  Fg[ind] = massVector[ind] * g

# Viscous parameters
rho_fluid = 1000 # kg/m^3 - density of water
visc = 1000 # viscosity of water

# Viscous damping matrix, C
C = np.zeros((ndof, ndof))
for k in range(nv):
  C[k,k] = 6 * np.pi * visc * r0[k] # Using Stoke's law assuming the viscous force on discrete spheres

# Initial DOF Vector
# DOF vector at t = 0
q0_finger1 = np.zeros(ndof)
q0_finger2 = np.zeros(ndof)
q0_finger3 = np.zeros(ndof)

# Assign initial node positions to q0 for each finger
for c in range(nv):
    ind = [4*c, 4*c+1, 4*c+2]
    q0_finger1[ind] = nodes[c, :] # Finger 1
    q0_finger2[ind] = rotate_points_3d(nodes[c, :], 120, axis='z')  # Rotate nodes for finger 2
    q0_finger3[ind] = rotate_points_3d(nodes[c, :], 240, axis='z') # Rotate nodes for finger 3


u = np.zeros_like(q0_finger1) # velocity vector
plt.figure(1)
plotrod_simple(q0_finger1, 0)

plt.figure(2)
plotrod_simple(q0_finger2, 0)

plt.figure(3)
plotrod_simple(q0_finger3, 0)

# FINGER 1

# Compute the reference (undeformed) length of each edge and the Voronoi length (undeformed) associated with each node
# Reference (undeformed) length of each edge
refLen = np.zeros(ne)
for c in range(ne): # loop over each edge
  dx = nodes[c+1, :] - nodes[c, :] # edge vector from one node to the next
  refLen[c] = np.linalg.norm(dx)

# Voronoi length of each node
voronoiRefLen = np.zeros(nv)
for c in range(nv): # loop over each node
  if c==0:
    voronoiRefLen[c] = 0.5 * refLen[c]
  elif c==nv-1:
    voronoiRefLen[c] = 0.5 * refLen[c-1]
  else:
    voronoiRefLen[c] = 0.5 * (refLen[c-1] + refLen[c])

# Reference frame (Space parallel transport at t=0)
a1 = np.zeros((ne,3)) # First reference director
a2 = np.zeros((ne,3)) # Second reference director
tangent = computeTangent(q0_finger1)

t0 = tangent[0,:] # tangent on the first edge
t1 = np.array([0, 0, -1]) # choosing an "arbitrary" vector so that a1 is perpendicular to t0
a1_first = np.cross(t0, t1) # This is perpendicular to tangent t0
# Check for null vector
if np.linalg.norm(a1_first) < 1e-6:
  t1 = np.array([0, 1, 0]) # new arbitrary vector
  a1_first = np.cross(t0, t1)
a1_first = a1_first / np.linalg.norm(a1_first) # Normalize to make it unit vector
a1, a2 = computeSpaceParallel(a1_first, q0_finger1)
# a1, a2, tangent all have size (ne,3)

# Material frame
theta = q0_finger1[3::4] # twist angles
m1, m2 = computeMaterialFrame(a1, a2, theta) # Compute material frame

# Reference twist
refTwist = np.zeros(nv)
refTwist = getRefTwist(a1, tangent, refTwist)

# Natural twist
twistBar = np.zeros(nv)

# Fixed and Free DOFs
fixedIndex = np.arange(0,7) # First seven (2 nodes and one edge) are fixed: clamped
freeIndex = np.arange(7,ndof)

Nsteps = round(totalTime / dt ) # Total number of steps
ctime = 0 # current time
endZ = np.zeros(Nsteps) # Store z-coordinate of the last node with time
endY = np.zeros(Nsteps) # Store y-coordinate of the last node with time
rod_positions_1 = [] # Store all of the positions

for timeStep in range(Nsteps):

  qGuess = q0_finger1.copy()
  q, u, a1, a2 = objfun(qGuess, # Guess solution
                        q0_finger1, u, a1, a2,
                        freeIndex, # Boundary conditions
                        dt, tol, # time stepping parameters
                        refTwist, # guess refTwist to compute the new refTwist
                        massVector, mMat, # Mass vector and mass matrix
                        EA, refLen, # reference length
                        EI, GJ, voronoiRefLen, twistBar, # stiffness
                        Fg, C, # external forces
                        ctime) # current time

  ctime += dt # Update current time

  # Update q0 with the new q
  q0_finger1 = q.copy()

  # Store the positions
  pos = np.zeros((3,nv))
  for c in range(nv):
    ind = [4*c, 4*c+1, 4*c+2]
    for i in range(3):
      pos[i,c] = q0_finger1[ind[i]]
  rod_positions_1.append(pos.copy())

  # Store the z-coordinate of the last node
  endZ[timeStep] = q[-1]
  endY[timeStep] = q[-2]

  if timeStep % 10 == 0:
    plotrod_simple(q, ctime)

# FINGER 2

# Reference frame (Space parallel transport at t=0)
a1 = np.zeros((ne,3)) # First reference director
a2 = np.zeros((ne,3)) # Second reference director
tangent = computeTangent(q0_finger2)

t0 = tangent[0,:] # tangent on the first edge
t1 = np.array([0, 0, -1]) # choosing an "arbitrary" vector so that a1 is perpendicular to t0
a1_first = np.cross(t0, t1) # This is perpendicular to tangent t0
# Check for null vector
if np.linalg.norm(a1_first) < 1e-6:
  t1 = np.array([0, 1, 0]) # new arbitrary vector
  a1_first = np.cross(t0, t1)
a1_first = a1_first / np.linalg.norm(a1_first) # Normalize to make it unit vector
a1, a2 = computeSpaceParallel(a1_first, q0_finger2)
# a1, a2, tangent all have size (ne,3)

# Material frame
theta = q0_finger2[3::4] # twist angles
m1, m2 = computeMaterialFrame(a1, a2, theta) # Compute material frame

# Reference twist
refTwist = np.zeros(nv)
refTwist = getRefTwist(a1, tangent, refTwist)

# Natural twist
twistBar = np.zeros(nv)

Nsteps = round(totalTime / dt ) # Total number of steps
ctime = 0 # current time
rod_positions_2 = [] # Store all of the positions

for timeStep in range(Nsteps):

  qGuess = q0_finger2.copy()
  q, u, a1, a2 = objfun(qGuess, # Guess solution
                        q0_finger2, u, a1, a2,
                        freeIndex, # Boundary conditions
                        dt, tol, # time stepping parameters
                        refTwist, # guess refTwist to compute the new refTwist
                        massVector, mMat, # Mass vector and mass matrix
                        EA, refLen, # reference length
                        EI, GJ, voronoiRefLen, twistBar, # stiffness
                        Fg, C, # external forces
                        ctime) # current time

  ctime += dt # Update current time

  # Update q0 with the new q
  q0_finger2 = q.copy()

  # Store the positions
  pos = np.zeros((3,nv))
  for c in range(nv):
    ind = [4*c, 4*c+1, 4*c+2]
    for i in range(3):
      pos[i,c] = q0_finger2[ind[i]]
  rod_positions_2.append(pos.copy())

    # Every 100 time steps, update material directors and plot the rod
  if timeStep % 10 == 0:
    plotrod_simple(q, ctime)

# FINGER 3

# Reference frame (Space parallel transport at t=0)
a1 = np.zeros((ne,3)) # First reference director
a2 = np.zeros((ne,3)) # Second reference director
tangent = computeTangent(q0_finger3)

t0 = tangent[0,:] # tangent on the first edge
t1 = np.array([0, 0, -1]) # choosing an "arbitrary" vector so that a1 is perpendicular to t0
a1_first = np.cross(t0, t1) # This is perpendicular to tangent t0
# Check for null vector
if np.linalg.norm(a1_first) < 1e-6:
  t1 = np.array([0, 1, 0]) # new arbitrary vector
  a1_first = np.cross(t0, t1)
a1_first = a1_first / np.linalg.norm(a1_first) # Normalize to make it unit vector
a1, a2 = computeSpaceParallel(a1_first, q0_finger3)
# a1, a2, tangent all have size (ne,3)

# Material frame
theta = q0_finger3[3::4] # twist angles
m1, m2 = computeMaterialFrame(a1, a2, theta) # Compute material frame

# Reference twist
refTwist = np.zeros(nv)
refTwist = getRefTwist(a1, tangent, refTwist)

# Natural twist
twistBar = np.zeros(nv)

Nsteps = round(totalTime / dt ) # Total number of steps
ctime = 0 # current time
rod_positions_3 = [] # Store all of the positions

for timeStep in range(Nsteps):

  qGuess = q0_finger3.copy()
  q, u, a1, a2 = objfun(qGuess, # Guess solution
                        q0_finger3, u, a1, a2,
                        freeIndex, # Boundary conditions
                        dt, tol, # time stepping parameters
                        refTwist, # guess refTwist to compute the new refTwist
                        massVector, mMat, # Mass vector and mass matrix
                        EA, refLen, # reference length
                        EI, GJ, voronoiRefLen, twistBar, # stiffness
                        Fg, C, # external forces
                        ctime) # current time

  ctime += dt # Update current time

  # Update q0 with the new q
  q0_finger3 = q.copy()

  # Store the positions
  pos = np.zeros((3,nv))
  for c in range(nv):
    ind = [4*c, 4*c+1, 4*c+2]
    for i in range(3):
      pos[i,c] = q0_finger3[ind[i]]
  rod_positions_3.append(pos.copy())

  if timeStep % 10 == 0:
    plotrod_simple(q, ctime)

######################
# PLOTTING THE GRIPPER
######################

animate_three_rods(rod_positions_1, rod_positions_2, rod_positions_3)

animate_three_rods_underwater(rod_positions_1, rod_positions_2, rod_positions_3)

plot_three_rods_timeline(rod_positions_1, rod_positions_2, rod_positions_3)

plot_gripper_plotly(rod_positions_1, rod_positions_2, rod_positions_3)

# Visualization after the loop
plt.figure(2)
time_array = np.arange(1, Nsteps + 1) * dt
plt.plot(time_array, endZ, 'ro-')
plt.box(True)
plt.xlabel('Time, t [sec]')
plt.ylabel('z-coord of last node, $\\delta_z$ [m]')
plt.show()
