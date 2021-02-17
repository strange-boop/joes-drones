import math
import numpy as np
from scipy.spatial.transform import Rotation
from src.data import measurements, System


def gen_rot_x(angle):
    co = math.cos(angle)
    si = math.sin(angle)

    return np.array([
        [1, 0, 0],
        [0, co, -si],
        [0, si, co]])


def gen_rot_z(angle):
    co = math.cos(angle)
    si = math.sin(angle)

    return np.array([
        [co, -si, 0],
        [si, co, 0],
        [0, 0, 1]])


# Transform a point in ref frame x to ref frame r using transformation x->r
def transform_point_x_to_r(v_x, Px_r, Rx_r):
    return np.dot(Rx_r, v_x) + Px_r


# Transform a rotation in ref frame x to ref frame r using transformation x->r
def transform_rot_x_to_r(Rx_y, Ry_r):
    return np.dot(Ry_r, Rx_y)


# Transform a base station (y) in ref frame x to ref frame r using
# transformation x->r
def transform_x_to_r(Sy_x, Sx_r):
    Py_r = transform_point_x_to_r(Sy_x.P, Sx_r.P, Sx_r.R)
    Ry_r = transform_rot_x_to_r(Sy_x.R, Sx_r.R)
    return System(Py_r, Ry_r, Sy_x.bs)


# Transform a point in ref frame r to ref frame x using transformation x->r
def transform_point_to_x_from_r(v_r, Px_r, Rx_r):
    Rr_x = np.matrix.transpose(Rx_r)
    return np.dot(Rr_x, v_r - Px_r)


# Find the transformation x->r when we know the bs pos/rot for both x and r
def transform_from_ref_x_to_r_same_bs(Sb_x, Sb_r):
    Rb_x_inv = np.matrix.transpose(Sb_x.R)
    Rx_r = np.dot(Sb_r.R, Rb_x_inv)
    Px_r = Sb_r.P - np.dot(Rx_r, Sb_x.P)

    return System(Px_r, Rx_r, Sb_x.bs)


# Averaging of quaternions
# From https://stackoverflow.com/a/61013769
def q_average(Q, W=None):
    if W is not None:
        Q *= W[:, None]
    eigvals, eigvecs = np.linalg.eig(Q.T@Q)
    return eigvecs[:, eigvals.argmax()]


def system_average(S):
    bs = S[0].bs
    for s in S:
        if s.bs != bs:
            raise Exception("Different base stations")

    Q = map(lambda s : Rotation.from_matrix(s.R).as_quat(), S)
    q = q_average(np.array(list(Q)))
    r = Rotation.from_quat(q).as_matrix()

    P = map(lambda s : s.P, S)
    p = np.average(np.array(list(P)), axis=0)

    return System(p, r, bs)


def print_system(system):
    print(f"Base station {system.bs} @ {system.P}, {system.R}")


def probe_position(Sbs_ref, Sbs0_other, Sbs1_other):
    """
    Find the position of

    Args:
        Sbs_ref: the reference system that defines global coordinates
        Sbs0_other: the reference system of our current measurement
        Sbs1_other: the reference system of a measurement not

    Returns:

    """

    # Find transform from other ref frame to global using bs0
    Sother_g = transform_from_ref_x_to_r_same_bs(Sbs0_other, Sbs_ref)

    # The measurement of base station 1 in the other ref frame, converted to
    # global
    Sbs1mOther_g = transform_x_to_r(Sbs1_other, Sother_g)

    return Sbs1mOther_g


print()

# Arbitrarily choose a base station measurement to use as our reference
ref = measurements[0][0]

# A list of Systems found in the reference system
result = [ref]

# A list of base stations found
found = [ref.bs]

not_done = True

while not_done:
    not_done = False
    samples = {}

    print("--- iteration")

    for measurement in measurements:
        # Find a reference system in this measurement
        from_bs = None  # from base station
        from_sys = None  # from system
        from_sys_g = None  # from sys g

        # Iterate through individual base stations in this measurement
        for system in measurement:
            bs = system.bs  # The base station for this measurement

            # Iterate through systems in the reference systems (our known path)
            # until we find one which has a common base station with the
            # reference systems
            for ref_sys in result:
                if ref_sys.bs == bs:
                    # The first ref_sys which has the same base station
                    # as our the current base station
                    from_bs = bs
                    from_sys = system
                    from_sys_g = ref_sys
                    break
            if from_bs is not None:
                break

        # Transform all base stations in this measurement to
        # the global system, unless we already have a result for a
        # particular base station.
        if from_bs is not None:
            print(f"Using {from_bs} as reference")
            for system in measurement:
                if system is not from_sys:
                    if system.bs not in found:
                        s = probe_position(from_sys_g, from_sys, system)
                        from_to = (from_bs, system.bs)
                        if not from_to in samples:
                            samples[from_to] = []
                        samples[from_to].append(s)
        else:
            not_done = True

    # Average over all samples sets
    for from_to, sample_set in samples.items():
        new_sys = system_average(sample_set)
        result.append(new_sys)
        found.append(new_sys.bs)


for system in result:
    print_system(system)
