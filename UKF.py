import numpy as np
from h_measurement_eqn import h_measurement_eqn


# check whether matrix is positive definite or not
def is_pos_def(A):
    M = np.matrix(A)
    return np.all(np.linalg.eigvals(M + M.transpose()) > 0)


# generate sigma points
def GenerateSPs(muZ, Pz, SPsPar):
    Pz = (0.5 * Pz) + np.transpose(0.5 * Pz)
    # Cholesky decomposition
    ok = is_pos_def(Pz)
    if not ok:
        Pz = Pz + 2.2204e-16 * np.eye(Pz)
        cholPz = np.linalg.cholesky(Pz)
    else:
        cholPz = np.linalg.cholesky(Pz)

    Zsp = np.transpose(np.block([[muZ], [muZ + np.transpose(SPsPar["gamma"] * cholPz)], \
                                 [muZ - np.transpose(SPsPar['gamma'] * cholPz)]]))
    return Zsp


# unscented transformation
def UT(gZsps, SPsPar, noiseCOVARIANCE):
    # estimate of mean and covariance matrix of s
    muS = np.transpose(np.matmul(SPsPar["Wm"], np.transpose(gZsps)))
    temp1 = np.zeros(gZsps.shape)
    for i in range(muS.shape[0]):
        temp1[i, :] = gZsps[i, :] - np.transpose(np.ones((gZsps.shape[1], 1)) * muS[i])
    temp = np.matmul(np.matmul(temp1, np.diag(SPsPar["Wc"].reshape((SPsPar["Wc"].shape[1],)))), np.transpose(temp1))
    Pss = temp + noiseCOVARIANCE

    return muS, Pss


def UKF(xcap_km1_km1, Pxxcap_km1_km1, Q_km1, R_k, measurement_Yk, GMinput, SPsPar, StepUpdateSize, UpdateNum,
        measure_vector, k0):
    # set up the UT parameters
    SPsPar["nz"] = len(xcap_km1_km1)  # the SP number
    SPsPar["lambda"] = SPsPar["alpha"] ** 2 * (SPsPar["nz"] + SPsPar["kappa"]) - SPsPar["nz"]
    SPsPar["gamma"] = np.sqrt(SPsPar["nz"] + SPsPar["lambda"])

    # weight coefficients of SPs to estimate mean of s: muS
    Wm0 = SPsPar["lambda"] / (SPsPar["nz"] + SPsPar["lambda"])
    Wmi = (1 / (2 * (SPsPar["nz"] + SPsPar["lambda"]))) * np.ones((1, 2 * SPsPar["nz"]))
    SPsPar["Wm"] = np.block([Wm0, Wmi])

    # weight coefficients of SPs to estimate covariance of s: Pss
    Wc0 = (SPsPar["lambda"] / (SPsPar["nz"] + SPsPar["lambda"])) + (1 - SPsPar["alpha"] ** 2 + SPsPar["beta"])
    Wci = (1 / (2 * (SPsPar["nz"] + SPsPar["lambda"]))) * np.ones((1, 2 * SPsPar["nz"]))
    SPsPar["Wc"] = np.block([Wc0, Wci])
    x_km1_km1_SPs = GenerateSPs(xcap_km1_km1, Pxxcap_km1_km1, SPsPar)

    # propagate SPs of x(k-1|k-1) through state equation
    x_k_km1_SPs = x_km1_km1_SPs  # x_k_km1_SPs = (ntheta x 2ntheta+1)

    # Parameter Prediction
    # calculate statistics of x(k|k-1)
    xcap_k_km1, Pxxcap_k_km1 = UT(x_k_km1_SPs, SPsPar, Q_km1)  # xcap_k_km1 = (ntheta x 1)

    # update step
    # propagate SPs through the measurement equation
    y_k_km1_SPs = h_measurement_eqn(x_k_km1_SPs, GMinput, StepUpdateSize, UpdateNum, measure_vector, k0)
    print("max:" + str(np.max(R_k)))
    ycap_k_km1, Pyycap_k_km1 = UT(y_k_km1_SPs, SPsPar, R_k)
    Pxycap_k_km1 = np.matmul(np.matmul(np.subtract(x_k_km1_SPs, xcap_k_km1), \
                                       np.diag(SPsPar["Wc"].reshape((SPsPar["Wc"].shape[1],)))) \
                             , np.transpose(np.subtract(y_k_km1_SPs, ycap_k_km1)))
    # Kalman gain
    Kk = np.matmul(Pxycap_k_km1, np.linalg.pinv(Pyycap_k_km1))

    # update parameters estimate
    xcap_k_k = np.add(xcap_k_km1, np.matmul(Kk, np.subtract(measurement_Yk, ycap_k_km1)))

    # Update Parameter Covariance
    Pxxcap_k_k = np.subtract(Pxxcap_k_km1, Kk.dot(Pyycap_k_km1).dot(np.transpose(Kk)))
    return xcap_k_k.reshape((xcap_k_k.shape[0],)), Pxxcap_k_k
