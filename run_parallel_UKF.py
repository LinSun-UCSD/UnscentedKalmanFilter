import numpy as np
import matplotlib.pyplot as plt
from ismember import ismember

from compute_RRMS import compute_RRMS
import time as t
from get_mass import get_mass
from get_stiffness import get_stiffness
from get_classical_damping import get_classical_damping
from get_continuous_state_space import get_continuous_state_space
from get_response_state_space import get_response_state_space
from compute_response_SPs import compute_response_SPs
from multiprocessing import Pool


def h_measurement_eqn(parameter_SP, GMinput, StepUpdateSize, UpdateNum):
    input_path = GMinput["path"]
    filename = GMinput["filename"]
    fs = GMinput["fs"]
    num_SP = parameter_SP.shape[1]
    # input K, M and C
    DOF = 8
    m0 = 625000
    M_global = get_mass(m0, DOF)
    c0 = 400000
    damping = {
        "mode": np.array([1, 3]),
        "ratio": np.array([0.05, 0.05])
    }
    B = np.ones((DOF, 1))
    # load the input
    temp = np.loadtxt(input_path + "/" + filename + ".txt", dtype=float)
    a = temp[2:temp.shape[0]] * 9.81

    # compute the response

    output_type = ["abs", "rel"]
    if num_SP > 1:
        step = StepUpdateSize * (UpdateNum+1)
        temp = np.zeros((DOF, step, num_SP))
        t = np.arange(0, step, 1) / fs
        response = np.zeros((DOF*StepUpdateSize, num_SP))
        # for i in range(num_SP):
        #     K_global = get_stiffness(parameter_SP[:, i], DOF)
        #     C_global, _, _, _ = get_classical_damping(K_global, M_global, damping, "no")
        #     Ac, Bc, Cc, Dc = get_continuous_state_space(K_global, M_global, C_global, B, output_type[0])
        #     temp[:, :, i], _, _, _, _, _ = get_response_state_space(Ac, Bc, Cc, Dc, a[0: step], t)
        toPass = [(parameter_SP[:, i], M_global, damping, DOF, B, output_type, step, a, t) for i in range(parameter_SP.shape[1])]
        if __name__ == '__main__':
            pool = Pool(processes=8)
            results = pool.starmap(compute_response_SPs, toPass)
            print(results)
            pool.close()
        for i in range(results):
            temp[:, :, i] = results[i]
        for j in range(num_SP):
            if StepUpdateSize == 1:
                temp1 = temp[:, temp.shape[1]-1, j]
                response[:, j] = temp1
            else:
                temp1 = temp[:, temp.shape[1]-1-(StepUpdateSize-1):temp.shape[1]-1+1, j]
                temp2 = np.reshape(np.transpose(temp1), (StepUpdateSize*temp.shape[0], 1))
                response[:, j] = temp2.reshape((temp2.shape[0],))
    elif num_SP == 1:
        step = StepUpdateSize * UpdateNum
        temp = np.zeros((DOF, step, num_SP))
        t = np.arange(0, step, 1) / fs
        K_global = get_stiffness(parameter_SP, DOF)
        C_global, _, _, _ = get_classical_damping(K_global, M_global, damping, "no")
        Ac, Bc, Cc, Dc = get_continuous_state_space(K_global, M_global, C_global, B, output_type[0])
        temp, _, _, _, _, _ = get_response_state_space(Ac, Bc, Cc, Dc, a[0:step], t)
        response = np.transpose(temp)
    return response


def is_pos_def(A):
    M = np.matrix(A)
    return np.all(np.linalg.eigvals(M + M.transpose()) > 0)


def GenerateSPs(muZ, Pz, SPsPar):
    Pz = (0.5 * Pz) + np.transpose(0.5 * Pz)
    # Cholesky decomposition
    ok = is_pos_def(Pz)
    if not ok:
        Pz = Pz + 2.2204e-16 * np.eye(Pz)
        cholPz = np.linalg.cholesky(Pz)
    else:
        cholPz = np.linalg.cholesky(Pz)

    Zsp = np.transpose(np.block ([[muZ], [muZ + np.transpose(SPsPar["gamma"] * cholPz)], \
                                 [muZ - np.transpose(SPsPar['gamma'] * cholPz)]]))
    return Zsp


def UT(gZsps, SPsPar, noiseCOVARIANCE):
    # estimate of mean and covariance matrix of s
    muS = np.transpose(np.matmul(SPsPar["Wm"], np.transpose(gZsps)))
    temp1 = np.zeros(gZsps.shape)
    for i in range(muS.shape[0]):
        temp1[i, :] = gZsps[i, :] - np.transpose(np.ones((gZsps.shape[1], 1)) * muS[i])
    temp = np.matmul(np.matmul(temp1, np.diag(SPsPar["Wc"].reshape((SPsPar["Wc"].shape[1],)))), np.transpose(temp1))
    Pss = temp + noiseCOVARIANCE
    return muS, Pss


def UKF(xcap_km1_km1, Pxxcap_km1_km1, Q_km1, R_k, measurement_Yk, GMinput, SPsPar, StepUpdateSize, UpdateNum):
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
    y_k_km1_SPs = h_measurement_eqn(x_k_km1_SPs, GMinput, StepUpdateSize, UpdateNum)
    ycap_k_km1, Pyycap_k_km1 = UT(y_k_km1_SPs, SPsPar, R_k)
    Pxycap_k_km1 = np.matmul(np.matmul(np.subtract(x_k_km1_SPs, xcap_k_km1),\
                             np.diag(SPsPar["Wc"].reshape((SPsPar["Wc"].shape[1],)))) \
                   , np.transpose(np.subtract(y_k_km1_SPs, ycap_k_km1)))
    # Kalman gain
    Kk = np.matmul(Pxycap_k_km1, np.linalg.pinv(Pyycap_k_km1))

    # update parameters estimate
    xcap_k_k = np.add(xcap_k_km1, np.matmul(Kk, np.subtract(measurement_Yk, ycap_k_km1)))

    # Update Parameter Covariance
    Pxxcap_k_k = np.subtract(Pxxcap_k_km1, Kk.dot(Pyycap_k_km1).dot(np.transpose(Kk)))
    return xcap_k_k.reshape((xcap_k_k.shape[0],)), Pxxcap_k_k

start_time = t.time()
g = 9.81
GMinput = {
    "totalStep": 1000,
    "fs": 50,
    "filename": 'NORTHR_SYL090',
    "path": "D:/UCSD PhD/MatlabCode/SystemIdentification/2D shear building model/matlab model/earthquake record"
}
NumData = 1000
StepUpdateSize = 1
UpdateStep = np.arange(StepUpdateSize, NumData + 1, StepUpdateSize)
TotalUpdateSteps = len(UpdateStep)
# Parameters to update
ParameterInfo = {
    "ParameterName": ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"],
    "TrueParameterValues": np.ones((8, 1), ) * 1e9,
    "UpdateParameterIndex": []
}
UpdateParameterName = ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
ParameterInfo["UpdateParameterIndex"] = ismember(ParameterInfo["ParameterName"], UpdateParameterName)
TrueUpdateParameterValues = np.zeros((8,))
for i in range(len(ParameterInfo["UpdateParameterIndex"])):
    if ParameterInfo["UpdateParameterIndex"][i]:
        TrueUpdateParameterValues[i] = ParameterInfo["TrueParameterValues"][i]

nTheta = len(UpdateParameterName)

# compute the true response
TrueResponse = h_measurement_eqn(ParameterInfo["TrueParameterValues"], GMinput, 1, GMinput["totalStep"])

# prior knowledge of the parameters
xcap_0_0 = np.multiply(np.array([[1.50, 1.50, 0.65, 0.245, 0.825, 1.235, 0.835, 0.45]]), TrueUpdateParameterValues)
cov_0_0 = 0.15
Pxxcap_0_0 = np.power(cov_0_0 * np.abs(xcap_0_0), 2)

# process noise
q = -0.00001
Q = np.power(q * xcap_0_0, 2)

# measurement noise
ny = 8
RMS_measurementNoise = 0.7 * 9.81 / 100
R = RMS_measurementNoise ** 2 * np.ones((StepUpdateSize * ny, 1))

# noise for polluting data
# NoisyTrueResponse = TrueResponse + np.random.randn(TrueResponse.shape[0], TrueResponse.shape[1])*0.01*g
NoisyTrueResponse = TrueResponse
##
xcap = np.zeros((nTheta, TotalUpdateSteps+1))
Pxxcap = np.zeros((nTheta, nTheta, TotalUpdateSteps + 1))
xcap[:, 0] = xcap_0_0
Pxxcap[:, :, 0] = np.diag(Pxxcap_0_0.reshape((Pxxcap_0_0.shape[1],)))
SPsPar = {
    "alpha": 0.01,
    "beta": 2,
    "kappa": 0
}

for UpdateNum in range(TotalUpdateSteps):
    GMinput["Nsteps"] = UpdateStep[UpdateNum]
    NoisyTrueResponseMatrix = NoisyTrueResponse[UpdateStep[UpdateNum] - StepUpdateSize:UpdateStep[UpdateNum], :]
    NoisyTrueResponseVector = np.reshape(NoisyTrueResponseMatrix, (StepUpdateSize * ny, 1))
    print("UpdateStep: " + str(UpdateStep[UpdateNum]) + "........Progress: " + str(UpdateNum * 100 / len(UpdateStep)) \
          + "% Done")
    xcap[:, UpdateNum+1], Pxxcap[:, :, UpdateNum+1] = UKF(xcap[:, UpdateNum],
                                                              Pxxcap[:, :, UpdateNum],
                                                              np.diag(Q.reshape((Q.shape[1],))),
                                                              np.diag(R.reshape((R.shape[0],))),
                                                              NoisyTrueResponseVector, GMinput,
                                                              SPsPar, StepUpdateSize, UpdateNum)

# plot the results
plt.figure()
for i in range(nTheta):
    sig = np.sqrt(Pxxcap[i, i, :])

    time = np.append([0], UpdateStep/GMinput["fs"])
    time.reshape((len(time), 1))
    X = np.block([time, np.flipud(time)])
    temp1 = ((xcap[i, 0:TotalUpdateSteps+1]-sig)/TrueUpdateParameterValues[i])
    temp2 = ((xcap[i, 0:TotalUpdateSteps+1]+sig)/TrueUpdateParameterValues[i])
    plt.subplot(4, 2, i+1)
    plt.fill_between(time, temp1, temp2, color=(0.75, 0.75, 0.75))
    plt.plot(time, xcap[i, 0:TotalUpdateSteps+1]/TrueUpdateParameterValues[i], color='b')
    plt.grid('auto')
    plt.hlines(1, 0, 1, colors='r', linestyles='--')
    plt.ylabel(UpdateParameterName[i] + "/" + UpdateParameterName[i] + r'$_{true}$')
    plt.subplots_adjust(hspace=0.8, wspace=0.5)
    plt.xlabel('Time [sec]')



# compute the RRMS
response = np.zeros((GMinput["totalStep"], xcap.shape[0], xcap.shape[1]))
for i in range(xcap.shape[1]):
    response[:, :, i] = h_measurement_eqn(xcap[:, i].reshape((xcap[:,i].shape[0],1)), GMinput, 1, GMinput["totalStep"])

RRMS = np.zeros((TrueResponse.shape[1], xcap.shape[1]))
for k in range(xcap.shape[1]):
    for DOF in range(TrueResponse.shape[1]):
        RRMS[DOF, k] = compute_RRMS(TrueResponse[:,DOF], response[:, DOF, k])
plt.figure()
for i in range(RRMS.shape[0]):
    plt.plot(time, RRMS[i, :], label="DOF"+str(i+1))
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('RRMS')
plt.grid('auto')

print("--- Execution time: %s seconds ---" + str(t.time() - start_time))
# plt.show()
# plt.close()
