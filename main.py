# this code is to do unscented Kalman Filter for a 2D shear building
# date: August 20, 2021
# Author: Lin Sun
# email: lsun@ucsd.edu

import numpy as np
from ismember import ismember
from h_measurement_eqn import h_measurement_eqn
from UKF import UKF
import time as t
import os as os

start_time = t.time()
resultPath = "result/"  # folder to store posterior
g = 9.81  # gravity acceleration
# ground motion input
GMinput = {
    "totalStep": 1000,  # earthquake record stpes
    "fs": 50,  # sampling rate
    "filename": 'NORTHR_SYL090',  # the earthquake file to load
    "path": os.getcwd() + "\\earthquake record"  # earthquake record folder
}
NumData = 1000  # total number of data
StepUpdateSize = 4  # update step size
UpdateStep = np.arange(StepUpdateSize, NumData + 1, StepUpdateSize)  # which steps to update
TotalUpdateSteps = len(UpdateStep)  # total update steps
# Parameters to update
ParameterInfo = {
    "ParameterName": ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"],  # stiffness at each floor
    "TrueParameterValues": np.ones((8, 1), ) * 1e9,  # true parameters
    "UpdateParameterIndex": []
}
k0 = 1e9
UpdateParameterName = ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
ParameterInfo["UpdateParameterIndex"] = ismember(ParameterInfo["ParameterName"], UpdateParameterName)
TrueUpdateParameterValues = np.zeros((len(UpdateParameterName),))
count = 0
for i in range(len(ParameterInfo["UpdateParameterIndex"])):
    if ParameterInfo["UpdateParameterIndex"][i]:
        TrueUpdateParameterValues[count] = ParameterInfo["TrueParameterValues"][i]
        count = count + 1

nTheta = len(UpdateParameterName)

# compute the true response
measure_vector = np.array([[0, 7]])  # take first floor and top floor acceleration as measurement
TrueResponse = h_measurement_eqn(ParameterInfo["TrueParameterValues"], GMinput, 1, GMinput["totalStep"], measure_vector,
                                 k0)

# prior knowledge of the parameters
xcap_0_0 = np.multiply(np.array([[1.14, 1.32, 0.35, 0.95, 0.87, 1.235, 0.935, 0.85]]), TrueUpdateParameterValues)
cov_0_0 = 0.3  # covarianve
Pxxcap_0_0 = np.power(cov_0_0 * np.abs(xcap_0_0), 2)  # prior knowledge of covariance

# process noise
q = 0.00001
Q = np.power(q * xcap_0_0, 2)

# measurement noise
ny = 2
RMS_measurementNoise = 1 * g / 100
R = RMS_measurementNoise ** 2 * np.ones((StepUpdateSize * ny, 1))

# noise for polluting data
NoisyTrueResponse = TrueResponse + np.random.randn(TrueResponse.shape[0], TrueResponse.shape[1]) * 0.01 * g

# array to store results
xcap = np.zeros((nTheta, TotalUpdateSteps + 1))  # posterior mean
Pxxcap = np.zeros((nTheta, nTheta, TotalUpdateSteps + 1))  # posterior covariance
xcap[:, 0] = xcap_0_0  # starts from prior mean
Pxxcap[:, :, 0] = np.diag(Pxxcap_0_0.reshape((Pxxcap_0_0.shape[1],)))  # starts from prior covariance
# sigma points parameters
SPsPar = {
    "alpha": 0.01,
    "beta": 2,
    "kappa": 0
}
# start unscented kalman filtering
for UpdateNum in range(TotalUpdateSteps):
    GMinput["Nsteps"] = UpdateStep[UpdateNum]
    # get the noisy true response matrix and vector
    NoisyTrueResponseMatrix = NoisyTrueResponse[UpdateStep[UpdateNum] - StepUpdateSize:UpdateStep[UpdateNum], :]
    NoisyTrueResponseVector = np.reshape(NoisyTrueResponseMatrix, (StepUpdateSize * ny, 1))
    print("UpdateStep: " + str(UpdateStep[UpdateNum]) + "........Progress: " + str(
        (UpdateNum + 1) * 100 / len(UpdateStep)) + "% Done")
    xcap[:, UpdateNum + 1], Pxxcap[:, :, UpdateNum + 1] = UKF(xcap[:, UpdateNum],
                                                              Pxxcap[:, :, UpdateNum],
                                                              np.diag(Q.reshape((Q.shape[1],))),
                                                              np.diag(R.reshape((R.shape[0],))),
                                                              NoisyTrueResponseVector, GMinput,
                                                              SPsPar, StepUpdateSize, UpdateNum, measure_vector, k0)
    print("Parameters at:   " + str(np.divide(xcap[:, UpdateNum + 1], TrueUpdateParameterValues)))

# save results to txt files
np.save(resultPath + "Pxxcap.npy", Pxxcap) # posterior covariance
np.save(resultPath + "xcap.npy", xcap) # posterior mean
np.save(resultPath + "TrueUpdateParameterValues.npy", TrueUpdateParameterValues)
np.save(resultPath + "TrueResponse.npy", TrueResponse)
np.save(resultPath + "measure_vector", measure_vector)

print("--- Execution time: %s seconds ---" + str(t.time() - start_time))
