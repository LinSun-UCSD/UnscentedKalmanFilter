from compute_RRMS import compute_RRMS
import matplotlib.pyplot as plt
import numpy as np
from h_measurement_eqn import h_measurement_eqn

resultPath = "result/"
k0 = 1e9
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
UpdateParameterName = ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
TotalUpdateSteps = len(UpdateStep)
nTheta = len(UpdateParameterName)

# load the results
Pxxcap = np.load(resultPath + "Pxxcap.npy")
xcap = np.load(resultPath + "xcap.npy")
TrueUpdateParameterValues = np.load(resultPath + "TrueUpdateParameterValues.npy")
TrueResponse = np.load(resultPath + "TrueResponse.npy")
measure_vector = np.load(resultPath + "measure_vector.npy")

# plot the results
plt.figure(figsize=(12, 7))
plt.rcParams['font.family'] = 'Times New Roman'


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
    plt.hlines(1, 0, time[len(time)-1], colors='r', linestyles='--')
    plt.ylabel(UpdateParameterName[i] + "/" + UpdateParameterName[i] + r'$_{true}$')
    plt.subplots_adjust(hspace=0.8, wspace=0.5)
    plt.xlabel('Time [sec]')
plt.savefig('parameter evolution.png', dpi=800)
plt.show()

# compute the RRMS
response = np.zeros((GMinput["totalStep"], len(measure_vector[0]), xcap.shape[1]))
for i in range(xcap.shape[1]):
    response[:, :, i] = h_measurement_eqn(xcap[:, i].reshape((xcap[:, i].shape[0], 1)), GMinput, 1, GMinput["totalStep"], measure_vector, k0)

RRMS = np.zeros((TrueResponse.shape[1], xcap.shape[1]))
for k in range(xcap.shape[1]):
    for DOF in range(TrueResponse.shape[1]):
        RRMS[DOF, k] = compute_RRMS(TrueResponse[:, DOF], response[:, DOF, k])
plt.figure(figsize=(8, 6))
for i in range(RRMS.shape[0]):
    plt.plot(time, RRMS[i, :], label="DOF"+str(measure_vector[0][i]))
plt.legend()
plt.xlabel('Time [sec]')
plt.ylabel('RRMS')
plt.grid('auto')
plt.savefig('RRMS.png', dpi=800)
plt.show()