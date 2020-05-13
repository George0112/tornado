import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import operator
from collections import OrderedDict

from dictionary.tornado_dictionary import TornadoDic
from regression.regression import SuperRegression
from sklearn import svm
from sklearn.metrics import r2_score


class Expo(SuperRegression):

    LEARNER_NAME = 'EXPO_CURVE_FIT'
    LEARNER_TYPE = TornadoDic.TRAINABLE
    LEARNER_TYPE = TornadoDic.NUM_REGRESSION

    def __init__(self):
        super().__init__()
        self.x = []
        self.y = []
        self.pred_y = []
        self.clf = svm.SVR()
        self.f = lambda x,a,b,c: a*np.exp(x*-b)+c
        pass

    def train(self, instance):
        # SVR
        # self.NUMBER_OF_INSTANCES_OBSERVED += 1
        # self.x.append([float(instance[0])])
        # self.y.append(float(instance[1]))
        # self.clf.fit(self.x[-100:], self.y[-100:])
        # self.pred_y.append(self.clf.predict([[float(instance[0])]]))
        # r = r2_score(self.y, self.pred_y)

        # CURVE FIT
        self.x.append(float(instance[0]))
        self.y.append(float(instance[1]))
        if len(self.x) < 3:
            return -1
        self.popt, pcov = curve_fit(self.f, self.x, self.y, maxfev=100000, p0=[1,1,0])
        residuals = np.array([])
        for i in range(len(self.x)):
            residuals = np.append(residuals, float(self.y[i]) - float(self.f(self.x[i], self.popt[0], self.popt[1], self.popt[2])))
        ss_res = np.sum(residuals[-100:]**2)
        ss_tot = np.sum(self.y - np.mean(self.y)**2)
        ymean = np.mean(self.y)
        ss_tot = np.dot((self.y-ymean), (self.y-ymean))
        r = 1-ss_res/ss_tot
        self.set_ready()
        return 1-r
        pass

    def test(self, instance):
        # return self.clf.predict([[float(instance[0])]])
        return self.f(instance[0], *self.popt)

    def getError(self):
        return super().getError()
    
    def reset(self):
        self.x = []
        self.y = []
        self.pred_y = []
        super()._reset_stats()

    

