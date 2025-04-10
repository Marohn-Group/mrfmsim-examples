"""
Script that processes .mat data.
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt



class freqvsfield(object):
    """Class to process the .mat data."""
    def __init__(self, filename, title, n):
        self.filename = filename
        self.title = title
        self.n = n

    def load(self):
        """Read the data."""

        mat = scipy.io.loadmat(self.filename)

        data = mat["data"]

        self.f = data
        self.F = data[0, self.n][0, 0][7][0]
        self.X = data[0, self.n][0, 0][5][0]
        self.Y = data[0, self.n][0, 0][6][0]
        # self.p_opt = 0
        # self.message = f.attrs['comment']
        # self.tip_sample = f.attrs['tip-sample_separation']

    def plot(self, inline=True):
        """Plot the X and Y lock-in data versus."""

        plt.figure()
        plt.plot(self.F, self.X, "-o", color="teal", label="X")
        plt.plot(self.F, self.Y, "-", color="grey", label="Y")
        plt.xlabel("B$_0$ [kG]")
        # plt.xlim([34.4,38])
        plt.ylabel("signal [mHz]")
        # plt.ylim([-200,250])
        # plt.vlines(x=1.236*28, ymin=-30, ymax=30, color='g', linestyles = 'dashed')
        plt.title(self.title)
        plt.legend(loc="best")
        plt.grid()
        plt.show()

    def baseline(self, mask):
        self.X = self.X - np.average(self.X[mask])
        self.Y = self.Y - np.average(self.Y[mask])

    def phase(self, p0=0):
        """Autocorrect the phase."""

        phases = np.pi * (np.linspace(0.0, 1.0, 512) + p0)
        rX = np.zeros(phases.size)
        rY = np.zeros(phases.size)

        for idx, p in enumerate(phases):
            Xp = np.cos(p) * self.X + np.sin(p) * self.Y
            Yp = np.sin(p) * self.X - np.cos(p) * self.Y

            rX[idx] = (Xp * Xp).sum()
            rY[idx] = (Yp * Yp).sum()

        self.p_opt = phases[rY.argmin()]

        # print(self.p_opt)
        Xnew = np.cos(self.p_opt) * self.X + np.sin(self.p_opt) * self.Y
        Ynew = np.sin(self.p_opt) * self.X - np.cos(self.p_opt) * self.Y

        self.Xorig = self.X
        self.Yorig = self.Y

        self.X = Xnew
        self.Y = Ynew

    def pick_phase(self, p0=0):
        """Autocorrect the phase."""

        phases = np.pi * (np.linspace(0.0, 1.0, 512) + p0)
        rX = np.zeros(phases.size)
        rY = np.zeros(phases.size)

        for idx, p in enumerate(phases):
            Xp = np.cos(p) * self.X + np.sin(p) * self.Y
            Yp = np.sin(p) * self.X - np.cos(p) * self.Y

            rX[idx] = (Xp * Xp).sum()
            rY[idx] = (Yp * Yp).sum()

        self.p_opt = phases[rY.argmin()]

        # print(self.p_opt)
        Xnew = np.cos(self.p_opt) * self.X + np.sin(self.p_opt) * self.Y
        Ynew = np.sin(self.p_opt) * self.X - np.cos(self.p_opt) * self.Y

        self.Xorig = self.X
        self.Yorig = self.Y

        self.X = Xnew
        self.Y = Ynew


def parse_mat(filename):
    """Parse the .mat file."""
    data = {}
    mat = scipy.io.loadmat(filename)
    length = mat["data"].shape[1]
    for i in range(length):
        tip_samp = float(mat["data"][0, i][0, 0][3][0].split(" ")[0])

        a = freqvsfield(filename, "tip sample = " + str(tip_samp), i)
        a.load()
        if tip_samp < 100.0:
            a.baseline(a.F > 8.0)
            a.phase(0.0)

            std = np.std(a.X[a.F > 8.0])
            mean = np.mean(a.X[a.F > 8.0])

            # print(std)

            val, k = 0, 0
            while abs(val - mean) < 2 * std:
                k = k - 1
                val = a.X[k]

            a.bound = a.F[k]

        else:
            a.baseline(a.F > 7.75)
            a.phase(0.0)

            std = np.std(a.X[a.F > 7.75])
            mean = np.mean(a.X[a.F > 7.75])

            val, k = 0, 0
            while abs(val - mean) < 4 * std:
                k = k - 1
                val = a.X[k]

            a.bound = a.F[k]

        a.angle = np.arctan2(a.Y, a.X)

        data[tip_samp] = a
    # sort
    data = dict(sorted(data.items()))

    return data

