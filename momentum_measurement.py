import ROOT as r
import matplotlib.pyplot as plt
import math
import argparse
import scipy.optimize as opt
import numpy

parser = argparse.ArgumentParser(description="Applies preselection cuts", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--task', type=str, default ="a", help='Select the task this data corresponds to.')
args = parser.parse_args()

tasks = {"a": {"B":0.5, "range": [0,200]}, "b_1": {"B": 0.25, "range": [0,200]}, "b_2": {"B":1, "range": [0,200]}, "c_1": {"B":0.5, "range": [0,400]}, "c_2": {"B":0.5, "range": [0,400]}}
n_bins = 80
bin_width = tasks[args.task]["range"][1]/n_bins


# define the detector properties
B = tasks[args.task]["B"]
plot_range = tasks[args.task]["range"]
#q = 1.60217662/(1e19)
q=3*(10**8)/(10**9) # Since we want GeV/c
L=1 # separation between the drift chambers
delta_z = 0.5 # separation between the layers in the drift chambers
mag_left_side = 4.5 # signed distance between start of first chamber and start of mag
mag_right_side = -2 # signed distance between end of mag and start of second chamber

def chisq(z,m,c): # function to minimise in track fitting
    return m*z+c

def track_reconstruction(event):
    #track1
    coords = []
    chamber_vector = [event.Dc1HitsVector_z,event.Dc1HitsVector_x]
    for i in range(len(chamber_vector[0])):
        coords.append([chamber_vector[0][i]*delta_z, chamber_vector[1][i]/1000])
    def chisq(pars): # function to minimise in track fitting
        chi2 = 0
        for x,z in coords:
            x_predicted = pars[0]*z+pars[1]
            chi2+=((x-x_predicted)*1e4)**2 # 1e4 comes from the uncertainty in x
        return chi2
    m = (coords[-1][1]-coords[0][1])/(coords[-1][0]-coords[0][0])
    c = coords[0][1] # treat start of chamber as z=0
    min = opt.minimize(chisq, x0=[m,c])
    m1 = min["x"][0]
    c1 = min["x"][1]

    #track2
    coords = []
    chamber_vector = [event.Dc2HitsVector_z,event.Dc2HitsVector_x]
    for i in range(len(chamber_vector[0])):
        coords.append([chamber_vector[0][i]*delta_z, chamber_vector[1][i]/1000])
    def chisq(pars): # function to minimise in track fitting
        chi2 = 0
        for x,z in coords:
            x_predicted = pars[0]*z+pars[1]
            chi2+=((x-x_predicted)*1e4)**2 # 1e4 comes from the uncertainty in x
        return chi2
    m = (coords[-1][1]-coords[0][1])/(coords[-1][0]-coords[0][0])
    c = coords[0][1] # treat start of chamber as z=0
    min = opt.minimize(chisq, x0=[m,c])
    m2 = min["x"][0]
    c2 = min["x"][1]

    return m1,c1,m2,c2

def track_angles(m1,m2): # calculate the angle between the outgoing (incoming) track and chamber 1 (2)
    return math.atan(m1), math.atan(m2)

def Delta_x(m1,c1,m2,c2):
    return abs((m1*mag_left_side+c1)-(m2*mag_right_side+c2))

def momentum(m1,c1,m2,c2):
    theta_1,theta_2 = track_angles(m1,m2)
    delta_x = Delta_x(m1,c1,m2,c2)
    return (B*q*math.sqrt(L**2 + delta_x**2))/(2*math.sin((theta_1+theta_2)/2))

def plot(momenta):
    n, bins, patches = plt.hist(momenta, bins=n_bins)
    plt.savefig("task_{}.pdf".format(args.task))
    plt.show()
    return n, bins

def FWHM(n, bins):
    maximum = max(n)
    candidates = []
    for i in range(len(n)):
        if n[i] > maximum/2: candidates.append(i)
    lower_edge = bins[candidates[0]]
    upper_edge = bins[candidates[-1]]
    #print(candidates, lower_edge, upper_edge)
    return 2*bin_width*(candidates[-1]-candidates[0]-1)/(1e9)


inf = r.TFile.Open("B5_{}.root".format(args.task))
tree = inf.Get("B5")
#tree.Print()
momenta = []
for event in tree:
    if len(event.Dc1HitsVector_x)==5 and len(event.Dc2HitsVector_x)==5:
        m1,c1,m2,c2 = track_reconstruction(event)
        m = momentum(m1,c1,m2,c2)
        if abs(m)>plot_range[0] and abs(m)<plot_range[1]: momenta.append(abs(m))


#for m in momenta:
    #if m>plot_range[0] and m<plot_range[1]: momenta.append(m)
n, bins = plot(momenta)
#print("{:.3f} GeV".format(FWHM(n, bins)))
