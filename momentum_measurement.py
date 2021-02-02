import ROOT as r
import matplotlib.pyplot as plt
import math
import argparse
import scipy.optimize as opt
import numpy

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Applies preselection cuts", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--task', type=str, default ="a", help='Select the task this data corresponds to.')
parser.add_argument('--skip_multiple_hits', default=False, action="store_true", help='Skip fitting events with multiple hits in at least one drift chamber layer to make it run quicker.')
args = parser.parse_args()

tasks = {"a": {"B":0.5, "range": [0,200]}, "b_1": {"B": 0.25, "range": [0,200]}, "b_2": {"B":1, "range": [0,200]}, "c_1": {"B":0.5, "range": [0,100]}, "c_2": {"B":0.5, "range": [0,400]}}
n_bins = 80
bin_width = tasks[args.task]["range"][1]/n_bins


# define the detector properties
B = tasks[args.task]["B"]
plot_range = tasks[args.task]["range"]
#q = 1.60217662/(1e19)
q=3*(10**8)/(10**9) # Since we want GeV/c
L=2 # separation between the drift chambers
delta_z = 0.5 # separation between the layers in the drift chambers
mag_left_side = 4.5 # signed distance between start of first chamber and start of mag
mag_right_side = -2. # signed distance between end of mag and start of second chamber

def track_reconstruction(chamber_vector): # reconstruct the track in a given chamber
    # dictionary to store hits in so I can handle events where there are multiple hits in a given drift chamber layer
    track_dict = {"0": [], "1": [], "2": [], "3": [], "4":[]}
    for i in range(len(chamber_vector[0])):
        track_dict[str(int(chamber_vector[0][i]))].append([chamber_vector[0][i]*delta_z, chamber_vector[1][i]/1000])
    tracks = [[i,j,k,l,m] for i in track_dict["0"] for j in track_dict["1"] for k in track_dict["2"] for l in track_dict["3"] for m in track_dict["4"]]
    if tracks != []: #skip events where we don't get a hit in every layer, could perhaps incorporate them but it will take some thought
        track_fits = []
        for coords in tracks:
            def chisq(pars): # function to minimise in track fitting
                chi2 = 0
                for x,z in coords:
                    x_predicted = pars[0]*z+pars[1]
                    chi2+=((x-x_predicted)*1e4)**2 # 1e4 comes from the uncertainty in x
                return chi2
            # Determine some initial parameter values
            m = (coords[-1][1]-coords[0][1])/(coords[-1][0]-coords[0][0])
            c = coords[0][1] # treat start of chamber as z=0
            # Fit the track
            minimum = opt.minimize(chisq, x0=[m,c])
            minimum_chi2 = minimum["fun"]
            m = minimum["x"][0]
            c = minimum["x"][1]
            track_fits.append([minimum_chi2,m,c])
            #print(minimum)
        # pick the track with the best chisq value
        track = min(track_fits)
        m = track[1]
        c = track[2]
        return m, c
    else: return None

def event_reconstruction(event): # reconstruct each track
    skip_event = False
    #track1
    chamber_vector = [event.Dc1HitsVector_z,event.Dc1HitsVector_x]
    track_params = track_reconstruction(chamber_vector)
    if track_params!=None: m1, c1 = track_params
    else: skip_event=True
    #track2
    chamber_vector = [event.Dc2HitsVector_z,event.Dc2HitsVector_x]
    track_params = track_reconstruction(chamber_vector)
    if track_params!=None: m2, c2 = track_params
    else: skip_event=True

    if not skip_event: return m1,c1,m2,c2
    else: return False

def track_angles(m1,m2): # calculate the angle between the outgoing (incoming) track and chamber 1 (2)
    return math.atan(m1), math.atan(m2)

def Delta_x(m1,c1,m2,c2): # Calculate the change in x value as the particle passes through the magnetic field
    return abs((m1*mag_left_side+c1)-(m2*mag_right_side+c2))

def momentum(m1,c1,m2,c2): # Calculate the momentum of the particle
    theta_1,theta_2 = track_angles(m1,m2)
    delta_x = Delta_x(m1,c1,m2,c2)
    return (B*q*math.sqrt(L**2 + delta_x**2))/(1*math.sin((theta_1+theta_2)/2))

def plot(momenta):
    n, bins, patches = plt.hist(momenta, bins=n_bins)
    plt.xlabel("P (GeV/c)")
    plt.ylabel(f"Event Count / ({bin_width} GeV/c)")
    plt.savefig("task_{}.pdf".format(args.task))
    #plt.show()
    return n, bins

def FWHM(n, bins):
    maximum = max(n)
    candidates = []
    for i in range(len(n)):
        if n[i] > maximum/2: candidates.append(i)
    lower_edge = bins[candidates[0]]
    upper_edge = bins[candidates[-1]]
    #print(candidates, lower_edge, upper_edge)
    return bin_width*(candidates[-1]-candidates[0]-1)


inf = r.TFile.Open("./B5_{}.root".format(args.task))
tree = inf.Get("B5")
momenta = []
event_counter=0

for event in tree:
    print(100*event_counter/1000,"% ",end="\r")
    event_counter+=1
    if (len(event.Dc1HitsVector_x)==5 and len(event.Dc2HitsVector_x)==5) or not args.skip_multiple_hits:
        event_params=event_reconstruction(event)
        if not event_params: continue
        else:
            m1,c1,m2,c2 = event_params
            m = momentum(m1,c1,m2,c2)
            if abs(m)>plot_range[0] and abs(m)<plot_range[1]: momenta.append(abs(m))


n, bins = plot(momenta)
print("{:.3f} GeV".format(FWHM(n, bins)))
