import ROOT as r
import matplotlib.pyplot as plt
import math
import argparse

parser = argparse.ArgumentParser(description="Applies preselection cuts", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--task', type=str, default ="a", help='Select the task this data corresponds to.')
args = parser.parse_args()

tasks = {"a": {"B":0.5, "range": [0,1e10]}, "b_1": {"B": 0.25, "range": [0,1e10]}, "b_2": {"B":1, "range": [0,2e10]}, "c_1": {"B":0.5, "range": [0,1e10]}, "c_2": {"B":0.5, "range": [0,1e10]}}
n_bins = 150
bin_width = tasks[args.task]["range"][1]/n_bins


# define the detector properties
B = tasks[args.task]["B"]
plot_range = tasks[args.task]["range"]
#q = 1.60217662/(1e19)
q=3*(10**8) # Since we want GeV
L=1 # separation between the drift chambers
delta_z = 0.5 # separation between the layers in the drift chambers

inf = r.TFile.Open("B5_{}.root".format(args.task))
tree = inf.Get("B5")
#tree.Print()

def track_reconstruction():
    event_counter=0
    hit_coords = []
    for event in tree:
        chamber_vector = [[event.Dc1HitsVector_x,event.Dc1HitsVector_z],
                          [event.Dc2HitsVector_x,event.Dc2HitsVector_z]]
        if len(chamber_vector[0][0])==5 and len(chamber_vector[1][0])==5:
            hit_coords.append([])
            for chamber in [1,2]:
                hits_vector = chamber_vector[chamber-1]
                for i in range(5):
                    hit_coords[event_counter].append([hits_vector[0][i]/1000, hits_vector[1][i]+(chamber-1)*5.])
            event_counter+=1
    return hit_coords

def track_chamber_angles(hit_coords): # calculate the angle between the outgoing (incoming) track and chamber 1 (2)
    #print(hit_coords)
    x0,z0 = hit_coords[0]
    x1,z1 = hit_coords[4]
    x2,z2 = hit_coords[5]
    x3,z3 = hit_coords[-1]
    return math.atan(x1-x0/(abs(z1-z0)*delta_z)), math.atan(x3-x2/(abs(z3-z2)*delta_z)), abs(x2-x1) #theta_1, theta_2, delta_x

def momentum(hit_coords):
    theta_1,theta_2,delta_x = track_chamber_angles(hit_coords)
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
    print(candidates, lower_edge, upper_edge)
    return 2*bin_width*(candidates[-1]-candidates[0]-1)/(1e9)

tracks = track_reconstruction()
momenta = []
for i in range(len(tracks)):
    m = abs(momentum(tracks[i]))
    if m>plot_range[0] and m<plot_range[1]: momenta.append(m)
n, bins = plot(momenta)
print("{:.3f} GeV".format(FWHM(n, bins)))
