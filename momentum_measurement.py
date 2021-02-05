import ROOT as r
import matplotlib.pyplot as plt
import math
import argparse
import scipy.optimize as opt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Applies preselection cuts", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--task', type=str, default ="a", help='Select the task this data corresponds to.')
parser.add_argument('--skip_multiple_hits', default=False, action="store_true", help='Skip fitting events with multiple hits in at least one drift chamber layer to make it run quicker.')
args = parser.parse_args()

tasks = {"a": {"B":0.5, "range": [100*0.9,100*1.1]}, "b_1": {"B": 0.25, "range": [100*0.9,100*1.1]}, "b_2": {"B":1, "range": [100*0.9,100*1.1]}, "c_1": {"B":0.5, "range": [50*0.9,50*1.1]}, "c_2": {"B":0.5, "range": [200*0.9,200*1.1]}}
n_bins = int(1000/25)


# define the detector properties
B = tasks[args.task]["B"]
plot_range = tasks[args.task]["range"]
#q = 1.60217662/(1e19)
q=3.*(10**8)/(10**9) # Since we want GeV/c
L=2. # width of the magnetic field
delta_z = 0.5 # separation between the layers in the drift chambers
mag_left_side = 3.0 # signed distance between start of first chamber and start of mag
mag_right_side = mag_left_side+L # signed distance between end of mag and start of second chamber

def track_reconstruction(chamber_vector,track_number): # reconstruct the track in a given chamber
    # dictionary to store hits in so I can handle events where there are multiple hits in a given drift chamber layer
    track_dict = {"0": [], "1": [], "2": [], "3": [], "4":[]}
    for i in range(len(chamber_vector[0])):
        z = chamber_vector[0][i]*delta_z + 6.*(track_number-1)
        #x = np.random.normal(chamber_vector[1][i]/1000, 1/1e4) # smear x to match the 100 micrometer uncertainty
        x = chamber_vector[1][i]/1000
        track_dict[str(int(chamber_vector[0][i]))].append([z, x])
    tracks = [[i,j,k,l,m] for i in track_dict["0"] for j in track_dict["1"] for k in track_dict["2"] for l in track_dict["3"] for m in track_dict["4"]] # determine all possible track candidates
    if tracks != []: #skip events where we don't get a hit in every layer, could perhaps incorporate them but it will take some thought
        track_fits = []
        for coords in tracks:
            def chisq(pars): # function to minimise in track fitting, define here to update data each time
                chi2 = 0
                for z,x in coords:
                    x_predicted = z*pars[0]+pars[1]
                    chi2+=((x-x_predicted)*1e4)**2 # 1e4 comes from the uncertainty in x
                return chi2
            # Determine some initial parameter values
            m = (coords[-1][1]-coords[0][1])/(coords[-1][0]-coords[0][0])
            #if track_number==1: c = coords[0][0]
            if track_number==2 or True: c = coords[0][1]-m*coords[0][0]
            # Fit the track
            minimum = opt.minimize(chisq, x0=[m,c])
            minimum_chi2 = minimum["fun"]
            m = minimum["x"][0]
            c = minimum["x"][1]
            track_fits.append([minimum_chi2,m,c,coords])
            #print(minimum)
        # pick the track with the best chisq value
        track = min(track_fits)
        m = track[1]
        c = track[2]
        coords = track[3]
        return m, c, coords
    else: return None

def event_reconstruction(event): # reconstruct each track
    skip_event = False
    #track1
    chamber_vector = [event.Dc1HitsVector_z,event.Dc1HitsVector_x]
    track_params = track_reconstruction(chamber_vector,1)
    if track_params!=None: m1, c1, coords1 = track_params
    else: skip_event=True
    #track2
    chamber_vector = [event.Dc2HitsVector_z,event.Dc2HitsVector_x]
    track_params = track_reconstruction(chamber_vector,2)
    if track_params!=None: m2, c2, coords2 = track_params
    else: skip_event=True
    if skip_event==False:
        #print(coords1,coords2)
        # reshape coords to plot later
        coords1_z = [i[0] for i in coords1]
        coords1_x = [i[1] for i in coords1]
        coords1 = [coords1_z,coords1_x]

        coords2_z = [i[0] for i in coords2]
        coords2_x = [i[1] for i in coords2]
        coords2 = [coords2_z,coords2_x]

    if not skip_event: return m1,c1,coords1,m2,c2,coords2
    else: return False

def track_angles(m1,m2): # calculate the angle between the outgoing (incoming) track and chamber 1 (2)
    return math.atan(abs(m1)), math.atan(abs(m2))

def Delta_x(m1,c1,m2,c2): # Calculate the change in x value as the particle passes through the magnetic field
    return abs((m1*mag_left_side+c1)-(m2*(mag_left_side+L)+c2))

def momentum(m1,c1,m2,c2): # Calculate the momentum of the particle
    theta_1,theta_2 = track_angles(m1,m2)
    delta_x = Delta_x(m1,c1,m2,c2)
    return (B*q*math.sqrt(L**2 + delta_x**2))/(2*math.sin(math.atan((m2-m1)/(1+m1*m2))/2))
    #return (B*q*math.sqrt(L**2 + delta_x**2))/(2*math.sin((theta_1+theta_2)/2))

def plot_momentum(momenta):
    plt.clf()
    n, bins, patches = plt.hist(momenta, bins=n_bins)
    bin_width = (bins[-1]-bins[0])/len(n)
    plt.xlabel("P (GeV/c)")
    plt.ylabel(f"Event Count / ({bin_width} GeV/c)")
    plt.savefig("task_{}.pdf".format(args.task))
    #plt.show()
    return n, bins

def plot_tracks(m1,c1,coords1,m2,c2,coords2):
    z = np.linspace(0,8)
    x1 = [m1*i+c1 for i in z]
    x2 = [m2*i+c2 for i in z]
    mag = np.linspace(-1/125,1/125)
    mag1 = [3. for i in mag]
    mag2 = [5. for i in mag]
    plt.plot(z,x1)
    plt.plot(z,x2)
    plt.plot(mag1,mag)
    plt.plot(mag2,mag)
    plt.scatter(coords1[0],coords1[1])
    plt.scatter(coords2[0],coords2[1])
    plt.legend(["Track 1", "Track 2", "Magnetic Region Left Side", "Magnetic Region Right Side"])
    plt.xlabel("z (m)")
    plt.ylabel("x (m)")
    plt.savefig("tracks.pdf")
    #plt.show()

inf = r.TFile.Open("./B5_{}.root".format(args.task))
tree = inf.Get("B5")
momenta = []
event_counter=0
first_allowed=0
for event in tree:
    print(100*event_counter/1000,"%",end="\r")
    event_counter+=1
    if (len(event.Dc1HitsVector_x)==5 and len(event.Dc2HitsVector_x)==5) or not args.skip_multiple_hits:
        event_params=event_reconstruction(event)
        if not event_params: continue
        else:
            m1,c1,coords1,m2,c2,coords2 = event_params
            z_intersect = (c2-c1)/(m1-m2)
            if (z_intersect<5. and z_intersect>3.) or True:
                if (z_intersect<5. and z_intersect>3.):
                    first_allowed+=1
                    if first_allowed==1:
                        plot_tracks(m1,c1,coords1,m2,c2,coords2)
                        #print(coords1,len(coords1))
                m = momentum(m1,c1,m2,c2)
                if abs(m)>plot_range[0] and abs(m)<plot_range[1]: momenta.append(abs(m))

def resolution(momenta):
    mean = 0
    for m in momenta: mean+=m
    mean=mean/len(momenta)
    #print(mean)

    var = 0
    for m in momenta: var+=(mean-m)**2
    var = var/len(momenta)
    sd = math.sqrt(var)
    #print(mean,var,sd)
    # assume we have a gaussian centered on the mean with our calculated standard deviation
    # we can approximate the FWHM as 2.2*sd
    return 2.2*sd

n, bins = plot_momentum(momenta)
#resolution(n,bins)
print("{:.3f} GeV".format(resolution(momenta)))
