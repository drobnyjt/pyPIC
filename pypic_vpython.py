import numpy as np
import vpython as vp
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
import pickle

file = Path("particle_output.p")
if file.is_file():
    with open('particle_output.p','rb') as handle:
        data = pickle.load(handle)
    #end with
else:
    data = np.genfromtxt('particle_output.txt',delimiter=",")
    with open('particle_output.p','wb') as handle:
        pickle.dump(data, handle)
    #end with
#end if

x = data[:,0]
v = data[:,1]

x/=np.max(x)/2.0
v/=np.max(v)

def distribution(x,num_particles=5000):
    start = 0
    while start < len(x):
        yield x[start:start + num_particles]
        start += num_particles
    #end while
#end def distribution

x_wrapped = itertools.cycle(distribution(x))
v_wrapped = itertools.cycle(distribution(v))

x0 = next(x_wrapped)
v0 = next(v_wrapped)

canvas = vp.canvas(width=600,height=400)
canvas.center = vp.vector(1.0,0.0,0.0)

colors = [vp.color.hsv_to_rgb(vp.vector(abs(v_),1.0,1.0)) for v_ in v0]
particles = [vp.sphere(pos=vp.vector(x_,v_,0),radius=0.01) for x_,v_ in zip(x0,v0)]

for index,particle in enumerate(particles):
    particle.color = colors[index]
#end for

time = 0
while time < 10000:
    time += 1
    vp.rate(60)
    x_dist = next(x_wrapped)
    v_dist = next(v_wrapped)
    for index,particle in enumerate(particles):
        shift = 2.0*(index%2) - 1.0
        particle.pos = vp.vector(x_dist[index]+shift, v_dist[index], 0)
    #end for
    if time%6 == 0 and time < 6*20: canvas.capture(str(time)+'.png')
#end while
