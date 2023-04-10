from __future__ import print_function
from yade import pack

PI = 3.14159265358979324

# Read file
fp = open('final.dat', 'r')
line = fp.readlines()
fp.close()

particle_num = int(line[2])
print("#############################\nparticle num=", particle_num)
print("#############################\n")

radius = float(line[3])/2.0

lattice = (line[4].strip()).split(' ')

box_l1 = float(lattice[0])/10.
box_l2 = float(lattice[4])/10.
box_l3 = float(lattice[8])/10.

box_l = (box_l1+box_l2+box_l3)/3.0

O.periodic = True

O.cell.hSize = Matrix3(box_l1, 0., 0.,
                       0., box_l2, 0.,
                       0., 0., box_l3)


id_Mat = O.materials.append(FrictMat(young=2.2e9, poisson=0.4,
                                     density=1000, frictionAngle=0))
Mat = O.materials[id_Mat]


volumeParticles = 0.0
for i in range(particle_num):
  center = (line[i+6].strip()).split(' ')
  particle = sphere((float(center[0])/10.,float(center[1])/10.,float(center[2])/10.),
                    radius=radius/10.,
                    material = Mat)
  id = O.bodies.append(particle)
  volumeParticles += O.bodies[id].state.mass/O.bodies[id].material.density


flag = 0
pd_initial = volumeParticles/O.cell.volume
l_initial = pow(O.cell.volume, 1.0/3.0)

pd_final = pd_initial+0.004
l_final = pow(volumeParticles/pd_final, 1.0/3.0)
sigmaIso = -(l_initial-l_final)/l_initial
print("#########################")
print(sigmaIso)
print("#########################")

# save the simulation parameters
fp = open('modulus_result.txt', 'w')

compressISO = PeriTriaxController(
		label='triax',
		goal=(2.0*sigmaIso,2.0*sigmaIso,2.0*sigmaIso),
		stressMask=0,
		dynCell=True,
		maxStrainRate=(1e-3,1e-3,1e-3),
		maxUnbalanced=.1,
		#relStressTol=0.01,
		doneHook='compressFinished()',
		dead=False,
)


O.engines = [
	  ForceResetter(),
	  InsertionSortCollider([Bo1_Sphere_Aabb()]),
	  InteractionLoop(
		  [Ig2_Sphere_Sphere_ScGeom()],
		  [Ip2_FrictMat_FrictMat_FrictPhys()],
		  [Law2_ScGeom_FrictPhys_CundallStrack()]
	  ),
	  compressISO,
	  NewtonIntegrator(damping=.4),
	  PyRunner(command='addPlotData()',iterPeriod=200),
]

O.dt = .5*PWaveTimeStep()


def compressFinished():
	#Filename='%dresult.txt'%aspect_ratio
	print('########################################')
	print('FinishedCompress\n')
	#O.exitNoBacktrace()
	#O.pause()
	import sys
	sys.exit(0)
 

def addPlotData():
	global pd_initial
	global box_l1
	global box_l2
	global box_l3

	pd = (volumeParticles)/O.cell.volume
	print("Running\n",(pd-pd_initial)*10000)
	e_volume = (box_l1*box_l2*box_l3-O.cell.hSize[0][0]*O.cell.hSize[1][1]*O.cell.hSize[2][2])/(box_l1*box_l2*box_l3)
	sx = -getStress().trace()/3.0
	fp.write(str(e_volume)+"\t"+str(sx)+"\n")


O.saveTmp()
O.run()


