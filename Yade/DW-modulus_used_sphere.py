#this is final program

from __future__ import print_function
from yade import pack

O.periodic=True

PI=3.141592653
flag=0
xl=1
Readname='%dSphere.pac'%xl

fp=open(Readname,'r')
line=fp.readlines()
fp.close()


box_l1=float(line[1])/100.0
box_l2=float(line[1])/100.0
box_l3=float(line[1])/100.0

box_l=(box_l1+box_l2+box_l3)/3.0
O.cell.hSize=Matrix3(box_l1,0,0, 0,box_l2,0, 0,0,box_l3)


particle_num=int(line[2])
print("#############################\nparticle num=",particle_num)
print("#############################\n")



id_Mat=O.materials.append(FrictMat(young=2.2e9,poisson=0.4,density=1000,frictionAngle=0))
Mat=O.materials[id_Mat]



volumeParticles=0.0
for i in range(particle_num):
	line[i+3] = line[i+3].strip()
	listFromLine = line[i+3].split('\t')	
	id_particle=O.bodies.append(sphere(((float(listFromLine[1]))/100.0,(float(listFromLine[2]))/100.0,(float(listFromLine[3]))/100.0),radius=(float(listFromLine[0]))/100.0,material=Mat))
	volumeParticles+=O.bodies[id_particle].state.mass/O.bodies[id_particle].material.density


pd_initial=volumeParticles/O.cell.volume
l_initial=pow(O.cell.volume,1.0/3.0)
pd_final=pd_initial+0.004
l_final=pow(volumeParticles/pd_final,1.0/3.0)
sigmaIso=-(l_initial-l_final)/l_initial
print("#########################")
print(sigmaIso)
print("#########################")


Filename='%d_modulus_result.txt'%xl
fp=open(Filename,'w')#save the simulation parameters


compressISO=PeriTriaxController(
		label='triax1',
		goal=(2.0*sigmaIso,2.0*sigmaIso,2.0*sigmaIso),
		stressMask=0,
		dynCell=True,
		maxStrainRate=(1e-3,1e-3,1e-3),
		maxUnbalanced=.1,
		#relStressTol=0.01,
		doneHook='compressFinished()',
		dead=False,
	)


expandISO=PeriTriaxController(
		label='triax2',
		goal=(-50,-50,-50),
		stressMask=7,
		dynCell=True,
		maxStrainRate=(1e-3,1e-3,1e-3),
		maxUnbalanced=0.1,
    relStressTol=1.0,
		doneHook='expandFinished()',
		dead=True,
		)

O.engines=[
	ForceResetter(),
	InsertionSortCollider([Bo1_Sphere_Aabb()]),
	InteractionLoop(
		[Ig2_Sphere_Sphere_ScGeom()],
		[Ip2_FrictMat_FrictMat_FrictPhys()],
		[Law2_ScGeom_FrictPhys_CundallStrack()]
	),
	compressISO,
	expandISO,
	NewtonIntegrator(damping=0.4),
	PyRunner(command='addPlotData()',iterPeriod=200),
]

O.dt=.5*PWaveTimeStep()


def expandFinished():
	global flag
	global box_l1
	global box_l2
	global box_l3
	sx=-getStress().trace()/3.0
	print("stress=",sx)
	print('########################################'),
	print('finishedExpand\n'),
	O.engines[-3].dead=True
	O.engines[-4].dead=False
	box_l1=O.cell.hSize[0][0]
	box_l2=O.cell.hSize[1][1]
	box_l3=O.cell.hSize[2][2]
	flag=2
	e_volume=(box_l1*box_l2*box_l3-O.cell.hSize[0][0]*O.cell.hSize[1][1]*O.cell.hSize[2][2])/(box_l1*box_l2*box_l3)
	sx=-getStress().trace()/3.0
	fp.write(str(e_volume)+"\t"+str(sx)+"\n")

def compressFinished():
	#Filename='%dresult.txt'%aspect_ratio
	global flag
	if flag==0:
		print('########################################'),
		print('finishedCompress(first step)\n'),
		#O.pause()
		#triax1.dead=True,
		#triax2.dead=False,
		O.engines[-4].dead=True
		O.engines[-3].dead=False
		flag=1
		#O.pause()
	else:
		print('########################################'),
		print('finishedCompress(second step)\n'),
		#Filename="modulus.txt",
		#plot.saveDataTxt("modulus-2000-particles-wu-pre-com.txt",vars = None),
		fp.close()
		O.pause()


def addPlotData():
	global flag
	global pd_initial
	global box_l1
	global box_l2
	global box_l3
	if flag==0:
		pd=(volumeParticles)/O.cell.volume
		print("flag=0\n",(pd-pd_initial)*10000)
		#fp.write("flag=0\n%f\n"%(pd))
	if flag==1:
		sx=getStress().trace()/3.0
		print("stress=",sx)
		#fp.write("stress=%f\n"%(sx))
	if flag==2:
		pd=(volumeParticles)/O.cell.volume
		print("flag=2\n",(pd-pd_initial)*10000)
		e_volume=(box_l1*box_l2*box_l3-O.cell.hSize[0][0]*O.cell.hSize[1][1]*O.cell.hSize[2][2])/(box_l1*box_l2*box_l3)
		sx=-getStress().trace()/3.0
		fp.write(str(e_volume)+"\t"+str(sx)+"\n")
		#fp.write("flag=0\n%f\n"%(pd))


O.saveTmp()
O.run()

