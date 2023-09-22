import pygame
import random
import sys
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time
import numpy as np

#
#PRESS P for plot and pause
#PRESS D for debug
#

#experiments
A=False #only Fluid 
B=False #chains and walls
C=True #rings and walls



DELAY = 0
DEBUG = True
SAVE_SCREENSHOT = False

RED= (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
YELLOW = (255,255,0)
GRAY = (127,127,127)
WHITE = (255,255,255)
BLACK = (0,0,0)

GRID_SIZE = 15
SCREEN_SIZE=900
UNIT_SIZE=SCREEN_SIZE//GRID_SIZE


####
SIGMA = 1
RC = 1 #cutoff radius
GAMMA = 4.5
MASS = 1
DENSITY = 4
DT = 0.01

KS = 100 #spring constant
F_BODY=0
RS = 0.1       #0.1 #0.3 #equilibrium constant
MOVING_WALLS = True

if(C):
    F_BODY=0.3#0.3
    RS = 0.3       #0.1 #0.3 #equilibrium constant
    MOVING_WALLS = False



THERMOSTAT_ON = False
T=1 #temperature
IV=2*T
MAX_T=5

INIT_N=int(DENSITY*GRID_SIZE**2)
#INIT_N=900
INIT_N_CHAINS=42
INIT_N_RINGS=10

#ID=GRID_SIZE/math.sqrt(INIT_N*SIGMA) #initial distance between particles
#print(ID)
ID=1/DENSITY
#INIT_N=300

###
WALL_SPEED=5

#init Cell lists
CL={}
#N_CELLS=GRID_SIZE
CELL_SIZE = RC
N_CELLS = int(GRID_SIZE//CELL_SIZE)
for x in range(N_CELLS):
    for y in range(N_CELLS):
        CL[(x,y)]={}

print(CL)

class Cell:

    TYPE_A=0
    TYPE_B=1
    TYPE_F=2
    TYPE_W=3
    
    #maybe faster with dictionary
    ids=0
    allCells={}
    
    #SIZE= SCREEN_SIZE//GRID_SIZE
    SIZE= UNIT_SIZE//8

    #symmetric random forces
    random_forces_epsilon={} #key=(id1,id2) value=(epsilon_ij) to be applied to both particles

    def __init__(self, posX, posY, type=TYPE_F, velocityX=0, velocityY=0):
        Cell.ids+=1
        self.id=Cell.ids
        
        self.posX=posX%GRID_SIZE
        self.posY=posY%GRID_SIZE
        self.velocityX=velocityX
        self.velocityY=velocityY
        self.accelerationX=0
        self.accelerationY=0
        self.random_accelerationX=0
        self.random_accelerationY=0
        self.forceX=0
        self.forceY=0
        self.random_forceX=0
        self.random_forceY=0
        self.potential=0
        self.kinetic=0

        self.f_body=0

        self.type=type
        if(self.type==Cell.TYPE_A):
            self.color=RED
        elif(self.type==Cell.TYPE_B):
            self.color=GREEN
        elif(self.type==Cell.TYPE_F):
            self.color=BLUE #YELLOW
        elif(self.type==Cell.TYPE_W):
            self.color=GRAY
        
        #print(self.posX,self.posY)

        Cell.allCells[self.id]=self
        CL[(self.posX//CELL_SIZE,self.posY//CELL_SIZE)][self.id]=self

        self.bonds=[]

    def coef_a(self,other):
        i=self.type
        j=other.type

        aij = [
        [50, 25, 25, 200],
        [25, 1, 300, 200],
        [25, 300, 25, 200],
        [200, 200, 200, 0]
        ]
        return aij[i][j]

    def calculate_forces(self):
        self.forceX=0
        self.forceY=self.f_body #constant body force
        self.random_forceX=0
        self.random_forceY=0
        self.potential=0
        self.distances=[]
        for x in range(-1,2):
            for y in range(-1,2):
                for _,cell in CL[((self.posX//CELL_SIZE)+x)%N_CELLS,((self.posY//CELL_SIZE)+y)%N_CELLS].items():
                    if(cell.id!=self.id):
                        dx = self.posX-cell.posX
                        dy = self.posY-cell.posY
                        
                        #periodic boundary conditions
                        if(dx>GRID_SIZE/2):
                            dx=dx-GRID_SIZE
                        elif(dx<=-GRID_SIZE/2):
                            dx=dx+GRID_SIZE
                        if(dy>GRID_SIZE/2):
                            dy=dy-GRID_SIZE
                        elif(dy<=-GRID_SIZE/2):
                            dy=dy+GRID_SIZE
                        
                        r = math.sqrt(dx**2+dy**2)
                        
                        #unit vector
                        dx_hat = dx/r
                        dy_hat = dy/r

                        vx=self.velocityX-cell.velocityX
                        vy=self.velocityY-cell.velocityY

                        
                        #print(f'id1={self.id} posX={self.posX} posY={self.posY} id2={cell.id} posX={cell.posX} posY={cell.posY} r={r} dx={dx} dy={dy} ')
                        if(r<=RC and r>0):
                            self.distances.append(r)
                            #F = (48/(r**2))*((SIGMA/r)**12 - 0.5*(SIGMA/r)**6) 
                            #self.forceX+=F*dx/r
                            #self.forceY+=F*dy/r
                            
                            #conservative force
                            fcx=self.coef_a(cell)*(1-(r/RC))*dx_hat
                            fcy=self.coef_a(cell)*(1-(r/RC))*dy_hat
                            self.forceX+=fcx
                            self.forceY+=fcy

                            #weighting functions
                            w_r=1.0-r/RC
                            w_d=w_r**2

                            #dissipative force
                            fdx = -GAMMA*w_d*dx_hat*(dx_hat*vx+dy_hat*vy)
                            fdy = -GAMMA*w_d*dy_hat*(dx_hat*vx+dy_hat*vy)
                            self.forceX+= fdx
                            self.forceY+=fdy

                            #random force needs to be symmetric
                            epsilon_ij=0
                            
                            if((cell.id,self.id) in Cell.random_forces_epsilon):
                                epsilon_ij=Cell.random_forces_epsilon.pop((cell.id,self.id))
                                #print(self.id,cell.id,epsilon_ij,Cell.random_forces_epsilon)
                            else:
                                epsilon_ij=random.gauss(0,1)
                                Cell.random_forces_epsilon[(self.id,cell.id)]=epsilon_ij
                                #print(self.id,cell.id,epsilon_ij,Cell.random_forces_epsilon)
                            
                            frx= (SIGMA*w_r*epsilon_ij*dx_hat)/math.sqrt(DT)
                            fry= (SIGMA*w_r*epsilon_ij*dy_hat)/math.sqrt(DT)
                            #self.random_forceX+=frx
                            #self.random_forceY+=fry
                            self.forceX+=frx
                            self.forceY+=fry

                        #BONDING FORCE
                        #if(self.type==Cell.TYPE_A or self.type==Cell.TYPE_B and cell.type==Cell.TYPE_A or cell.type==Cell.TYPE_B):
                        if(cell.id in self.bonds):
                            #print(r)
                            #print(dx_hat,dy_hat)
                            fbx=KS*(1-r/RS)*dx_hat
                            fby=KS*(1-r/RS)*dy_hat
                            self.forceX+=fbx
                            self.forceY+=fby
                        
                        

                           
                            #potential energy
                            #self.potential+=4*((SIGMA/r)**12 - (SIGMA/r)**6) - 4*((SIGMA/RC)**12 - (SIGMA/RC)**6)
            
    def update_position(self):
        #-0.00000000000000001%15 = 15 ????
        #new_posX = (self.posX + self.velocityX*DT + 0.5*self.accelerationX*DT**2 + 0.5*self.random_accelerationX*DT )%GRID_SIZE%GRID_SIZE
        #new_posY = (self.posY + self.velocityY*DT + 0.5*self.accelerationY*DT**2 + 0.5*self.random_accelerationY*DT )%GRID_SIZE%GRID_SIZE
        new_posX = (self.posX + self.velocityX*DT + 0.5*self.accelerationX*DT**2)%GRID_SIZE%GRID_SIZE
        new_posY = (self.posY + self.velocityY*DT + 0.5*self.accelerationY*DT**2)%GRID_SIZE%GRID_SIZE
        
        #print(self.posX,new_posX)
        #print(self.posY,new_posY)
        #print((self.posY + self.velocityY*DT + 0.5*self.accelerationY*DT**2 + 0.5*self.random_accelerationY*DT )) 
        #update cell list
        if((self.posX//CELL_SIZE,self.posY//CELL_SIZE) != (new_posX//CELL_SIZE,new_posY//CELL_SIZE)):
            CL[(self.posX//CELL_SIZE,self.posY//CELL_SIZE)].pop(self.id)
            CL[(new_posX//CELL_SIZE,new_posY//CELL_SIZE)][self.id]=self

        self.posX=new_posX
        self.posY=new_posY
    
    def update_velocity_accelleration(self):
        if(self.type==Cell.TYPE_W):
            #self.velocityX=self.velocityX_wall
            #self.velocityY=self.velocityY_wall
            self.accelerationX=0
            self.accelerationY=0
            self.random_accelerationX=0
            self.random_accelerationY=0
            return 
        
        #self.calculate_forces() 

        new_accelerationX = self.forceX/MASS
        new_accelerationY = self.forceY/MASS
        #new_random_accelerationX = self.random_forceX/MASS
        #new_random_accelerationY = self.random_forceY/MASS

        self.velocityX+=(self.accelerationX+new_accelerationX)*DT*0.5# + new_random_accelerationX* math.sqrt(DT)*0.5
        self.velocityY+=(self.accelerationY+new_accelerationY)*DT*0.5# + new_random_accelerationY* math.sqrt(DT)*0.5
        #self.velocityX+=new_accelerationX*DT
        #self.velocityY+=new_accelerationY*DT
        self.accelerationX=new_accelerationX
        self.accelerationY=new_accelerationY
        #self.random_accelerationX=new_random_accelerationX
        #self.random_accelerationY=new_random_accelerationY

        v=math.sqrt(self.velocityX**2+self.velocityY**2)
        #print(f'id={self.id} vx={self.velocityX} vy={self.velocityY} v={v}')
        #kinetic energy
        self.kinetic=0.5*MASS*v**2

        #v*=255/MAX_T
        #if(v>255):
        #    v=255
        #self.color=(v,0,255-v)

    def total_velocity():
        tot_v=0
        for _,cell in Cell.allCells.items():
            tot_v+=math.sqrt(cell.velocityX**2+cell.velocityY**2)
        return tot_v
    
    def total_temperaure():
        #return Cell.total_velocity()/INIT_N
        return Cell.total_kinetic()/(3/2*INIT_N)

    def total_momentum():
        tot_mx=0
        tot_my=0
        for _,cell in Cell.allCells.items():
            tot_mx+=cell.velocityX*MASS
            tot_my+=cell.velocityY*MASS
        tot_m=math.sqrt(tot_mx**2+tot_my**2)
        #print(tot_mx,tot_my)
        return tot_m
    
    def total_potential():
        tot_pot=0
        for _,cell in Cell.allCells.items():
            tot_pot+=cell.potential
        return tot_pot
    
    def total_kinetic():
        tot_kin=0
        for _,cell in Cell.allCells.items():
            tot_kin+=cell.kinetic
        return tot_kin
    
    def total_energy():
        return Cell.total_potential()+Cell.total_kinetic()
    """
    def average_radial_distribution_function():
        BINS_SIZE=0.05
        BINS= np.arange(BINS_SIZE,RC,BINS_SIZE)
        N_BINS= BINS.size
        RADIAL_DISTRIBUTION=[0]*N_BINS
        for _,cell in Cell.allCells.items():
            for r in cell.distances:
                for i,bin in enumerate(BINS):
                    if r>=bin and r<bin+BINS_SIZE:
                        RADIAL_DISTRIBUTION[i]+=1 
        
        AVG_DENSITY=INIT_N/(GRID_SIZE**2)
        for i,bin in enumerate(BINS):
            #normalized average radial distribution function
            #RADIAL_DISTRIBUTION[i]/=INIT_N
            RADIAL_DISTRIBUTION[i]/=AVG_DENSITY*2*math.pi*bin*BINS_SIZE*INIT_N
            
        return  BINS,RADIAL_DISTRIBUTION
    """
    
    def berendsen_thermostat():

        current_temperature = Cell.total_temperaure()
        # Calculate scaling factor
        #DT/tau=0.0025
        scaling_factor = np.sqrt(1 + (0.0025) * ((T / current_temperature) - 1))

        # Scale velocities
        for _,cell in Cell.allCells.items():
            cell.velocityX *= scaling_factor
            cell.velocityY *= scaling_factor

    def update_desired_temperature(new_T):
        global T
        T=new_T

#beautiful code
def spawn_ring_molecule(center):
    cell0=Cell(random.uniform(center[0]-ID,center[0]+ID),random.uniform(center[1]-ID,center[1]+ID),Cell.TYPE_A)
    cell1=Cell(random.uniform(center[0]-ID,center[0]+ID),random.uniform(center[1]-ID,center[1]+ID),Cell.TYPE_A)
    cell1.bonds.append(cell0.id)
    cell0.bonds.append(cell1.id)
    cell2=Cell(random.uniform(center[0]-ID,center[0]+ID),random.uniform(center[1]-ID,center[1]+ID),Cell.TYPE_A)
    cell2.bonds.append(cell1.id)
    cell1.bonds.append(cell2.id)
    cell3=Cell(random.uniform(center[0]-ID,center[0]+ID),random.uniform(center[1]-ID,center[1]+ID),Cell.TYPE_A)
    cell3.bonds.append(cell2.id)
    cell2.bonds.append(cell3.id)
    cell4=Cell(random.uniform(center[0]-ID,center[0]+ID),random.uniform(center[1]-ID,center[1]+ID),Cell.TYPE_A)
    cell4.bonds.append(cell3.id)
    cell3.bonds.append(cell4.id)
    cell5=Cell(random.uniform(center[0]-ID,center[0]+ID),random.uniform(center[1]-ID,center[1]+ID),Cell.TYPE_A)
    cell5.bonds.append(cell4.id)
    cell4.bonds.append(cell5.id)
    cell6=Cell(random.uniform(center[0]-ID,center[0]+ID),random.uniform(center[1]-ID,center[1]+ID),Cell.TYPE_A)
    cell6.bonds.append(cell5.id)
    cell5.bonds.append(cell6.id)
    cell7=Cell(random.uniform(center[0]-ID,center[0]+ID),random.uniform(center[1]-ID,center[1]+ID),Cell.TYPE_A)
    cell7.bonds.append(cell6.id)
    cell6.bonds.append(cell7.id)
    cell8=Cell(random.uniform(center[0]-ID,center[0]+ID),random.uniform(center[1]-ID,center[1]+ID),Cell.TYPE_A)
    cell8.bonds.append(cell7.id)
    cell7.bonds.append(cell8.id)
    cell0.bonds.append(cell8.id)
    cell8.bonds.append(cell0.id)


def spawn_chain_molecule(center):
    #A−A−B−B−B−B−B
    cell0=Cell(center[0],center[1],Cell.TYPE_A)
    cell1=Cell(center[0],center[1]+ID,Cell.TYPE_A)
    cell1.bonds.append(cell0.id)
    cell0.bonds.append(cell1.id)
    cell2=Cell(center[0],center[1]+2*ID,Cell.TYPE_B)
    cell2.bonds.append(cell1.id)
    cell1.bonds.append(cell2.id)
    cell3=Cell(center[0],center[1]+3*ID,Cell.TYPE_B)
    cell3.bonds.append(cell2.id)
    cell2.bonds.append(cell3.id)
    cell4=Cell(center[0],center[1]+4*ID,Cell.TYPE_B)
    cell4.bonds.append(cell3.id)
    cell3.bonds.append(cell4.id)
    cell5=Cell(center[0],center[1]+5*ID,Cell.TYPE_B)
    cell5.bonds.append(cell4.id)
    cell4.bonds.append(cell5.id)
    cell6=Cell(center[0],center[1]+6*ID,Cell.TYPE_B)
    cell6.bonds.append(cell5.id)
    cell5.bonds.append(cell6.id)
            
###########
pygame.init()
pygame.display.set_caption('Press P or D')
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE), 0)
#WHITE = (255,255,255)


lattice = [(x,y) for x in np.arange(0,GRID_SIZE,ID) for y in np.arange(0,GRID_SIZE,ID)]
#for x,y in lattice:
#    Cell(x,y)

print("Number of Particles:", INIT_N)
#print(lattice)
initial_positions = random.sample(lattice, INIT_N)
#for x,y in initial_positions:
#    Cell(x,y,)

if(A):
    for x,y in initial_positions:
        Cell(x,y,Cell.TYPE_F)



#SPAWN SEQUENCE OF A MOLECULES 
#center of grid

#COUETTE FLOW
if(B):
    for i,center in enumerate(initial_positions):
        #center = (GRID_SIZE//2,GRID_SIZE//2)
        #spawn_ring_molecule(center)
        if(i<=INIT_N_CHAINS):
            spawn_chain_molecule(center)
        elif(i>=INIT_N_CHAINS*7):
            Cell(center[0],center[1],Cell.TYPE_F)

#POUISSELLE FLOW
if(C):
    for i,center in enumerate(initial_positions):
        #center = (GRID_SIZE//2,GRID_SIZE//2)
        #spawn_ring_molecule(center)
        if(i<INIT_N_RINGS):
            spawn_ring_molecule(center)
        elif(i>INIT_N_RINGS*9):
            Cell(center[0],center[1],Cell.TYPE_F)
    
      
#initial velocities, momentum=0
"""
meanx=0
meany=0
for _,cell in Cell.allCells.items():
    #cell.velocityX=random.gauss(0,IV)
    #cell.velocityY=random.gauss(0,IV)
    cell.velocityX=random.uniform(-IV,IV)
    cell.velocityY=random.uniform(-IV,IV)
    meanx+=cell.velocityX
    meany+=cell.velocityY

meanx/=INIT_N
meany/=INIT_N

for _,cell in Cell.allCells.items():
    cell.velocityX-=meanx
    cell.velocityY-=meany
"""

for _,cell in Cell.allCells.items():
    cell.velocityX=0
    cell.velocityY=0
#Cell.allCells[1].velocityX=20
#Cell.allCells[1].velocityY=20

#INITIALIZE WALLS on simulation borders with width=1
#SAME DIRECTION WALLS
"""
for _,cell in Cell.allCells.items():
    if(cell.posX <RC or cell.posX>GRID_SIZE-RC ):
        cell.velocityX=0
        cell.velocityY=100
        cell.type=Cell.TYPE_W
        cell.color=GRAY
"""
"""
#OPPOSITE DIRECTION WALLS COOL BUT WRONG
for _,cell in Cell.allCells.items():
    
    if(cell.posY<RC):
        cell.velocityX=0
        cell.velocityY=WALL_SPEED
        cell.type=Cell.TYPE_W
        cell.color=GRAY

    if(cell.posY>GRID_SIZE-RC ):
        cell.velocityX=0
        cell.velocityY=-WALL_SPEED
        cell.type=Cell.TYPE_W
        cell.color=GRAY
"""


#OPPOSITE DIRECTION WALLS
if(B or C):
    if(not MOVING_WALLS):
        WALL_SPEED=0

    for _,cell in Cell.allCells.items():

        if(cell.type==Cell.TYPE_F):
            if(cell.posX<RC):
               
                cell.velocityX=0
                cell.velocityY=WALL_SPEED
                cell.type=Cell.TYPE_W
                cell.color=GRAY

            elif(cell.posX>GRID_SIZE-RC ):
            
                cell.velocityX=0
                cell.velocityY=-WALL_SPEED
                cell.type=Cell.TYPE_W
                cell.color=GRAY

if(C):
    #constant body force
    for _,cell in Cell.allCells.items():
        if(cell.type!=Cell.TYPE_W):
            cell.f_body=F_BODY
        


n_iterations=0

#plt.ion()
#fig=plt.figure()
total_t=[]
total_m=[]
total_p=[]
total_k=[]
total_e=[]

expected_t=[]


#print("momentum: ", total_m[-1])
end=False
start_time= time.time()
tot_time=0
while(not end):
    iter_time = time.time()
    n_iterations+=1
    
    #draw stuff
    draw_time = time.time()
    screen.fill(0)
    
    for _,cell in Cell.allCells.items():
        #pygame.draw.rect(screen,cell.color,(cell.posX*cell.SIZE, cell.posY*Cell.SIZE, Cell.SIZE, Cell.SIZE))
        pygame.draw.circle(screen,cell.color,(cell.posX*UNIT_SIZE, cell.posY*UNIT_SIZE), Cell.SIZE//2, width=0)
       
    #DRAW CL
    if(DEBUG):
        for x,y in CL.keys():
            pygame.draw.rect(screen, GRAY, (x*CELL_SIZE*UNIT_SIZE, y*CELL_SIZE*UNIT_SIZE, CELL_SIZE*UNIT_SIZE, CELL_SIZE*UNIT_SIZE), width=1)
            
    ##
    for _,cell in Cell.allCells.items():
        cell.update_position()
    for _,cell in Cell.allCells.items():
        cell.calculate_forces()
    for _,cell in Cell.allCells.items():
        cell.update_velocity_accelleration()

    #plot stuff
    total_t.append(Cell.total_temperaure())
    total_m.append(Cell.total_momentum())
    total_p.append(Cell.total_potential())
    total_k.append(Cell.total_kinetic())
    total_e.append(Cell.total_energy())
    #boltzman constant
    
    expected_t.append((SIGMA**2)/(2*GAMMA))
    print("momentum: ", total_m[-1],end='     \r')

    if(THERMOSTAT_ON):
        Cell.berendsen_thermostat()
        
    #events
    for event in pygame.event.get():
        if(event.type == pygame.QUIT):
            pygame.display.quit()
            pygame.quit()
            end=True
            #sys.exit()
        if(event.type == pygame.KEYDOWN):
            if(event.key == pygame.K_UP):           
                DELAY-=10                
            if(event.key == pygame.K_DOWN):            
                DELAY+=10
            if(event.key == pygame.K_SPACE):
                 pygame.image.save(screen, f"-Iteration={n_iterations}-N={INIT_N}-dt{DT}.jpg") 
            if(event.key == pygame.K_d):
                DEBUG=not DEBUG
            if(event.key == pygame.K_p):
                #subplots
                #fig,ax = plt.subplots(4,1,figsize=(8, 10))
                fig,ax = plt.subplots(3,1,figsize=(8, 10))
                
                ax[0].plot(range(n_iterations),total_m,label='Total momentum')
                #ax[1].plot(range(n_iterations),total_e,label='Total energy')
                #ax[1].plot(range(n_iterations),total_p,label='Total potential')
                ax[1].plot(range(n_iterations),total_k,label='Total kinetic')
                #bins,avg_radial_distribution=Cell.average_radial_distribution_function()
                #ax[2].plot(bins,avg_radial_distribution,label='Radial distribution function')
                ax[2].plot(range(n_iterations),total_t,label='Total temperature')
                ax[2].plot(range(n_iterations),expected_t,label='Expected temperature')
                if(THERMOSTAT_ON):
                    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
                    slider = Slider(slider_ax, 'Desired Temp', 0, MAX_T, valinit=T, valstep=0.1)
                    slider.on_changed(Cell.update_desired_temperature)
                ax[0].legend()
                ax[1].legend()
                ax[2].legend()
                #ax[3].legend()
                plt.show()
                
        if(event.type == pygame.MOUSEBUTTONDOWN):
            mousePosX,mousePosY= pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]
            #Cell(mousePosX/UNIT_SIZE, mousePosY/UNIT_SIZE, Cell.TYPE_F)
            #for _ in range(10):
                #Cell((mousePosX//Cell.SIZE)+EATING_DISTANCE*2, mousePosY//Cell.SIZE, Cell.RABBIT)
            
            #move walls with mouse
            
                   
                   
                    

    iter_time=time.time()-iter_time
    tot_time= time.time()-start_time
    #print(f'Iteration:{n_iterations} N_WOLVES={len(Cell.wolves)} N_RABBITS={len(Cell.rabbits)} -TIMING: sim={tot_time/60:.2f}m iter={iter_time:.3f}s : draw={draw_time:.3f}s move={time_move:.3f}s eat={time_eat:.3f}s repr={time_replicate:.3f}s surv={time_survive:.3f}s      ', end='\r')
    
    if(not end):
        pygame.display.update()     
        pygame.time.delay(DELAY)

    







