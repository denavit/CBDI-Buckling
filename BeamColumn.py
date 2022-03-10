import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
from math import pi,sqrt
from CBDI import PcrCBDI

# Input parameters
# Units: kips, in, ksi

cross_section_type = 'RC'

if cross_section_type == 'RC':
    # Parameters for RC section
    D = 36
    xp = 3 # Distance from center of bar to edge of cross section
    L_over_D = 20
    rho_s = 0.02
    n_bar = 8
    fc = 4
    Fy = 60
    Es = 29000
    L = L_over_D*D
    Ag = pi/4*D**2
    As = rho_s*Ag
    Ab = As/n_bar
    Po = 0.85*fc*(Ag-As) + Fy*As
    Ec = 57*sqrt(1000*fc)
    bar_angles = np.linspace(0,2*pi,n_bar,endpoint=False)
    bay_y = (0.5*D-xp)*np.sin(bar_angles)
    Isr = sum(Ab*bay_y**2)
    EIgross = Ec*(1/64)*pi*D**4 + Es*Isr
elif cross_section_type == 'WF':
    # Parameters for WF section (W14x159)
    d = 15.0
    bf = 15.6
    tf = 1.19
    tw = 0.745
    A = 2*bf*tf + (d-2*tf)*tw
    Fy = 60
    E = 29000
    Po = Fy*A
    L_over_d = 30
    L = L_over_d*d
    I = (1/12)*(bf*d**3 - (bf-tw)*(d-2*tf)**3)
    EIgross = E*I
elif cross_section_type == 'Elastic':
    pass
else:
    raise Exception('Unknown cross_section_type')


# Loading
P_over_Po = 0.3
P = P_over_Po*Po
max_disp = 12
num_steps = 200
disp_increment = max_disp/num_steps

# Initilize output
end_moment = np.zeros(num_steps)
mid_moment = np.zeros(num_steps)
mid_disp = np.zeros(num_steps)
end_rotation = np.zeros(num_steps)
section_stiffness = np.zeros((num_steps,6))

# Create OpenSees model
ops.wipe()  
ops.model('basic', '-ndm', 2, '-ndf', 3)

# Define Nodes
ops.node(1,0,0)
ops.node(2,0,1/6*L)
ops.node(3,0,1/3*L)
ops.node(4,0,1/2*L)
ops.node(5,0,2/3*L)
ops.node(6,0,5/6*L)
ops.node(7,0,L)
ops.fix(1,1,1,0)
ops.fix(7,1,0,0)

# Define cross section
if cross_section_type == 'RC':
    # RC cross section
    ops.uniaxialMaterial('Concrete01', 605, fc, 2*fc/Ec, 0.2*fc, 0.01)
    #ops.uniaxialMaterial('ENT',605,2*fc/0.002)
    #ops.uniaxialMaterial('Concrete01', 605, fc, 0.002, 0.005, 4500.0)
    ops.uniaxialMaterial('ElasticPP', 336, Es, Fy/Es)
    ops.section('Fiber',893)
    ops.patch('circ', 605, 30, 10, 0.0, 0.0, 0.0, 0.5*D, 0.0, 360.0)
    ops.layer('circ', 336, n_bar, Ab, 0.0, 0.0, 0.5*D-xp)
elif cross_section_type == 'WF':
    ops.uniaxialMaterial('ElasticPP', 336, E, Fy/E)
    ops.section('WFSection2d', 893, 336, d, tw, bf, tf, 50, 20)
elif cross_section_type == 'Elastic':
    ops.section('Elastic', 893, 29000, 1000, 300)
else:
    raise Exception('Unknown cross_section_type')


# Define Elements
ops.geomTransf('Corotational', 937)
ops.beamIntegration('Legendre', 342, 893, 3)
#element_type = 'mixedBeamColumn';
element_type = 'dispBeamColumn';
ops.element(element_type, 1, 1, 2, 937, 342)
ops.element(element_type, 2, 2, 3, 937, 342)
ops.element(element_type, 3, 3, 4, 937, 342)
ops.element(element_type, 4, 4, 5, 937, 342)
ops.element(element_type, 5, 5, 6, 937, 342)
ops.element(element_type, 6, 6, 7, 937, 342)


# Define vertical load
ops.timeSeries("Linear", 177)
ops.pattern("Plain", 582, 177)
ops.load(7,0,-P,0)

ops.system("UmfPack")
ops.numberer("Plain")
ops.constraints("Plain")
ops.test("NormUnbalance",1e-8,10,0)
ops.algorithm("Newton")
ops.integrator("LoadControl", 1.0)
ops.analysis("Static")
ops.analyze(1)
ops.reactions()
ops.loadConst('-time',0.0)

# Define moment
ops.timeSeries("Linear", 836)
ops.pattern("Plain", 769, 836)
ops.load(1,0,0,-1)
ops.load(7,0,0,1)

ops.integrator("DisplacementControl", 4, 1, disp_increment)

for iStep in range(num_steps):
    ops.analyze(1)
    ops.reactions()
    
    end_moment[iStep] = ops.getTime()
    mid_moment[iStep] = ops.eleForce(3,6)
    mid_disp[iStep] = ops.nodeDisp(4,1)
    end_rotation[iStep] = ops.nodeDisp(1,3)
    for i in range(6):
        section_stiffness[iStep,i] = ops.sectionStiffness(i+1,2)[3]

    
# CBDI Buckling Calculations   
Pcr = np.zeros(num_steps)
xi = [1/12, 3/12, 5/12, 7/12, 9/12, 11/12]
for iStep in range(num_steps):
    Pcr[iStep] = PcrCBDI(xi,section_stiffness[iStep,:],L)

ind = np.argmax(end_moment)
mid_disp_at_max = mid_disp[ind]    
    
# Plot results
make_simple_plots = False
if make_simple_plots:
    ymax = 1.1*max(mid_moment)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    plt.plot(mid_disp,end_moment,label='Applied (End) Moment')
    plt.plot(mid_disp,mid_moment,label='Max (Mid-height) Moment')
    plt.plot([mid_disp_at_max,mid_disp_at_max],[0,ymax],label='Point of Max Applied Moment')
    plt.xlabel('Mid-height Displacement (in.)')
    plt.ylabel('Bending Moment (kip-in)')
    plt.legend()
    ax1.set_ylim([0.0, ymax])

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    for iIP in range(6):
        plt.plot(mid_disp,section_stiffness[:,iIP]/EIgross,'-o',label=f'Section {iIP}')
    plt.plot([mid_disp_at_max,mid_disp_at_max],ax2.get_ylim(),label='Point of Max Applied Moment')
    plt.xlabel('Mid-height Displacement (in.)')
    plt.ylabel('Normalized Flexural Stiffness, EI/EI_initial')
    plt.legend()

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1,1,1)
    plt.plot(mid_disp,Pcr,label='Pcr (CBDI)')
    plt.plot([0,max_disp],[P,P],label='Applied Axial Load')
    plt.plot([mid_disp_at_max,mid_disp_at_max],ax3.get_ylim(),label='Point of Max Applied Moment')
    plt.xlabel('Mid-height Displacement (in.)')
    plt.ylabel('Critical Buckling Load (kips)')
    plt.legend()

else:
    # @todo - work on legends
        
    in_to_mm = 25.4
    kips_to_MN = 0.004448222
    kipin_to_MNm = 0.000112984839

    plt.rc('font',family='serif')
    plt.rc('mathtext',fontset='dejavuserif')
    plt.rc('axes',labelsize=8)
    plt.rc('axes',titlesize=8)
    plt.rc('legend',fontsize=8)
    plt.rc('xtick',labelsize=8)
    plt.rc('ytick',labelsize=8)

    # Bending Moment vs. Mid-height Displacement
    ymax = 3.0
    fig1 = plt.figure(figsize=(3.5,2.5))
    ax1 = fig1.add_axes([0.14,0.18,0.83,0.79])
    plt.plot(mid_disp*in_to_mm,end_moment*kipin_to_MNm,label='Applied (End) Moment')
    plt.plot(mid_disp*in_to_mm,mid_moment*kipin_to_MNm,label='Max (Mid-height) Moment')
    plt.plot([mid_disp_at_max*in_to_mm,mid_disp_at_max*in_to_mm],[0,ymax],'--k',label='Point of Max Applied Moment')
    plt.xlabel('Mid-height Displacement (mm)')
    plt.ylabel('Bending Moment (MN-m)')
    plt.legend()
    ax1.set_xlim([0.0, max_disp*in_to_mm])
    ax1.set_ylim([0.0, ymax])
    plt.savefig('Figure_Xa_BeamColumn.png',dpi=300)

    # Flexural Stiffness vs. Mid-height Displacement
    ymax = 1.0
    fig2 = plt.figure(figsize=(3.5,2.5))
    ax2 = fig2.add_axes([0.13,0.18,0.84,0.80])
    for iIP in range(6):
        plt.plot(mid_disp*in_to_mm,section_stiffness[:,iIP]/EIgross,'-',label=f'Section {iIP}')
    plt.plot([mid_disp_at_max*in_to_mm,mid_disp_at_max*in_to_mm],ax2.get_ylim(),'--k',label='Point of Max Applied Moment')
    plt.xlabel('Mid-height Displacement (mm)')
    plt.ylabel('Normalized Flexural Stiffness, $EI/EI_{initial}$')
    plt.legend()
    ax2.set_xlim([0.0, max_disp*in_to_mm])
    ax2.set_ylim([0.0, ymax])
    plt.savefig('Figure_Xb_BeamColumn.png',dpi=300)

    # Critical Buckling Load vs. Mid-height Displacement
    ymax = 32.0
    fig3 = plt.figure(figsize=(3.5,2.5))
    ax3 = fig3.add_axes([0.13,0.18,0.84,0.80])
    plt.plot(mid_disp*in_to_mm,Pcr*kips_to_MN,label='Pcr (CBDI)')
    plt.plot([0,max_disp*in_to_mm],[P*kips_to_MN,P*kips_to_MN],label='Applied Axial Load')
    plt.plot([mid_disp_at_max*in_to_mm,mid_disp_at_max*in_to_mm],[0,ymax],'--k',label='Point of Max Applied Moment')
    plt.xlabel('Mid-height Displacement (mm)')
    plt.ylabel('Critical Buckling Load (MN)')
    plt.legend()
    ax3.set_xlim([0.0, max_disp*in_to_mm])
    ax3.set_ylim([0.0, ymax])
    plt.savefig('Figure_Y_BeamColumn.png',dpi=300)

plt.show()