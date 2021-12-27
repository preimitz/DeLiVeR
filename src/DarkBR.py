# Libraries to load
import numpy,scipy,math
import os
import matplotlib
import matplotlib.pyplot as plt
import csv
from src.form_factors import F2pi,F3pi,F4pi,F6pi,FPiGamma,FEtaGamma,FK,FKKpi,FKKpipi,FEtaOmega,FEtaPhi,FPhiPi,FPhiPiPi,FOmegaPion,FOmPiPi,FEtaPiPi,FEtaPrimePiPi,Fppbar
from src.functions import alpha, Resonance

######################
### SM parameters ####
######################

# quark masses #
me_ = 0.5109989461*1e-3
mMu_ = 0.105658
mtau_ = 1.77686
mlep_ = [me_,mMu_,mtau_,0.,0.,0.]
mc_ =   1.5
mb_ = 4.8
mt_ = 171.0

#lightest charmed meson mass
mD0 = 1.864
#lightest bottom meson mass
mUps = 9.46

# fundamental constants #
hbar = 6.58211951e-25 # GeV.s
cLight = 299792458 # m/s

######################
### BSM parameters ###
######################
g  = 1.
ge   = 3.02822e-1       # Electromagnetic coupling (unitless).
cDM = 3 # mDM = cDM*mX

allDMtypes = ["complex scalar","Majorana","Majorana fermion","Dirac","Dirac fermion","inelastic","No"]

######################
### decay channels ###
######################

## hadronic channels ##
# form factors #
Fchannels = [F2pi,F3pi,FPiGamma,FEtaGamma,FEtaOmega,FEtaPhi,FPhiPi,FOmegaPion,FEtaPiPi,FEtaPrimePiPi,F4pi,F6pi,FK,FKKpi,FKKpipi,FPhiPiPi,FOmPiPi,Fppbar]    
# label #
labelhad = ["2pi","3pi","PiGamma","EtaGamma","EtaOmega","EtaPhi","PhiPi","OmegaPion","EtaPiPi","EtaPrimePiPi","4pi","6pi","KK","KKpi","KKpipi","PhiPiPi","OmPiPi","ppbar","nnbar"]
## leptonic channels ##
# labels #
labellep = ["elec","muon", "tau", "nue","numu","nutau"]

# labels combined #
label = labelhad+labellep

# all possibilities
allchannels = {'2pi' : ["F2pi",-1], '3pi' : ["F3pi",-1],'PiGamma' : ["FPiGamma",-1],'EtaGamma' : ["FEtaGamma",-1],'EtaOmega' : ["FEtaOmega",-1],'EtaPhi' : ["FEtaPhi",-1],'PhiPi': ["FPhiPi",-1],'OmegaPion' : ["FOmegaPion",-1],'EtaPiPi' : ["FEtaPiPi",-1],'EtaPrimePiPi' : ["FEtaPrimePiPi",-1], '6pi_c' : ["F6pi",1],'6pi_n' : ["F6pi",0],'KK_c' : ["FK",1],'KK_n' : ["FK",0],'KKpi_0' : ["FKKpi",0],'KKpi_1' : ["FKKpi",1],'KKpi_2' : ["FKKpi",2],'KKpipi_0' : ["FKKpipi",0],'KKpipi_1' : ["FKKpipi",1],'KKpipi_2' : ["FKKpipi",2],'KKpipi_3' : ["FKKpipi",3],'PhiPiPi_n' : ["FPhiPiPi",0],'PhiPiPi_c' : ["FPhiPiPi",1],'OmPiPi_n' : ["FOmPiPi",0],'OmPiPi_c' : ["FOmPiPi",1],'ppbar' : ["Fppbar",0],'nnbar' : ["Fppbar",1],'4pi_c': [F4pi,1], '4pi_n' : [F4pi,0]}


######################
### cross sections ###
######################

# e+e- -> mu+mu- #
def xSection2e2mu(Q2):
    sigma2e2mu = (4 * math.pi * alpha.alphaEM(Q2)**2 * Resonance.gev2nb)/(3*Q2)
    return sigma2e2mu

def pertNormXsec(Q2,Qf,mF):
    if Q2<(2*mF)**2: return 0
    pre = 3*(alpha.alphaEM(Q2)**2)*Resonance.beta(Q2,mF,mF)*Qf**2/4./Q2
    phi = 2.*math.pi
    intThet = scipy.integrate.quad(pertDiffXsec,0, math.pi,args=(Q2,mF))[0]
    return pre*phi*intThet*Resonance.gev2nb/xSection2e2mu(Q2)

####################
### decay widths ###
####################

### widths with SM final states ###
# vector -> f fbar
def GammaDMf(Cf, g, m, xF, mF):
    if m<2*mF:
        return 0
    else:  
        pre = Cf*(g* xF)**2/12/math.pi
        kin = m*(1 + 2*(mF**2/m**2))*numpy.sqrt(1- 4*(mF**2/m**2))
        return pre*kin

def Gamma3gamma(xLep):
    mass = []
    widths_gamma_gamma_gamma = []
    m = 0
    while m < 3.0 :
        mass.append(m)
        width3gamma= GammaDM3gamma(g, m, xLep[0])
        widths_gamma_gamma_gamma.append(width3gamma)
        if m<2*me_+0.001:
            m += 0.00001
        else:
            m+=0.01
    return (mass,widths_gamma_gamma_gamma)

def calcWidthQuarksDP():
    mass = []
    widthscc = []
    widthsbb = []
    widthstt = []
    m = 0
    while m < 3.0 :
        mass.append(m)
        width2cc = GammaDMf(3.,g,m,2*ge/3,mc_)
        width2bb = GammaDMf(3.,g,m,-1*ge/3,mb_)
        width2tt = GammaDMf(3.,g,m,2*ge/3,mt_)
        widthscc.append(width2cc)
        widthsbb.append(width2bb)
        widthstt.append(width2tt)
        m+=0.001
    return (mass,widthscc,widthsbb,widthstt)


# vector -> 3 gamma
def GammaDM3gamma(g, m, xF):
    if m>2*me_:
        return 0
    else:  
        return (((g*xF)**2.0*ge**6.0)/(4.0*math.pi)**4.0/(
                    2.0**7.0*3.0**6.0*5.0**2.0*math.pi**3.0)*(m**9.0/me_**8.0)*(
                    17.0/5.0 + (67.0*m**2.0)/(42.0*me_**2.0) +
                    (128941.0*m**4.0)/(246960.0*me_**4.0)))

### Widths with DM final states ###

# vector -> DM DM
def Gamma2DM(g, mX, mDM,DMtype="No", splitting=0.1): # Vector Boson -> X X (DM)
    if mX<2*mDM: return 0
    #model-independent prefactor for two body decay
    pre = g**2/48./math.pi*mX*(1-4*mDM**2/mX**2)**(1/2)/mX**2
    #model-dependent matrix element
    me = 0.
    if DMtype in ["complex scalar","Majorana","Majorana fermion"]:
        me = mX**2*(1- (4*mDM**2/mX**2))
        # identical particle factor
        if DMtype in ["Majorana","Majorana fermion"]: me*=2.
    elif DMtype in ["Dirac","Dirac fermion"]:
        me = 4*mX**2*(1+2*mDM**2/mX**2)
    elif DMtype=="inelastic":
        mDM2 = mDM+splitting
        if mX<mDM+mDM2: return 0
        else:
            pre = g**2/48./math.pi*((1-(mDM+mDM2)**2/mX**2)*(1-splitting**2/mX**2))**0.5
            me = 4.*mX**2*((1+2*mDM*mDM2/mX**2)-splitting**2*(1+(mDM+mDM2)**2/mX**2)/2./mX**2)
    else:
        print("DM type not specified")
    return pre*me
    

# Decay Width
def Width_calc(labelhad,labellep,xLep,DMtype="No",Rchi=3,splitting=0.1):
    ## check if DMtype correct
    if DMtype not in allDMtypes: 
        print("DM type not specified correctly")
        return 0
    ## define mass array and width dictionaries
    mass = []
    wtotal=[]
    widthshad = {}
    widthslep = {}
    ## reset dark param
    for i in range(0, len(labelhad)):
        widthshad["w_{0}".format(labelhad[i])] = []
    for i in range(0, len(labellep)):
        widthslep["w_{0}".format(labellep[i])] = []
    if DMtype != "No": widthslep["w_DM"] = []
    # widthsDMlep["wDM_gamma_gamma_gamma"]= []
    # initialization
    F4pi.readHadronic_Current()   
    FK.initialize()
    # calc Widths
    m = 2* me_ 
    while m < 2.0 :
        mass.append(m)
        # Gmuon = GammaDMf(1.,gDM,mDM,-1,mMu_)
        wTot = 0
        for i in range(0, 17):
            width = getattr(Fchannels[i], 'GammaDM')(m)
            widthshad["w_{0}".format(labelhad[i])].append(width)
            wTot += width
        w_ppbar = Fppbar.GammaPDM(m)
        widthshad["w_ppbar"].append(w_ppbar)
        w_nnbar = Fppbar.GammaNDM(m)
        widthshad["w_nnbar"].append(w_nnbar)
        # sum hadrons
        wTot += w_ppbar+w_nnbar
        # leptons
        coeff = [1.,1.,1.,1/2,1./2,1./2]
        for i in range(0,len(mlep_)):
            wlep = GammaDMf(coeff[i],g,m,xLep[i],mlep_[i])
            widthslep["w_%s" %labellep[i]].append(wlep)
            wTot+=wlep
        # DM
        if DMtype !="No":
            if DMtype=="inelastic": width2DM = Gamma2DM(g, m, m/Rchi,DMtype,splitting=splitting)
            else: width2DM = Gamma2DM(g, m, m/Rchi,DMtype)
            widthslep["w_DM"].append(width2DM)
            wTot += width2DM
        wtotal.append(wTot)  
        #print (mDM)
        m+=0.001
    return (mass,widthshad,widthslep,wtotal)

def calcTotalWidth(label,mass,widthshad,widthslep):
    wtotal=[]
    widths = {**widthshad, **widthslep}
    for i in range(0, len(mass)):
        wTot = 0
        for channel in range(0, len(label)):
            if label[channel] == 'nu':  
                wTot += 3*widths["w_nu"][i]
            else:
                wTot += widths["w_{0}".format(label[channel])][i]
        wtotal.append(wTot)
    return (wtotal)

### saving widths and lifetime ###
def Width_save(model,label,mass,wtotal,widths,BRs,DMtype="No"):
    if DMtype not in allDMtypes: 
        print("DM type not specified correctly")
        return 0
    if DMtype !="No":
        brType = 'DM_'+DMtype.replace(" ", "_")
        label+=["DM"]
    else:
        brType = 'SM'
    for channel in range(0, len(label)):
        with open('models/'+model+"_"+brType+'/widths/width_'+model+'_%s.txt' % label[channel], 'w') as txtfile:
            for i in range(0,len(mass)):
                txtfile.write("%s \t %s\n" %(mass[i],widths["w_{0}".format(label[channel])][i]))
            txtfile.close()   
    with open('models/'+model+"_"+brType+'/widths/width_'+model+'_total_'+brType+'.txt', 'w') as txtfile:
        for i in range(0,len(mass)):
            txtfile.write("%s \t %s\n" %(mass[i],wtotal[i]))
        txtfile.close()  
    #lifetime    
    with open('models/'+model+"_"+brType+'/lifetime_tau_'+model+'_'+brType+'.txt', 'w') as txtfile:
        for i in range(0,len(mass)):
            txtfile.write("%s \t %s\n" %(mass[i],hbar/(wtotal[i])))
        txtfile.close()  
    with open('models/'+model+"_"+brType+'/lifetime_ctau_'+model+'_'+brType+'.txt', 'w') as txtfile:
        for i in range(0,len(mass)):
            txtfile.write("%s \t %s\n" %(mass[i],hbar*cLight/(wtotal[i])))
        txtfile.close()    
        
# Decay Width
def Width_calc_single(model,channel,DMtype="No"):
    if DMtype not in allDMtypes: 
        print("DM type not specified correctly")
        return 0
    if DMtype !="No":
        brType = 'DM_'+DMtype.replace(" ", "_")
    else:
        brType = 'SM'
    ## define mass array and width dictionaries
    mass = []
    width_tot = []
    with open('models/'+model.replace(" ", "_")+"_"+brType+'/widths/width_'+model.replace(" ", "_")+'_total_'+brType+'.txt',"r") as f:
            lines = f.readlines()
            mass = [line.split()[0] for line in lines]
            width_tot = [line.split()[1] for line in lines]
    # initialization
    F4pi.readHadronic_Current()   
    FK.initialize()
    # calc Widths
    width = []
    for imass in range(0,len(mass)):
        # Gmuon = GammaDMf(1.,gDM,mDM,-1,mMu_)
        func,mode = allchannels.get(channel)
        if mode<0:
            width.append(getattr(func,"GammaDM")(float(mass[imass])))
        else:
            width.append(getattr(func,"GammaDM_mode")(float(mass[imass]),int(mode)))
    with open('models/'+model.replace(" ", "_")+"_"+brType+'/widths/width_'+model.replace(" ", "_")+"_"+channel+"_"+brType+'.txt',"w") as txtfile:
        for i in range(0,len(mass)):
            txtfile.write("%s \t %s\n" %(mass[i],width[i]))
        txtfile.close()
    with open('models/'+model.replace(" ", "_")+"_"+brType+'/brs_'+brType+'/bfrac_'+model.replace(" ", "_")+"_"+channel+"_"+brType+'.txt',"w") as txtfile:
        for i in range(0,len(mass)):
            br = 0
            if float(width_tot[i])>0: br = width[i]/float(width_tot[i])
            txtfile.write("%s \t %s\n" %(mass[i],br))
        txtfile.close()
        
########################
### branching ratios ###
########################

# calculate all branching ratios
def calcBRs(labelhad,labellep,mass,widthshad,widthslep,wtotal,DMtype="No"):
    if DMtype not in allDMtypes: 
        print("DM type not specified correctly")
        return 0
    BRshad = {}     
    BRslep = {}
    for i in range(0, len(labelhad)):
        BRshad["br_{0}".format(labelhad[i])] = []
    for i in range(0, len(labellep)):
        BRslep["br_{0}".format(labellep[i])] = []
    if DMtype !="No":
        BRslep["br_DM"] = []
    for i in range(0, len(mass)):
        for channel in range(0, len(labelhad)):
            BRshad["br_{0}".format(labelhad[channel])].append(widthshad["w_{0}".format(labelhad[channel])][i]/wtotal[i])
        for lepchannel in range(0, len(labellep)):
            BRslep["br_{0}".format(labellep[lepchannel])].append(widthslep["w_{0}".format(labellep[lepchannel])][i]/wtotal[i])
        if DMtype !="No":
            BRslep["br_DM"].append(widthslep["w_DM"][i]/wtotal[i])
    return (BRshad,BRslep)

# Saving Files

def Br_save(model,label,mass,wtotal,widths,BRs,DMtype="No"):
    if DMtype not in allDMtypes: 
        print("DM type not specified correctly")
        return 0
    if DMtype !="No":
        brType = 'DM_'+DMtype.replace(" ", "_")
        label+=["DM"]
    else:
        brType = 'SM'
    for channel in range(0, len(label)):
        with open('models/'+model+"_"+brType+'/brs_'+brType+'/bfrac_'+model+'_%s.txt' % label[channel], 'w') as txtfile:
            for i in range(0,len(mass)):
                txtfile.write("%s \t %s\n" %(mass[i],BRs["br_{0}".format(label[channel])][i]))
            txtfile.close() 
    # visible and invisible branching ratios
    # visible
    with open('models/'+model+"_"+brType+'/bfrac_'+model+'_visible_'+brType+'.txt', 'w') as txtfile:
        for imass in range(0,len(mass)):
            BRvis = 0
            for channel in range(0, len(label)):
                if label[channel] in ['DM','nue','numu','nutau']:
                    continue
                BRvis += BRs["br_{0}".format(label[channel])][imass]
            txtfile.write("%s \t %s\n" %(mass[imass],BRvis))
        txtfile.close()
    # invisible
    with open('models/'+model+"_"+brType+'/bfrac_'+model+'_invisible_'+brType+'.txt', 'w') as txtfile:
        for imass in range(0,len(mass)):
            BRinvis = 0
            for inu in range(3,len(labellep)):
                BRinvis += BRs["br_{0}".format(labellep[inu])][imass] 
            if DM !="No": BRinvis +=BRs["br_DM"][imass]
            txtfile.write("%s \t %s\n" %(mass[imass],BRinvis))
        txtfile.close()  
    #hadrons
    with open('models/'+model+"_"+brType+'/bfrac_'+model+'_hadrons_'+brType+'.txt', 'w') as txtfile:
        for i in range(0,len(mass)):
            BRhad = 0
            for channel in range(0, len(labelhad)):
                BRhad += BRs["br_{0}".format(labelhad[channel])][i]
            txtfile.write("%s \t %s\n" %(mass[i],BRhad))
        txtfile.close() 
    #leptons
    with open('models/'+model+"_"+brType+'/bfrac_'+model+'_leptons_'+brType+'.txt', 'w') as txtfile:
        for i in range(0,len(mass)):
            BRlep = 0
            for channel in range(0, len(labellep)):
                BRlep += BRs["br_{0}".format(labellep[channel])][i]
            txtfile.write("%s \t %s\n" %(mass[i],BRlep))
        txtfile.close() 

#joining KK channels

def saveKKFiles(model,mass,widths):
    with open('models/'+model+'/widths_'+model+'/width_'+model+'_K_K.txt', 'w') as txtfile:
        for i in range(0,len(mass)):
            txtfile.write("%s \t %s\n" %(mass[i],widths["w_KK_n"][i]+widths["w_KK_c"][i]))
        txtfile.close()   
    with open('models/'+model+'/widths_'+model+'/width_'+model+'_K_K_pi.txt', 'w') as txtfile:
        for i in range(0,len(mass)):
            txtfile.write("%s \t %s\n" %(mass[i],widths["w_KKpi_0"][i]+widths["w_KKpi_1"][i]+widths["w_KKpi_2"][i]))
        txtfile.close()              
            
# Function executing everything
def generate_br(model="dark photon",DMtype="No",Rchi=3,label=label):
    if DMtype not in allDMtypes: 
        print("DM type not specified correctly")
        return 0
    if DMtype !="No":
        brType = 'DM_'+DMtype.replace(" ", "_")
        label+=["DM"]
    elif DMtype == False or DMtype=="No":
        brType = 'SM'
    else:
        print("Unclear specification of DM type. Possible candidates are")
        print("complex scalar")
        print("Majorana fermion")
        print("Dirac fermion")
        print("If the vector mediator does not couple to DM, just set 'DMtype='No'")    
    if isinstance(model, str):
        if model=="check":
            print("All generated models:")
            print("")
            dirs = os.listdir(r'./models')
            for folder in dirs:
                if os.path.isdir('./models/'+folder):
                    print(folder)
        elif model=="help":
            print("")
            print("All generated models:")
            print("")
            dirs = os.listdir(r'./models')
            for folder in dirs:
                if os.path.isdir('./models/'+folder):
                    print(folder)
            print("")
            print("If you want to add a new model, just set the couplings directly in the format")
            print("")
            print("generate_br(model=['model_name', g_d, g_u,g_s,g_c,g_b,g_t,g_e,g_mu,g_tau,g_nue,g_numu,g_nutau],DMtype='complex scalar'):")
            print("Possible DM candidates are")
            print("complex scalar")
            print("Majorana fermion")
            print("Dirac fermion")
            print("The DM mass is defined through 'Rchi'. For example, if the vector mediator is three times the DM mass, we set Rchi=3.")
            print("If the vector mediator does not couple to DM, just set 'DMtype='No''")
    else:
        if os.path.isdir("models/"+model[0].replace(" ", "_")+"_"+brType):
            print("model ", model[0]," with ",brType, " couplings already exists!")
            print("Please choose a different model name or delete folder to create a new one!")
        else:
            if len(model) != 13:
                return print("The model was not entered in the right way. See generate_br(model='help') for help")
            os.mkdir("models/"+model[0].replace(" ", "_")+"_"+brType)
            os.mkdir("models/"+model[0].replace(" ", "_")+"_"+brType+"/widths")
            os.mkdir("models/"+model[0].replace(" ", "_")+"_"+brType+"/brs_"+brType)
            print("new model called ", model[0]," with ",brType," couplings")
            cMed_d, cMed_u, cMed_s,cMed_charm,cMed_bottom,cMed_top,c_e,c_mu,c_tau,c_nue,c_numu,c_nutau = [model[i] for i in range(1,len(model))]
            xLep = [c_e,c_mu,c_tau,c_nue,c_numu,c_nutau]
            for channel in Fchannels:
                channel.resetParameters(g,0.,0.,0.,cMed_u,cMed_d,cMed_s)   
            mass,widthshad,widthslep,wtotal = Width_calc(labelhad,labellep, xLep, DMtype = DMtype,Rchi=Rchi)
            BRshad,BRslep = calcBRs(labelhad,labellep,mass,widthshad,widthslep,wtotal,DMtype=DMtype)
            # saving BRs/widths in txt files
            widths = {**widthshad, **widthslep}
            BRs = {**BRshad, **BRslep}
            Br_save(model[0].replace(" ", "_"),label,mass,wtotal,widths,BRs,DMtype = DMtype)
            

##############################            
########### R ratio ##########
##############################

cMed_u = 2/3.
cMed_d = -1./3.
cMed_s = -1./3.
gDM = 1.

# non-perturbative
def Rvalues_had(labelhad=labelhad):
    ## define mass array and width dictionaries
    massDM = []
    Rtotal=[]
    RvalLVP = {}
    # initialize R dict
    for i in range(0, len(labelhad)):
        RvalLVP[label[i]] = []
    # initialization of some currents
    F4pi.readHadronic_Current()   
    FK.initialize()
    # calc Widths
    mDM = 1e-4
    while mDM < 2.0:
        massDM.append(mDM)
        norm = xSection2e2mu(mDM**2)
        Rsum = 0
        for i in range(0, 17):
            xsec = getattr(Fchannels[i], 'sigmaSM')(mDM**2)
            Rval = xsec/norm
            RvalLVP[label[i]].append(Rval)
            Rsum += Rval
        #ppbar/nnbar
        r_ppbar = Fppbar.sigmaP(mDM**2)/norm
        RvalLVP["ppbar"].append(r_ppbar)
        r_nnbar = Fppbar.sigmaN(mDM**2)/norm
        RvalLVP["nnbar"].append(r_nnbar)
        # sum hadrons
        Rsum += r_ppbar+r_nnbar
        # total
        Rtotal.append(Rsum)
        mDM += 0.01
    return (massDM,RvalLVP,Rtotal)

# perturbative
#---------------  eq. (9.7) of PDG - Quantum Chromodynamics  -------------------# 

def xSecInclusive(Q,cMed_u,cMed_d,cMed_s):
    if Q<(2*mD0):
        sumQq = (1*(cMed_u)+1*(cMed_d)+1*(cMed_s))
        sumQq2 = (1*(cMed_u)**2+1*(cMed_d)**2+1*(cMed_s)**2)
        nf=3
    if (2*mD0)<=Q<=(2* mUps):
        sumQq =(2*(cMed_u)+1*(cMed_d)+1*(cMed_s))
        sumQq2 = (2*(cMed_u)**2+1*(cMed_d)**2+1*(cMed_s)**2)
        nf = 4
    if (2* mUps)<Q:
        sumQq = (2*(cMed_u)+2*(cMed_d)+1*(cMed_s))
        sumQq2 = (2*(cMed_u)**2+2*(cMed_d)**2+1*(cMed_s)**2)
        nf= 5
    nc = 3
    Rew = nc * sumQq2#/ge**2
    eta = sumQq**2/(3*sumQq2)
    coeff = [1.,1.9857 - 0.1152*nf,-6.63694 - 1.20013*nf - 0.00518*nf**2 - 1.240*eta,
         -156.61 + 18.775*nf - 0.7974*nf**2 + 0.0215*nf**3-(17.828 - 0.575*nf)*eta]
    lambQCD =0
    if Q>1.:
        for n in range(0,4):
            lambQCD += coeff[n]*(alpha.alphaQCD(Q)/math.pi)**(n+1)
    return Rew*(1+lambQCD)

# data
#---------------  PDG - total rate to hadrons (2020) -------------------# 
dataPDG = numpy.loadtxt(os.getcwd()+'/../xsec_files/python_xsec/R_txt_files/R_Gamma_PDG_2020.txt')
massPDG = dataPDG[:,0]
RPDG= dataPDG[:,3]
up_eRPDG= dataPDG[:,4]
low_eRPDG= dataPDG[:,5]
#---------------  inclusive R (< 2GeV) -------------------# 

dataInc = numpy.loadtxt(os.getcwd()+'/src/data/R_inclusive.txt')
massInc = dataInc[:,0]
RInc= dataInc[:,1]

# function to save Rvalues
def Rvalues_calc():
    for channel in Fchannels:
        channel.resetParameters(gDM,0.,0.,0.,cMed_u,cMed_d,cMed_s)   

    masses,Rval,Rtotal = Rvalues_had()
    Rpert = []
    for i in range(0,len(masses)):
        Rpert.append(xSecInclusive(masses[i],cMed_u,cMed_d,cMed_s))
    return masses, Rval, Rtotal, Rpert
    
# function to plot Rvalues
#---------------  plot -------------------# 
def Rvalues_plot(masses, Rtot, Rpert, Rvalues, Rsingle=[], title="model"):
    plt.figure(figsize=(9., 4.8))
    plt.errorbar(massPDG,RPDG, [low_eRPDG,up_eRPDG] , color ='black', ls ='none', fmt = '.', ms =2.5, fillstyle='full',elinewidth=0.5, capsize=0.5,capthick = 0.3, label= "PDG data 2020", zorder = -5)

    plt.plot(masses,Rtot, c='darkorange', lw =1.3, label='DP $\\gamma-like$')
    if len(Rsingle)>0:
        for xR in Rsingle:
            plt.plot(masses,Rvalues[xR], lw=1.2, label=xR)
    plt.plot(masses,Rpert, c='limegreen', lw =1.2, label='quarks+QCD corrections')
    #plt.plot(np.asarray(massRq)[idxDPge[-1]], np.asarray(RtotalDPgeint)[idxDPge[-1]], 'r.', label='1.73 GeV')
    #plt.plot(np.asarray(massRq)[idxDPgenQCD[-1]], np.asarray(RtotalDPgeint)[idxDPgenQCD[-1]], 'r.', label='1.8 GeV')

    # plt.axhline(y=pertNormXsec(cMed_u)+pertNormXsec(cMed_d)+pertNormXsec(cMed_s), color='purple',lw =0.8, linestyle='-')
    plt.title(title)
    plt.xlabel("$\\sqrt{s}$ [GeV]",fontsize=14)
    plt.ylabel("$R^{F}_{\mu} = \\frac{\sigma(e^+e^- \\to \\; V \\; \\to \\; \\mathrm{hadrons})}{\sigma{(e^+e^- \\to \\; \\mu^+\\mu^-)}}$",fontsize=14)
    plt.legend(ncol=2)

    plt.xlim([0.3, 2])
    plt.ylim([1.e-1, 100])
    plt.yscale("log")
    # plt.xscale("log")
    # plt.savefig("plots/Rplot_DP_ge_DC_model_QCDcorrections.pdf")
    plt.show()
    plt.close()

    
def Rvalues_save(masses, Rtot, Rpert, Rvalues,model="dark photon",DM="No"):
    Rtype = 'SM'
    if DM in allDMtypes:
        RType = 'DM_'+DM.replace(" ", "_")
    elif DM=="No":
        RType = 'SM'
    else:
        print("DM type not specified")
    for ichannel in labelhad:
        with open('models/'+model.replace(" ", "_")+'_'+Rtype+'/Rvalues/R_'+ichannel+'.txt', 'w') as txtfile:
            for i in range(0,len(masses)):
                txtfile.write("%s \t %s\n" %(masses[i],Rvalues[ichannel][i]))
            txtfile.close()

#function that reads a table in a .txt file and converts it to a numpy array
def readfile(filename):
    array = []
    with open(filename) as f:
        for line in f:
            if line[0]=="#":continue
            words = [float(elt.strip()) for elt in line.split( )]
            array.append(words)
    return numpy.array(array)
            
def plot_br(model="dark photon",had_channels=labelhad,lep_channels=labellep,DMtype=False):
    # set if DM included or not
    if DMtype in allDMtypes:
        DM = True
        brType = 'DM_'+DMtype.replace(" ", "_")
    if DM == False:
        brType = 'SM'
    
    
    #colors = ["k","b","r","g", "darkorange", "purple", "dodgerblue", "yellowgreen","violet", "crimson","teal", "chocolate"]
    lepcolors = ["turquoise","forestgreen", "gold","plum"]

    # initiate figure
    matplotlib.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(9., 4.8))
    
    # leptonic channels
    for channel in range(0, len(lep_channels)):
        data=readfile('models/'+model.replace(" ", "_")+"_"+brType+'/brs_'+brType+'/bfrac_'+model.replace(" ", "_")+'_%s.txt' % lep_channels[channel])
        ax.plot(data.T[0], data.T[1],label="$\\mathcal{F} = %s $" % lep_channels[channel] ,color=lepcolors[channel],ls="dashed",lw=1)
        #zorder+=1
    
    # hadronic channels
    for channel in range(0, len(had_channels)):
        data=readfile('models/'+model.replace(" ", "_")+"_"+brType+'/brs_'+brType+'/bfrac_'+model.replace(" ", "_")+'_%s.txt' % had_channels[channel])
        ax.plot(data.T[0], data.T[1],label="$\\mathcal{F} = %s $" % had_channels[channel] ,lw=1)
    
    # DM channel
    if DM==True:
        data=readfile('models/'+model.replace(" ", "_")+"_"+brType+'/brs_'+brType+'/bfrac_'+model.replace(" ", "_")+'_DM.txt')
        ax.plot(data.T[0], data.T[1],label="$\\mathcal{F} = DM $"  ,color="black",ls="dashed",lw=1)
    
    #frame
    ax.set_yscale("log")     
    ax.set_xlabel("$m_{X} \\;$/GeV")
    ax.set_ylabel("$BR(X \\to \\mathcal{F})$")
    ax.set_title("%s" %model)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return plt
    