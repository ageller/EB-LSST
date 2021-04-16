import numpy as np
from astropy.table import Table
from scipy import integrate
from astropy import units,constants


class unitsHolder():
	def __init__(self,
		time_unit = units.yr,
		length_unit = units.solRad,
		mass_unit = units.solMass):

		self.time_unit = time_unit
		self.length_unit = length_unit
		self.mass_unit = mass_unit

		self.Gconst = constants.G.to(length_unit**3*mass_unit**-1*time_unit**-2).value

class Star():
	#this should be sent values with units, and will return unitless values in the units from above
	def __init__(self, 
		mass, 
		radius, 
		luminosity, 
		convective_envelope_mass, 
		convective_envelope_radius,
		omega, #rad/time
		units_holder,
		sev = None):

		self.mass = mass.to(units_holder.mass_unit).value
		self.radius = radius.to(units_holder.length_unit).value
		self.luminosity = luminosity.to(units_holder.mass_unit*units_holder.length_unit**2*units_holder.time_unit**-3).value
		self.convective_envelope_mass = convective_envelope_mass.to(units_holder.mass_unit).value
		self.convective_envelope_radius = convective_envelope_radius.to(units_holder.length_unit).value
		self.omega = omega.to(1./units_holder.time_unit).value

		self.rzams = self.radius 
		self.core_mass = 0.
		self.core_radius = 0.
		self.gyration_radius = self.calcRg()
		self.moment_of_inertia = self.calcI()

		self.units_holder = units_holder

		self.sev = sev

	def calcI(self):
	#eq 35 of Hurley et al 2002. 
	#mc and rc are core mass and radius
	#with mc=rc=0, this reduces to the usual formula for moment of inertia
		#k2p = 0.1
		k2p = self.gyration_radius**2.
		k3p = 0.21
		# return rg**2.*(m - mc)*r**2. + k3p*mc*rc**2.
		return k2p*(self.mass - self.core_mass)*self.radius**2. + k3p*self.core_mass*self.core_radius**2.

	def calcRg(self):
	#from BSE
		logm = np.log10(self.mass)
		A = np.max([0.81, np.max([0.68, 0.68 + 0.4*logm])])
		C = np.max([-2.5, np.min([-1.5, -2.5 + 5.0*logm])])
		D = -0.1
		E = 0.025

		k2z = np.min([0.21, np.max([0.09 - 0.27*logm, 0.037 + 0.033*logm])])
		k2e = (k2z - E)*(self.radius/self.rzams)**C + E*(self.radius/self.rzams)**D #=rg**2.

		#print("gyration radius", k2e**0.5, k2e)
		return k2e**0.5

	def evolve(self, time):
		if (self.sev is not None):
			t = time.to(self.units_holder.time_unit).value
			self.mass = np.interp(t, (self.sev['tphys']).to(self.units_holder.time_unit).value, self.sev['mass'].to(self.units_holder.mass_unit).value)
			self.radius = np.interp(t, (self.sev['tphys']).to(self.units_holder.time_unit).value, self.sev['rad'].to(self.units_holder.length_unit).value)
			self.luminosity = np.interp(t, (self.sev['tphys']).to(self.units_holder.time_unit).value, self.sev['lum'].to(self.units_holder.mass_unit*self.units_holder.length_unit**2*self.units_holder.time_unit**-3).value)
			self.convective_envelope_mass = np.interp(t, (self.sev['tphys']).to(self.units_holder.time_unit).value, self.sev['menv'].to(self.units_holder.mass_unit).value)
			self.convective_envelope_radius = np.interp(t, (self.sev['tphys']).to(self.units_holder.time_unit).value, self.sev['renv'].to(self.units_holder.length_unit).value)
			self.core_mass = np.interp(t, (self.sev['tphys']).to(self.units_holder.time_unit).value, self.sev['massc'].to(self.units_holder.mass_unit).value)
			self.core_radius = np.interp(t, (self.sev['tphys']).to(self.units_holder.time_unit).value, self.sev['radc'].to(self.units_holder.length_unit).value)
			self.gyration_radius = self.calcRg()
			self.moment_of_inertia = self.calcI()

class Binary():
	#this should be sent values with units, and will return unitless values in the units from above
	#star1 and star2 should be of the star class from above
	def __init__(self,
		star1,
		star2,
		semi_major_axis,
		eccentricity,
		units_holder,
		bcm = None #astropy table, e.g., from cosmic : bcm = Table.from_pandas(bcmIn),
		):

		self.star1 = star1
		self.star2 = star2
		self.semi_major_axis = semi_major_axis.to(units_holder.length_unit).value
		self.period = 2.*np.pi*(self.semi_major_axis**3./(units_holder.Gconst*(self.star1.mass + self.star2.mass)))**0.5
		self.eccentricity = eccentricity

		self.units_holder = units_holder

		self.bcm = bcm

		#define the sev terms for the stars 
		if (bcm is not None):
			t1 = Table()
			t1['tphys'] = bcm['tphys']*units.Myr
			t1['mass'] = bcm['mass_1']*units.solMass
			t1['menv'] = bcm['menv_1']*units.solMass
			t1['massc'] = bcm['massc_1']*units.solMass
			t1['rad'] = bcm['rad_1']*units.solRad
			t1['renv'] = bcm['renv_1']*units.solRad
			t1['radc'] = bcm['radc_1']*units.solRad
			t1['lum'] = bcm['lum_1']*units.solLum
			self.star1.sev = t1

			t2 = Table()
			t2['tphys'] = bcm['tphys']*units.Myr
			t2['mass'] = bcm['mass_2']*units.solMass
			t2['menv'] = bcm['menv_2']*units.solMass
			t2['massc'] = bcm['massc_2']*units.solMass
			t2['rad'] = bcm['rad_2']*units.solRad
			t2['renv'] = bcm['renv_2']*units.solRad
			t2['radc'] = bcm['radc_2']*units.solRad
			t2['lum'] = bcm['lum_2']*units.solLum
			self.star2.sev = t2


	def angularMomentum(self):
		Jspin1 = self.star1.moment_of_inertia*self.star1.omega
		Jspin2 = self.star2.moment_of_inertia*self.star2.omega

		return self.Jorb() + Jspin1 + Jspin2

	def Jorb(self):
		#equation taken from BSE (matches)
		#    oorb = twopi/tb
		#    jorb = mass(1)*mass(2)/(mass(1)+mass(2))
		# &          *SQRT(1.d0-ecc*ecc)*sep*sep*oorb
		return self.star1.mass*self.star2.mass*(self.units_holder.Gconst*self.semi_major_axis*(1. - self.eccentricity**2.)/(self.star1.mass + self.star2.mass))**0.5

	def aFromJorb(self, Jorb):
		#equation taken from BSE (some constants in BSE that don't come up here?)
		#     sep = (mass(1) + mass(2))*jorb*jorb/
		# &         ((mass(1)*mass(2)*twopi)**2*aursun**3*(1.d0-ecc*ecc))
		return Jorb**2.*(self.star1.mass + self.star2.mass)/(self.units_holder.Gconst*(self.star1.mass*self.star2.mass)**2.*(1. - self.eccentricity**2.))

	def Eggleton_RL(self):
	#Eggleton (1983) Roche Lobe
	#assuming synchronous rotation
	#but taking the separation at pericenter
		a = self.semi_major_axis*(1. - self.eccentricity)

		q1 = self.star1.mass/self.star2.mass
		rl1 = a*0.49*q1**(2./3.)/(0.6*q1**(2./3.) + np.log(1. + q1**(1./3.))) 

		q2 = self.star2.mass/self.star1.mass
		rl2 = a*0.49*q2**(2./3.)/(0.6*q2**(2./3.) + np.log(1. + q2**(1./3.))) 

		return rl1, rl2

	def evolve(self, time):
		#this only advances the stars in stellar evolution
		if (self.bcm is not None):
			self.star1.evolve(time)
			self.star2.evolve(time)


class TidesIntegrator():
	def __init__(self, binary, units_holder = None):

		self.binary = binary
		self.end_time = None

		self.current_time = 0.
		self.TCfac = 1.
		self.fconvexp = 2.

		self.deTarget = 0.001

		self.result = Table()
		self.integrateSuccess = True

		if (units_holder is None):
			units_holder = unitsHolder()
		self.units_holder = units_holder

		self.MBfac = (-5.83*10**(-16)*units.solMass*units.yr/units.solRad).to(units_holder.mass_unit*units_holder.time_unit/units_holder.length_unit).value

		self.diffo10 = 1.
		self.diffo10 = 2.


		self.dadt = 0.
		self.dpdt = 0.
		self.dedt = 0.
		self.do1dt = 0.
		self.do2dt = 0.


	def KT_equilibrium(self, m, r, me, re, L, omega, a, q):
	#for equilibrium tides on stars with convective envelopes
	#following Hurley et al. 2002, MNRAS, 329, 897, who follow Rasio et al. (1996)
	#NOTE: in Hurley et al. 2002 (and BSE) there is a factor of 0.4311 in front of this eqn to convert the luminosity units (not needed here!)
		p = 2.*np.pi*(a**3./(self.units_holder.Gconst*m*(1. + q)))**0.5
		tconv = (me*re*(r - 0.5*re)/(3.*L))**(1./3.)
		ch = ((1e-10)*units.yr**-1).to(self.units_holder.time_unit**-1).value
		Ptid = 1./(ch + np.abs(1./p - omega/(2.*np.pi))) #1e-10 added as in BSE to ensure not dividing by zero
		fconv = np.min([1.,(Ptid/(2.*tconv))**self.fconvexp])
		return 2./21.*fconv/tconv*me/m*self.TCfac

	def Tcirc_equilibrium(self, m, r, me, re, L, omega, a, q):
		KT = self.KT_equilibrium(m, r, me, re, L, omega, a, q)
		return 1./(self.TCfac*21./2.*KT*q*(1. + q)*(r/a)**8.)


	def omega_eq(self, e, p):
	#eq 34 in Hurley et al. 2002
		f2 = 1. + 15./2.*e**2. +  45./8.*e**4. +   5./16.*e**6.
		f5 = 1. +     3.*e**2. +   3./8.*e**4.
		omega_orb = 2.*np.pi/p
		return f2*omega_orb/(f5*(1. - e**2.)**(3./2.))


	def Hut_odes(self, a, e, m, q, r, rg, re, me, lum, omega):
	#Hut 1981 A&A
	#the is adapted from my AMUSE code.  The comments below are from the Matsumura ODE; not sure if they are relevant here as well.

		#get tidal dissipation terms
		f1 = 1. + 31./2.*e**2. + 255./8.*e**4. + 185./16.*e**6. + 25./64.*e**8.
		f2 = 1. + 15./2.*e**2. +  45./8.*e**4. +   5./16.*e**6.
		f3 = 1. + 15./4.*e**2. +  15./8.*e**4. +   5./64.*e**6.
		f4 = 1. +  3./2.*e**2. +   1./8.*e**4.
		f5 = 1. +     3.*e**2. +   3./8.*e**4.

		KT = self.KT_equilibrium(m, r, me, re, lum, omega, a, q)

		n = (self.units_holder.Gconst*(m + m*q)/(a**3.))**(0.5) #2*pi/period
		oe = 1. - e**2.

		adot =  -6.*KT*q*(1. + q)*(r/a)**8.*a/oe**(15./2.)*(f1 -         oe**(3./2.)*f2*omega/n)
		edot = -27.*KT*q*(1. + q)*(r/a)**8.*e/oe**(13./2.)*(f3 - 11./18.*oe**(3./2.)*f4*omega/n)
		odot =   3.*KT*(q/rg)**2.*(r/a)**6.*n/oe**(6.)*    (f2 -         oe**(3./2.)*f5*omega/n)

		#print("inside", TF,eta,oe,omega, n, f2, f5)

		return adot,edot,odot

	def magneticBraking_ode(self, m, me, r, mI, omega):
	#magnetic braking from Hurley et al. (2002)
	#Note Hurley gives equation for Jdot and J = I*omega so omegadot = Jdot/I
		return self.MBfac*(me/m)*(r*omega)**3./mI

	def integrateTidesStep(self, dt=None):
		#binary is a class containing star1 and star2 (also classes) containing information about each stars and without units

		if (dt is None):
			dt = self.end_time - self.current_time

		if (self.binary.semi_major_axis > 0):

			#get the ODEs
			adot1,edot1,omegadot1 = self.Hut_odes(self.binary.semi_major_axis, self.binary.eccentricity, self.binary.star1.mass, self.binary.star2.mass/self.binary.star1.mass, self.binary.star1.radius, self.binary.star1.gyration_radius, self.binary.star1.convective_envelope_radius, self.binary.star1.convective_envelope_mass, self.binary.star1.luminosity, self.binary.star1.omega)
			adot2,edot2,omegadot2 = self.Hut_odes(self.binary.semi_major_axis, self.binary.eccentricity, self.binary.star2.mass, self.binary.star1.mass/self.binary.star2.mass, self.binary.star2.radius, self.binary.star2.gyration_radius, self.binary.star1.convective_envelope_radius, self.binary.star1.convective_envelope_mass, self.binary.star1.luminosity, self.binary.star2.omega)


			omegadotmb1 = 0.
			omegadotmb2 = 0.
			omegadotmb1 = self.magneticBraking_ode(self.binary.star1.mass, self.binary.star1.convective_envelope_mass, self.binary.star1.radius, self.binary.star1.omega, self.binary.star1.moment_of_inertia)
			omegadotmb2 = self.magneticBraking_ode(self.binary.star2.mass, self.binary.star2.convective_envelope_mass, self.binary.star2.radius, self.binary.star2.omega, self.binary.star2.moment_of_inertia)

			#limit the timestep
			de = np.abs((edot1 + edot2)*dt)
			if (de > self.deTarget):
				dt = self.deTarget/np.abs(edot1 + edot2)

			#ensure that the spin doesn't cross the equilibrium spin (as in BSE)
			#check which side of the equilibrium spin we are on
			oeq = self.omega_eq(self.binary.eccentricity, self.binary.period)
			o1 = self.binary.star1.omega + (omegadot1 + omegadotmb1)*dt
			o2 = self.binary.star2.omega + (omegadot2 + omegadotmb2)*dt
			diffo1 = np.sign(o1 - oeq)
			if (self.diffo10 != diffo1):
				#print('omega1 hitting eq limit', self.current_time, o1, oeq, self.diffo10, diffo1)
				o1 = oeq
			diffo2 = np.sign(o2 - oeq)
			if (self.diffo20 != diffo2):
				#print('omega2 hitting eq limit', self.current_time, o2, seq, self.diffo20, diffo2)
				o2 = oeq

			#print(self.current_time, self.diffo10, diffo1, oeq, o1)
			#print(start_time, dt, (edot1 + edot2)*dt, (adot1 + adot2)*dt)

			o10 = self.binary.star1.omega
			o20 = self.binary.star2.omega
			self.binary.star1.omega = o1
			self.binary.star2.omega = o2
			self.do1dt = (o10 - self.binary.star1.omega)/dt
			self.do2dt = (o20 - self.binary.star1.omega)/dt

			Jb = self.binary.Jorb()

			e0 = self.binary.eccentricity
			self.binary.eccentricity += (edot1 + edot2)*dt
			self.binary.eccentricity = np.clip(self.binary.eccentricity, 0., 1.)
			self.dedt = (e0 - self.binary.eccentricity)/dt

			#self.binary.semi_major_axis += (adot1 + adot2)*dt
			#define the semi-major axis based on angular momentum?
			Jdotspin = self.binary.star1.moment_of_inertia*(omegadot1 + omegadotmb1) + self.binary.star2.moment_of_inertia*(omegadot2 + omegadotmb2)
			Jb -= Jdotspin*dt
			a0 = self.binary.semi_major_axis
			self.binary.semi_major_axis = self.binary.aFromJorb(Jb)
			self.binary.semi_major_axis = np.max([0., self.binary.semi_major_axis])
			self.dadt = (a0 - self.binary.semi_major_axis)/dt

			p0 = self.binary.period
			self.binary.period = 2.*np.pi*(self.binary.semi_major_axis**3./(self.units_holder.Gconst*self.binary.star1.mass*(1. + self.binary.star2.mass/self.binary.star1.mass)))**0.5
			self.dpdt = (p0 - self.binary.period)/dt

		self.current_time += dt


	def evolve(self, end_time, dt = None):
		#check
		# Jorb = JorbBinary(binary)
		# print("check", binary.semi_major_axis, aFromJorb(binary, Jorb))#, Jorb, 
		# 	binary.star1.mass*binary.star2.mass/(binary.star1.mass + binary.star2.mass)*(1. - binary.eccentricity**2.)**0.5*binary.semi_major_axis**2.*(2.*np.pi/binary.period), 
		# 	binary.star1.mass*binary.star2.mass/(binary.star1.mass + binary.star2.mass)**0.5*(1. - binary.eccentricity**2.)**0.5*binary.semi_major_axis**0.5*Gconst**0.5
		# 	)


		self.end_time = end_time.to(self.units_holder.time_unit).value
		if (dt is not None):
			dt = dt.to(self.units_holder.time_unit).value

		self.current_time = 0.

		t_out = [self.current_time]
		e_out = [self.binary.eccentricity]
		a_out = [self.binary.semi_major_axis]
		p_out = [self.binary.period]
		o1_out = [self.binary.star1.omega]
		o2_out = [self.binary.star2.omega]
		dadt_out = [self.dadt]
		dedt_out = [self.dedt]
		dpdt_out = [self.dpdt]
		do1dt_out = [self.do1dt]
		do2dt_out = [self.do2dt]

		oeq = self.omega_eq(self.binary.eccentricity, self.binary.period)
		self.diffo10 = np.sign(self.binary.star1.omega - oeq)
		self.diffo20 = np.sign(self.binary.star2.omega - oeq)

		while self.current_time < self.end_time:

			self.binary.evolve(self.current_time*self.units_holder.time_unit)
			self.integrateTidesStep(dt = dt)

			#check for a merger
			if (self.binary.semi_major_axis > 0):
				rl1, rl2 = self.binary.Eggleton_RL()
				if (self.binary.star1.radius/rl1 > 1 or self.binary.star2.radius/rl2 > 1):
					self.binary.semi_major_axis = 0
					self.binary.period = 0
					self.binary.eccentricity = 0


			if (self.integrateSuccess):

				t_out.append(self.current_time)
				e_out.append(self.binary.eccentricity)
				a_out.append(self.binary.semi_major_axis)
				p_out.append(self.binary.period)
				o1_out.append(self.binary.star1.omega)
				o2_out.append(self.binary.star2.omega)
				dadt_out.append(self.dadt)
				dedt_out.append(self.dedt)
				dpdt_out.append(self.dpdt)
				do1dt_out.append(self.do1dt)
				do2dt_out.append(self.do2dt)
			else:
				print('**INTEGRATION FAILED', self.current_time*self.units_holder.time_unit.to(units.Myr))
				break

		#format the output
		self.result['tphys'] = np.array(t_out)*self.units_holder.time_unit
		self.result['ecc'] = np.array(e_out)
		self.result['sma'] = np.array(a_out)*self.units_holder.length_unit
		self.result['porb'] = np.array(p_out)*self.units_holder.time_unit
		self.result['omega1'] = np.array(o1_out)/self.units_holder.time_unit
		self.result['omega2'] = np.array(o2_out)/self.units_holder.time_unit
		self.result['dadt'] = np.array(dadt_out)*self.units_holder.length_unit/self.units_holder.time_unit
		self.result['dedt'] = np.array(dedt_out)/self.units_holder.time_unit
		self.result['dpdt'] = np.array(dpdt_out)*self.units_holder.time_unit/self.units_holder.time_unit
		self.result['do1dt'] = np.array(do1dt_out)/self.units_holder.time_unit**2.
		self.result['do2dt'] = np.array(do2dt_out)/self.units_holder.time_unit**2.


def initBinary(m1 = 1*units.solMass, 
			   m2 = 1*units.solMass, 
			   r1 = 1*units.solRad,
			   r2 = 1*units.solRad,
			   L1 = 1*units.solLum,
			   L2 = 1*units.solLum,
			   p = 8*units.day, 
			   e = 0.8,
			   m1e = None,
			   m2e = None,
			   r1e = None,
			   r2e = None,
			   m1c = None,
			   m2c = None,
			   r1c = None,
			   r2c = None,
			   omega1 = None, 
			   omega2 = None, 
			   units_holder = None,
			   bcm = None):

	if (units_holder is None):
		units_holder = unitsHolder()

	#units defined in unitsHolder

	#gyration radius^2 is I/(M*R^2)
	#I think this is sometimes called the moment of inertia factor
	#Moment of inertia factor of the Sun, I = 0.070 from https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
	#is the gyration radius unitless?

	#current Sun's spin = 1./(27.*units.day),
	#but start synchronized here


	if (bcm is not None):
		m1 = bcm['mass_1'].data[0]*units.solMass
		m2 = bcm['mass_2'].data[0]*units.solMass
		m1e = bcm['menv_1'].data[0]*units.solMass
		m2e = bcm['menv_2'].data[0]*units.solMass
		m1c = bcm['massc_1'].data[0]*units.solMass
		m2c = bcm['massc_2'].data[0]*units.solMass
		r1 = bcm['rad_1'].data[0]*units.solRad
		r2 = bcm['rad_2'].data[0]*units.solRad
		r1e = bcm['renv_1'].data[0]*units.solRad
		r2e = bcm['renv_2'].data[0]*units.solRad
		r1c = bcm['radc_1'].data[0]*units.solRad
		r2c = bcm['radc_2'].data[0]*units.solRad
		omega1 = bcm['omega_spin_1'].data[0]*units.yr**-1
		omega2 = bcm['omega_spin_2'].data[0]*units.yr**-1
		L1 = bcm['lum_1'].data[0]*units.solLum
		L2 = bcm['lum_2'].data[0]*units.solLum

	a = ((((p/(2.*np.pi))**2.*constants.G*(m1 + m2)))**(1./3.)).decompose().to(units.AU)
	print('binary semi-major axis = ',a)

	if (omega1 is None):
		omega1 = 2*np.pi/p
		#omega1 = omega_eq(e, p.to(time_unit).value)*time_unit**-1
	if (omega2 is None):
		omega2 = 2*np.pi/p
		#omega2 = omega_eq(e, p.to(time_unit).value)*time_unit**-1

	print('binary omega1 = ',omega1, omega1.to(units_holder.time_unit**-1).value)

	star1 = Star(
		mass = m1,
		radius = r1,
		luminosity = L1,
		convective_envelope_mass = m1e, 
		convective_envelope_radius = r1e, 
		omega = omega1,
		units_holder = units_holder
		)

	star2 = Star(
		mass = m2,
		radius = r2,
		luminosity = L2,
		convective_envelope_mass = m2e, 
		convective_envelope_radius = r2e,
		omega = omega2,
		units_holder = units_holder
		)

	binary = Binary(
		star1 = star1,
		star2 = star2,
		eccentricity = e,
		semi_major_axis = a,
		units_holder = units_holder,
		bcm = bcm
	)


	return binary




if __name__=="__main__":

	binary = initBinary()
	integrator = TidesIntegrator(binary)
	integrator.evolve(10*units.Gyr)
	print('t = ',integrator.result['tphys'])
	print('e = ',integrator.result['ecc'])
	print('a = ',integrator.result['sma'])
