import numpy as np
from astropy.table import Table
from scipy import integrate
from astropy import units,constants

time_unit = units.yr 
length_unit = units.solRad
mass_unit = units.solMass

Gconst = constants.G.to(length_unit**3*mass_unit**-1*time_unit**-2).value

MBfac = (-5.83*10**(-16)*units.solMass*units.yr/units.solRad).to(mass_unit*time_unit/length_unit).value


def angularMomentum(binary):
	Jspin1 = binary.star1.moment_of_inertia*binary.star1.omega
	Jspin2 = binary.star2.moment_of_inertia*binary.star2.omega

	return JorbBinary(binary) + Jspin1 + Jspin2

def JorbBinary(binary):
	#equation taken from BSE (matches)
	#    oorb = twopi/tb
	#    jorb = mass(1)*mass(2)/(mass(1)+mass(2))
	# &          *SQRT(1.d0-ecc*ecc)*sep*sep*oorb
	return binary.star1.mass*binary.star2.mass*(Gconst*binary.semi_major_axis*(1. - binary.eccentricity**2.)/(binary.star1.mass + binary.star2.mass))**0.5

def aFromJorb(binary, Jorb):
	#equation taken from BSE (some constants in BSE that don't come up here?)
	#     sep = (mass(1) + mass(2))*jorb*jorb/
	# &         ((mass(1)*mass(2)*twopi)**2*aursun**3*(1.d0-ecc*ecc))
	return Jorb**2.*(binary.star1.mass + binary.star2.mass)/(Gconst*(binary.star1.mass*binary.star2.mass)**2.*(1. - binary.eccentricity**2.))

def KT_equilibrium(m, r, me, re, L, omega, a, q, fconvexp=2., TCfac=1.):
#for equilibrium tides on stars with convective envelopes
#following Hurley et al. 2002, MNRAS, 329, 897, who follow Rasio et al. (1996)
#NOTE: in Hurley et al. 2002 (and BSE) there is a factor of 0.4311 in front of this eqn to convert the luminosity units (not needed here!)
	p = 2.*np.pi*(a**3./(Gconst*m*(1. + q)))**0.5
	tconv = (me*re*(r - 0.5*re)/(3.*L))**(1./3.)
	ch = ((1e-10)*units.yr**-1).to(time_unit**-1).value
	Ptid = 1./(ch + np.abs(1./p - omega/(2.*np.pi))) #1e-10 added as in BSE to ensure not dividing by zero
	fconv = np.min([1.,(Ptid/(2.*tconv))**fconvexp])
	return 2./21.*fconv/tconv*me/m*TCfac

def Tcirc_equilibrium(m, r, me, re, L, omega, a, q, fconvexp=2., TCfac=1.):
	KT = KT_equilibrium(m, r, me, re, L, omega, a, q, fconvexp=fconvexp, TCfac=TCfac)
	return 1./(TCfac*21./2.*KT*q*(1. + q)*(r/a)**8.)

def omega_eq(e, p):
#eq 34 in Hurley et al. 2002
	f2 = 1. + 15./2.*e**2. +  45./8.*e**4. +   5./16.*e**6.
	f5 = 1. +     3.*e**2. +   3./8.*e**4.
	omega_orb = 2.*np.pi/p
	return f2*omega_orb/(f5*(1. - e**2.)**(3./2.))

def moment_of_inertia(m, r, rg, mc=0., rc=0.):
#found in eq 35 of Hurley et al 2002.  (previously I had used rg for this.  Maybe I should here too?)
#mc and rc are core mass and radius
#with mc=rc=0, this reduces to the usual formula for moment of inertia
	k2p = 0.1
	k3p = 0.21
	# return rg**2.*(m - mc)*r**2. + k3p*mc*rc**2.
	return k2p*(m - mc)*r**2. + k3p*mc*rc**2.

def Hut_odes(a, e, m, q, r, rg, re, me, lum, omega, TCfac=1):
#Hut 1981 A&A
#the is adapted from my AMUSE code.  The comments below are from the Matsumura ODE; not sure if they are relevant here as well.

	#get tidal dissipation terms
	f1 = 1. + 31./2.*e**2. + 255./8.*e**4. + 185./16.*e**6. + 25./64.*e**8.
	f2 = 1. + 15./2.*e**2. +  45./8.*e**4. +   5./16.*e**6.
	f3 = 1. + 15./4.*e**2. +  15./8.*e**4. +   5./64.*e**6.
	f4 = 1. +  3./2.*e**2. +   1./8.*e**4.
	f5 = 1. +     3.*e**2. +   3./8.*e**4.

	KT = KT_equilibrium(m, r, me, re, lum, omega, a, q, TCfac=TCfac)

	n = (Gconst*(m + m*q)/(a**3.))**(0.5) #2*pi/period
	oe = 1. - e**2.

	adot =  -6.*KT*q*(1. + q)*(r/a)**8.*a/oe**(15./2.)*(f1 -         oe**(3./2.)*f2*omega/n)
	edot = -27.*KT*q*(1. + q)*(r/a)**8.*e/oe**(13./2.)*(f3 - 11./18.*oe**(3./2.)*f4*omega/n)
	odot =   3.*KT*(q/rg)**2.*(r/a)**6.*n/oe**(6.)*    (f2 -         oe**(3./2.)*f5*omega/n)

	#print("inside", TF,eta,oe,omega, n, f2, f5)

	return adot,edot,odot

def magneticBraking_ode(m, me, r, rg, omega):
#magnetic braking from Hurley et al. (2002)
#Note Hurley gives equation for Jdot and J = I*omega so omegadot = Jdot/I
	return MBfac*(me/m)*(r*omega)**3./moment_of_inertia(m, r, rg)

def integrateTidesStep(binary, start_time, end_time,
					nsteps=1e6, rtol=1e-8, use_vode=False, use_lsoda=False, TCfac=1., deTarget=0.001):
	#binary is a class containing star1 and star2 (also classes) containing information about each stars and without units

	success = True

	dt = end_time - start_time

	if (binary.semi_major_axis > 0):

		#get the ODEs
		adot1,edot1,omegadot1 = Hut_odes(binary.semi_major_axis, binary.eccentricity, binary.star1.mass, binary.star2.mass/binary.star1.mass, binary.star1.radius, binary.star1.gyration_radius, binary.star1.convective_envelope_radius, binary.star1.convective_envelope_mass, binary.star1.luminosity, binary.star1.omega, TCfac=TCfac)
		adot2,edot2,omegadot2 = Hut_odes(binary.semi_major_axis, binary.eccentricity, binary.star2.mass, binary.star1.mass/binary.star2.mass, binary.star2.radius, binary.star2.gyration_radius, binary.star1.convective_envelope_radius, binary.star1.convective_envelope_mass, binary.star1.luminosity, binary.star2.omega, TCfac=TCfac)


		omegadotmb1 = 0.
		omegadotmb2 = 0.
		omegadotmb1 = magneticBraking_ode(binary.star1.mass, binary.star1.convective_envelope_mass, binary.star1.radius, binary.star1.gyration_radius, binary.star1.omega)
		omegadotmb2 = magneticBraking_ode(binary.star2.mass, binary.star2.convective_envelope_mass, binary.star2.radius, binary.star2.gyration_radius, binary.star2.omega)

		#limit the timestep
		de = np.abs((edot1 + edot2)*dt)
		if (de > deTarget):
			dt = deTarget/np.abs(edot1 + edot2)

		#ensure that the spin doesn't cross the equilibrium spin (as in BSE)
		#check which side of the equilibrium spin we are on
		oeq = omega_eq(binary.eccentricity, binary.period)
		diffo10 = np.sign(binary.star1.omega - oeq)
		diffo20 = np.sign(binary.star2.omega - oeq)
		o1 = binary.star1.omega + (omegadot1 + omegadotmb1)*dt
		o2 = binary.star2.omega + (omegadot2 + omegadotmb2)*dt
		diffo1 = np.sign(o1 - oeq)
		if (diffo10 != diffo1):
			#print('omega1 hitting eq limit', start_time, o1, oeq, diffo10, diffo1)
			o1 = oeq
		diffo2 = np.sign(o2 - oeq)
		if (diffo20 != diffo2):
			#print('omega2 hitting eq limit', start_time, o2, seq, diffo20, diffo2)
			o2 = oeq

		#print(start_time, dt, (edot1 + edot2)*dt, (adot1 + adot2)*dt)

		do1 = binary.star1.omega - o1
		do2 = binary.star1.omega - o2
		binary.star1.omega = o1
		binary.star2.omega = o2

		Jb = JorbBinary(binary)

		binary.eccentricity += (edot1 + edot2)*dt
		binary.eccentricity = np.clip(binary.eccentricity, 0., 1.)

		#define the semi-major axis based on angular momentum?
		Jdotspin = binary.star1.moment_of_inertia*(omegadot1 + omegadotmb1) + binary.star2.moment_of_inertia*(omegadot2 + omegadotmb2)
		Jb -= Jdotspin*dt
		da1 = binary.semi_major_axis - aFromJorb(binary, Jb)
		da2 = (adot1 + adot2)*dt
		binary.semi_major_axis = aFromJorb(binary, Jb)

		#print(start_time, dt, da1, da2, da1/da2)
		#print(start_time, dt, (omegadot1 + omegadotmb1), JorbBinary(binary), Jdotspin, JorbBinary(binary)/(Jdotspin*dt))


		#binary.semi_major_axis += (adot1 + adot2)*dt
		binary.semi_major_axis = np.max([0., binary.semi_major_axis])

		binary.period = 2.*np.pi*(binary.semi_major_axis**3./(Gconst*binary.star1.mass*(1. + binary.star2.mass/binary.star1.mass)))**0.5



		return success, start_time + dt


	# success = True

	# X0 = np.array([binary.eccentricity, binary.star1.spin, binary.star2.spin, binary.semi_major_axis])
# 	#use the ODEs from Hut 1981
# 	def dX_dt(t,X):
# 		e = X[0]
# 		s1 = X[1]
# 		s2 = X[2]
# 		a = X[3]

# 		#safety check
# 		# if (X[0] < 0): X[0] = 0.
# 		# if (X[1] < 0): X[1] = 0.
# 		# if (X[2] < 0): X[2] = 0.
# 		# if (X[3] < 0): X[3] = 0.

# 		#check for RLOF or merger?

# 		#ensure that the spin doesn't cross the equilibrium spin (as in BSE)
# 		seq = spin_eq(binary.eccentricity, binary.period)
# 		diffs1 = np.sign(s1 - seq)
# 		if (diffs10 != diffs1):
# 			#print('spin1 hitting eq limit', current_time, s1, seq, diffs10, diffs1)
# 			s1 = seq
# 		diffs2 = np.sign(s2 - seq)
# 		if (diffs20 != diffs2):
# 			#print('spin2 hitting eq limit', current_time, s2, seq, diffs20, diffs2)
# 			s2 = seq
# 		#print(current_time, diffs10, diffs1, diffs20, diffs2)



# 		adot1,edot1,omegadot1 = Hut_odes(a, e, binary.star1.mass, binary.star2.mass/binary.star1.mass, binary.star1.radius, binary.star1.gyration_radius, binary.star1.convective_envelope_radius, binary.star1.convective_envelope_mass, binary.star1.luminosity, s1, TCfac=TCfac)
# 		adot2,edot2,omegadot2 = Hut_odes(a, e, binary.star2.mass, binary.star1.mass/binary.star2.mass, binary.star2.radius, binary.star2.gyration_radius, binary.star1.convective_envelope_radius, binary.star1.convective_envelope_mass, binary.star1.luminosity, s2, TCfac=TCfac)


# 		omegadotmb1 = 0.
# 		omegadotmb2 = 0.
# 		omegadotmb1 = magneticBraking_ode(binary.star1.mass, binary.star1.convective_envelope_mass, binary.star1.radius, binary.star1.gyration_radius, s1)
# 		omegadotmb2 = magneticBraking_ode(binary.star2.mass, binary.star2.convective_envelope_mass, binary.star2.radius, binary.star2.gyration_radius, s2)


# #set up the array of ode's

# 		#print(X[1], X[0], binary.star1.mass, binary.star2.mass, binary.star1.radius, binary.star2.radius, binary.star1.gyration_radius, binary.star2.gyration_radius, binary.star1.convective_envelope_radius, binary.star2.convective_envelope_radius, binary.star1.convective_envelope_mass, binary.star2.convective_envelope_mass, binary.star1.luminosity, binary.star2.luminosity, binary.star1.spin, binary.star2.spin,edot1, edot2, adot1, adot2, omegadot1, omegadot2)

# 		#print(t, edot1, edot2, adot1, adot2, omegadot1, omegadot2, X)
# 		#print(t, edot1, edot2, omegadot1, omegadot2, omegadotmb1, omegadotmb2)

# 		return np.array([edot1 + edot2, omegadot1 + omegadotmb1, omegadot2 + omegadotmb2, adot1 + adot2])


# 	if (use_vode):
# 		odeint = integrate.ode(dX_dt).set_integrator('vode', nsteps=nsteps, rtol=rtol, method='bdf').set_initial_value(X0, start_time)
# 	elif (use_lsoda):
# 		odeint = integrate.ode(dX_dt).set_integrator('lsoda', nsteps=nsteps, rtol=rtol).set_initial_value(X0, start_time)

# 	else:
# 		odeint = integrate.ode(dX_dt).set_integrator('dopri5', nsteps=nsteps, rtol=rtol).set_initial_value(X0, start_time)

# 	odeint.integrate(start_time + dt)

# 	if (not odeint.successful()):
# 		print("**WARNING: odeint was NOT successful")


# 	return odeint.y, odeint.successful()


def integrateTides(binary, end_time, TCfac=1):

	#check
	# Jorb = JorbBinary(binary)
	# print("check", binary.semi_major_axis, aFromJorb(binary, Jorb))#, Jorb, 
	# 	binary.star1.mass*binary.star2.mass/(binary.star1.mass + binary.star2.mass)*(1. - binary.eccentricity**2.)**0.5*binary.semi_major_axis**2.*(2.*np.pi/binary.period), 
	# 	binary.star1.mass*binary.star2.mass/(binary.star1.mass + binary.star2.mass)**0.5*(1. - binary.eccentricity**2.)**0.5*binary.semi_major_axis**0.5*Gconst**0.5
	# 	)

	current_time = 0.

	t_out = [current_time]
	e_out = [binary.eccentricity]
	a_out = [binary.semi_major_axis]
	p_out = [binary.period]
	o1_out = [binary.star1.omega]
	o2_out = [binary.star2.omega]
	while current_time < end_time:

		e0 = binary.eccentricity


		success, t = integrateTidesStep(binary, current_time, end_time, TCfac=TCfac)

		if (success):

			current_time = t

			t_out.append(current_time)
			e_out.append(binary.eccentricity)
			a_out.append(binary.semi_major_axis)
			p_out.append(binary.period)
			o1_out.append(binary.star1.omega)
			o2_out.append(binary.star2.omega)


		else:
			print('**INTEGRATION FAILED', current_time*time_unit.to(units.Myr))
			break



	#format the output
	tab = Table()
	tab['tphys'] = np.array(t_out)*time_unit
	tab['ecc'] = np.array(e_out)
	tab['sma'] = np.array(a_out)*length_unit
	tab['porb'] = np.array(p_out)*time_unit
	tab['omega1'] = np.array(o1_out)/time_unit
	tab['omega2'] = np.array(o2_out)/time_unit

	return tab



class Star():
	#this should be sent values with units, and will return unitless values in the units from above
	def __init__(self, 
		mass, 
		radius, 
		luminosity, 
		convective_envelope_mass, 
		convective_envelope_radius,
		omega, #rad/time
		gyration_radius):

		self.mass = mass.to(mass_unit).value
		self.radius = radius.to(length_unit).value
		self.luminosity = luminosity.to(mass_unit*length_unit**2*time_unit**-3).value
		self.convective_envelope_mass = convective_envelope_mass.to(mass_unit).value
		self.convective_envelope_radius = convective_envelope_radius.to(length_unit).value
		self.omega = omega.to(1./time_unit).value
		self.gyration_radius = gyration_radius
		self.moment_of_inertia = moment_of_inertia(self.mass, self.radius, self.gyration_radius)
		
class Binary():
	#this should be sent values with units, and will return unitless values in the units from above
	#star1 and star2 should be of the star class from above
	def __init__(self,
		star1,
		star2,
		semi_major_axis,
		eccentricity):

		self.star1 = star1
		self.star2 = star2
		self.semi_major_axis = semi_major_axis.to(length_unit).value
		self.period = 2.*np.pi*(self.semi_major_axis**3./(Gconst*(self.star1.mass + self.star2.mass)))**0.5
		self.eccentricity = eccentricity


def initBinary(m1 = 1*units.solMass, m2 = 1*units.solMass, p = 8*units.day, e = 0.8, omega1 = None, omega2 = None,
				t_unit = time_unit,
				l_unit = length_unit,
				m_unit = mass_unit):

	global time_unit, length_unit, mass_unit, Gconst, MBfac

	time_unit = t_unit
	length_unit = l_unit
	mass_unit = m_unit

	Gconst = constants.G.to(length_unit**3*mass_unit**-1*time_unit**-2).value
	MBfac = (-5.83*10**(-16)*units.solMass*units.yr/units.solRad).to(mass_unit*time_unit/length_unit).value

	print("Gconst", Gconst)
	#units defined above

	#gyration radius^2 is I/(M*R^2)
	#I think this is sometimes called the moment of inertia factor
	#Moment of inertia factor of the Sun, I = 0.070 from https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
	#is the gyration radius unitless?

	#current Sun's spin = 1./(27.*units.day),
	#but start synchronized here


	a = ((((p/(2.*np.pi))**2.*constants.G*(m1 + m2)))**(1./3.)).decompose().to(units.AU)
	print('binary semi-major axis = ',a)

	if (omega1 is None):
		omega1 = 2*np.pi/p
		#omega1 = omega_eq(e, p.to(time_unit).value)*time_unit**-1
	if (omega2 is None):
		omega2 = 2*np.pi/p
		#omega2 = omega_eq(e, p.to(time_unit).value)*time_unit**-1

	print('binary omega1 = ',omega1, omega1.to(time_unit**-1).value)

	star1 = Star(
		mass = m1,
		radius = 1.*units.solRad,
		luminosity = 1.*units.solLum,
		convective_envelope_radius = 0.230702*units.solRad, #from BSE
		convective_envelope_mass = 0.032890*units.solMass, #from BSE
		omega = omega1,
		gyration_radius = 0.07**0.5
		)

	star2 = Star(
		mass = m2,
		radius = 1.*units.solRad,
		luminosity = 1.*units.solLum,
		convective_envelope_radius = 0.230702*units.solRad, #from BSE
		convective_envelope_mass = 0.032890*units.solMass, #from BSE
		omega = omega2,
		gyration_radius = 0.07**0.5
		)

	binary = Binary(
		star1 = star1,
		star2 = star2,
		eccentricity = e,
		semi_major_axis = a
	)


	return binary

def evolve(binary, tf, TCfac=1):

	adot1,edot1,omegadot1 = Hut_odes(binary.semi_major_axis, binary.eccentricity, binary.star1.mass, binary.star2.mass/binary.star1.mass, binary.star1.radius, binary.star1.gyration_radius, binary.star1.convective_envelope_radius, binary.star1.convective_envelope_mass, binary.star1.luminosity, binary.star1.omega)
	adot2,edot2,omegadot2 = Hut_odes(binary.semi_major_axis, binary.eccentricity, binary.star2.mass, binary.star1.mass/binary.star2.mass, binary.star2.radius, binary.star2.gyration_radius, binary.star1.convective_envelope_radius, binary.star1.convective_envelope_mass, binary.star1.luminosity, binary.star2.omega)

	ODE0 = np.array([edot1 + edot2, adot1 + adot2, omegadot1, omegadot2])

	tab = integrateTides(binary, tf.to(time_unit).value, TCfac=TCfac)

	return ODE0, tab


if __name__=="__main__":

	binary = initBinary()
	ODE0, tab = evolve(binary, 0, 1e6, 1e5)
	print(ODE0)
	print('t = ',tab['tphys'])
	print('e = ',tab['ecc'])
	print('a = ',tab['sma'])
