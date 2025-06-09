from ascfd.grid import Grid2D
from ascfd.constants import Constants
import ascfd.ics as ics
from ascfd.ebs import EmbeddedBoundaries

import glob
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import copy

import sys
from ascfd.flux import Flux

from ascfd.euler import Euler

from ascfd.bcs import BoundaryConditions

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import os


class Simulation:
    def __init__(self, a_inputs):
        self.inp = a_inputs
        self.c = Constants(a_inputs)
        self.euler = Euler(self.c)

        self.grid = Grid2D(self.inp.xlim, self.inp.ylim, self.inp.nx,
                           self.inp.ny, self.inp.numghosts, self.c.NUMQ)
        self.bcs = BoundaryConditions(
            self.grid, self.inp.bcs_lo, self.inp.bcs_hi, self.c)
        self.ebs = EmbeddedBoundaries(self.grid, self.c)
        self.flux = Flux(self.c, self.inp.flux)

        self.apply_ics()

        self.bcs.apply_bcs()
        self.grid.check_grid(self.c)
        # setup initial time to be the starting time from the inputs file.
        # The starting timestep will always be 0.
        self.t = self.inp.t0
        self.timestepNum = 0

        # -1 is no output. Always output ICs if we are outputting.
        if self.inp.output_freq >= 0:
            self.output()

    def run(self):
        
        # TODO: add eb parsing from .txt or .dat (use airfoil .dat as example!)
        # BIG r = 0.25 CIRCLE CENTERED AT (0.3, 0.5)
        # vertices = [(0.55, 0.5), (0.5487510413195065, 0.524958354161707), (0.5450166444603104, 0.5496673326987653), (0.5388341222814015, 0.5738800516653348), (0.5302652485007213, 0.5973545855771626), (0.5193956404725932, 0.6198563846510508), (0.5063339037274196, 0.6411606183487588), (0.4912105468211221, 0.6610544218094228), (0.47417667733679136, 0.6793390227248807), (0.45540249206766614, 0.6958317274068708), (0.43507557646703493, 0.7103677462019741), (0.41339903035639436, 0.7228018400153589), (0.3905894386191684, 0.7330097714918066), (0.36687470715614684, 0.7408895463542983), (0.34249178572506017, 0.7463624324971151), (0.31768430041692564, 0.7493737466510136), (0.2927001194246777, 0.7498934007603762), (0.2677888764261187, 0.7479162026131172), (0.24319947632672811, 0.7434619077195488), (0.219177608284124, 0.7365750219218535), (0.1959632908632143, 0.7273243567064204), (0.1737884738500355, 0.7158023416622183), (0.15287472068616342, 0.7021241009548974), (0.1334309946800438, 0.68642630304418), (0.11565157111468849, 0.6688657951377877), (0.09971409611326643, 0.6496180360259889), (0.08577781165776305, 0.6288753429553658), (0.0739819644957346, 0.6068449700584573), (0.06444441483283536, 0.5837470375389759), (0.0572604587126023, 0.5598123323034953), (0.05250187584988858, 0.5352800020149665), (0.05021621243168012, 0.5103951656083223), (0.05042630605131174, 0.4854064641431046), (0.053130057522783825, 0.4605635764641875), (0.058300451855134855, 0.43611472449329175), (0.06588582817730107, 0.41230419307759464), (0.07581039591646344, 0.38936988917628645), (0.0879749920723982, 0.3675409647728763), (0.10225807202139614, 0.34703552726431985), (0.11851692394996532, 0.3280584602040062), (0.13658909478409734, 0.31079937617301767), (0.15629401336668305, 0.29543072223389716), (0.1774347946648253, 0.28210605689660284), (0.19980020698000633, 0.2709585158126362), (0.22316678250539515, 0.26209948152762097), (0.24730105014230508, 0.25561747058372575), (0.27196186826623625, 0.2515772490916339), (0.29690283413427715, 0.2500191856089748), (0.32187474585986137, 0.25095884779103983), (0.3466280923556435, 0.25438684684391677), (0.37091554636580615, 0.26026893133421525), (0.3944944356782446, 0.2685463294180667), (0.4171291678250937, 0.27913633606996135), (0.43859358404478965, 0.2919331394440243), (0.45867321898565794, 0.30680887811100266), (0.4771674435728144, 0.3236149186074014), (0.4938914696275618, 0.3421833405319189), (0.5086781962097893, 0.36232861435058966), (0.5213798792353292, 0.38384945514655966), (0.5318696076860084, 0.40653083379243976), (0.5400425716625912, 0.43014612545026726), (0.5458171096106459, 0.4544593739319747), (0.5491355242558043, 0.47922764929562434)]
        
        # SMALL r = 0.12 CIRCLE CENTERED AT (0.3, 0.5)
        vertices = [(0.42, 0.5), (0.4194004998333631, 0.5119800099976194), (0.417607989340949, 0.5238403196954073), (0.4146403786950727, 0.5354624247993608), (0.4105273192803462, 0.5467302010770381), (0.4053099074268447, 0.5575310646325043), (0.3990402737891614, 0.5677570968074043), (0.3917810624741386, 0.577306122468523), (0.38360480512165984, 0.5860827309079427), (0.3745931961924797, 0.593999229155298), (0.36483627670417673, 0.6009765181769475), (0.3544315345710693, 0.6069448832073723), (0.3434829305372008, 0.6118446903160671), (0.33209985943495046, 0.6156269822500632), (0.3203960571480289, 0.6182539675986152), (0.3084884642001243, 0.6196993983924866), (0.2964960573238453, 0.6199488323649807), (0.284538660684537, 0.6189997772542962), (0.2727357486368295, 0.6168617157053834), (0.2612052519763795, 0.6135560105224898), (0.25006237961434286, 0.6091156912190818), (0.23941846744801704, 0.6035851239978648), (0.22937986592935844, 0.5970195684583508), (0.220046877446421, 0.5894846254612064), (0.21151275413505047, 0.581055581666138), (0.20386276613436788, 0.5718166572924747), (0.19717334959572624, 0.5618601646185756), (0.1915113429579526, 0.5512855856280595), (0.18693331911976097, 0.5401985780187084), (0.1834850201820491, 0.5287099195056777), (0.1812009004079465, 0.5169344009671839), (0.18010378196720644, 0.5049896794919947), (0.18020462690462963, 0.49299510278869024), (0.18150242761093624, 0.48107051670281), (0.18398421689046474, 0.46933506775678), (0.1876251975251045, 0.4579060126772454), (0.19238899003990245, 0.4468975468046175), (0.19822799619475112, 0.4364196630909806), (0.20508387457027016, 0.42657705308687355), (0.21288812349598335, 0.417468060897923), (0.22156276549636672, 0.4091837005630485), (0.23102112641600786, 0.4018067466722706), (0.24116870143911615, 0.3954109073103694), (0.25190409935040303, 0.3900600875900654), (0.26312005560258966, 0.38580775113325805), (0.27470450406830643, 0.3826963858801884), (0.2865416967677934, 0.3807570795639843), (0.29851336038445303, 0.3800092090923079), (0.31049987801273343, 0.3804602469396991), (0.32238148433070885, 0.3821056864850801), (0.33403946225558695, 0.38492908704042333), (0.3453573291255574, 0.38890223812067204), (0.35622200055604497, 0.3939854413135815), (0.366524920341499, 0.40012790693313166), (0.3761631451131158, 0.40726826149328127), (0.38504037291495086, 0.41533516093155265), (0.39306790542122966, 0.4242480034553211), (0.4001655341806989, 0.43391773488828306), (0.40626234203295797, 0.44424773847034865), (0.411297411689284, 0.4551348002203711), (0.41522043439804374, 0.4664701402161283), (0.41799221261311004, 0.4781404994873479), (0.41958505164278603, 0.4900292716618997)]        
        
        # FOR (200, 200) RESOLUTION GRID: SMALL r = 0.25 CIRCLE CENTERED AT (0.5, 1)
        # vertices = [(0.75, 1.0), (0.7487510413195064, 1.024958354161707), (0.7450166444603104, 1.0496673326987653), (0.7388341222814014, 1.0738800516653348), (0.7302652485007213, 1.0973545855771627), (0.7193956404725932, 1.1198563846510508), (0.7063339037274196, 1.1411606183487588), (0.6912105468211222, 1.1610544218094228), (0.6741766773367914, 1.1793390227248808), (0.6554024920676661, 1.1958317274068708), (0.6350755764670349, 1.2103677462019742), (0.6133990303563943, 1.2228018400153589), (0.5905894386191683, 1.2330097714918065), (0.5668747071561469, 1.2408895463542982), (0.5424917857250602, 1.246362432497115), (0.5176843004169257, 1.2493737466510135), (0.49270011942467773, 1.2498934007603764), (0.4677888764261187, 1.2479162026131172), (0.4431994763267281, 1.2434619077195488), (0.419177608284124, 1.2365750219218536), (0.3959632908632143, 1.2273243567064203), (0.37378847385003555, 1.2158023416622183), (0.35287472068616343, 1.2021241009548975), (0.3334309946800438, 1.18642630304418), (0.3156515711146885, 1.1688657951377877), (0.29971409611326644, 1.1496180360259889), (0.28577781165776306, 1.1288753429553657), (0.27398196449573464, 1.1068449700584573), (0.2644444148328354, 1.083747037538976), (0.2572604587126023, 1.0598123323034954), (0.2525018758498886, 1.0352800020149664), (0.25021621243168013, 1.0103951656083223), (0.2504263060513118, 0.9854064641431046), (0.25313005752278384, 0.9605635764641876), (0.25830045185513484, 0.9361147244932918), (0.2658858281773011, 0.9123041930775946), (0.27581039591646345, 0.8893698891762865), (0.2879749920723982, 0.8675409647728762), (0.30225807202139615, 0.8470355272643199), (0.31851692394996534, 0.8280584602040062), (0.3365890947840974, 0.8107993761730177), (0.3562940133666831, 0.7954307222338972), (0.3774347946648253, 0.7821060568966028), (0.39980020698000635, 0.7709585158126362), (0.4231667825053952, 0.762099481527621), (0.4473010501423051, 0.7556174705837257), (0.47196186826623626, 0.7515772490916339), (0.49690283413427716, 0.7500191856089748), (0.5218747458598614, 0.7509588477910398), (0.5466280923556435, 0.7543868468439168), (0.5709155463658061, 0.7602689313342152), (0.5944944356782447, 0.7685463294180668), (0.6171291678250936, 0.7791363360699614), (0.6385935840447896, 0.7919331394440243), (0.658673218985658, 0.8068088781110027), (0.6771674435728143, 0.8236149186074013), (0.6938914696275619, 0.8421833405319189), (0.7086781962097893, 0.8623286143505897), (0.7213798792353292, 0.8838494551465597), (0.7318696076860085, 0.9065308337924398), (0.7400425716625911, 0.9301461254502672), (0.7458171096106458, 0.9544593739319747), (0.7491355242558042, 0.9792276492956243)]
        
        # AIRFOIL
        # vertices = [(1.500362, 1.001207), (1.499415, 1.001635), (1.496575, 1.002915), (1.491851, 1.005031), (1.485257, 1.007956), (1.476812, 1.011653), (1.4665409999999999, 1.016078), (1.454476, 1.021177), (1.440653, 1.026886), (1.425115, 1.033139), (1.407912, 1.039859), (1.3890989999999999, 1.046969), (1.3687390000000001, 1.054385), (1.346901, 1.062022), (1.323658, 1.069793), (1.299092, 1.07761), (1.27329, 1.085386), (1.246345, 1.093035), (1.2183570000000001, 1.100473), (1.18943, 1.10762), (1.1596739999999999, 1.114396), (1.129204, 1.120731), (1.098139, 1.126555), (1.066605, 1.131806), (1.034729, 1.136429), (1.002644, 1.140374), (0.970484, 1.143601), (0.9383870000000001, 1.146077), (0.9064920000000001, 1.1477789999999999), (0.874044, 1.148488), (0.841845, 1.147792), (0.810292, 1.145719), (0.7795540000000001, 1.142334), (0.749791, 1.137721), (0.72116, 1.131984), (0.6938070000000001, 1.125241), (0.66787, 1.117627), (0.643472, 1.109283), (0.620725, 1.100361), (0.599729, 1.091016), (0.580568, 1.081406), (0.563315, 1.071683), (0.54803, 1.061999), (0.53476, 1.052493), (0.52354, 1.043297), (0.514395, 1.034528), (0.50734, 1.026286), (0.50238, 1.018658), (0.499513, 1.011707), (0.498726, 1.00548), (0.5, 1.0), (0.503247, 0.995407), (0.508372, 0.991824), (0.515332, 0.989225), (0.524077, 0.987574), (0.534549, 0.986823), (0.546684, 0.986917), (0.560413, 0.987787), (0.575663, 0.98936), (0.592357, 0.991553), (0.610415, 0.994278), (0.629758, 0.997441), (0.650306, 1.000943), (0.6719809999999999, 1.004684), (0.694706, 1.008559), (0.718407, 1.012465), (0.743013, 1.016297), (0.768456, 1.019952), (0.794667, 1.023329), (0.821583, 1.026332), (0.849138, 1.028866), (0.877266, 1.030845), (0.906127, 1.032201), (0.93628, 1.033226), (0.966726, 1.034045), (0.997356, 1.034626), (1.0280610000000001, 1.034939), (1.058728, 1.034964), (1.089242, 1.034687), (1.119486, 1.034104), (1.149343, 1.033216), (1.178695, 1.032035), (1.207422, 1.030577), (1.235408, 1.028866), (1.262537, 1.026934), (1.288694, 1.024814), (1.313766, 1.022547), (1.337647, 1.020175), (1.360229, 1.017742), (1.381414, 1.015294), (1.401105, 1.012876), (1.419213, 1.010534), (1.435654, 1.008309), (1.450351, 1.006243), (1.463235, 1.004372), (1.474244, 1.00273), (1.483326, 1.001346), (1.4904359999999999, 1.000244), (1.49554, 0.999443), (1.498612, 0.998956), (1.499638, 0.998793)]
        
        # TODO: add back i_start and i_end
        i_start, i_end = self.grid.Nghost, self.grid.Nx + self.grid.Nghost
        j_start, j_end = self.grid.Nghost, self.grid.Ny + self.grid.Nghost
        
        print("called inside_near_polygon!")
        inside_polygon, near_polygon, near_polygon_points = self.ebs.find_inside_near_points(vertices, i_start, i_end, j_start, j_end)
                
        # TIME LOOP
        while (self.t < self.inp.t_finish) and self.timestepNum < self.inp.nt:
            print(f"Timestep: {self.timestepNum}, Current time: {self.t}")

            self.bcs.apply_bcs()

            self.grid.assert_variable_type("prim")

            # Store primitive variables at the start of the step for Powell terms
            primU_n = np.copy(self.grid.grid)

            # Determine timestep dt based on CFL condition
            if self.inp.system == "euler2D":
                density = self.grid.grid[self.c.RHOCOMP]
                pressure = self.grid.grid[self.c.PCOMP]
                u = self.grid.grid[self.c.UCOMP]
                v = self.grid.grid[self.c.VCOMP]
                # Ensure pressure and density are positive before sqrt
                pressure = np.maximum(pressure, 1e-12)
                density = np.maximum(density, 1e-12)
                a = np.sqrt(self.c.gamma * pressure / density)  # Sound speed
                max_speed_x = np.max(np.abs(u) + a)
                max_speed_y = np.max(np.abs(v) + a)
                # More robust estimate
                max_speed = max(max_speed_x, max_speed_y)

            elif self.inp.system == "mhd2d":
                density = self.grid.grid[self.c.RHOCOMP]
                pressure = self.grid.grid[self.c.PCOMP]
                u = self.grid.grid[self.c.UCOMP]
                v = self.grid.grid[self.c.VCOMP]
                Bx = self.grid.grid[self.c.BXCOMP]
                By = self.grid.grid[self.c.BYCOMP]

                # Ensure pressure and density are positive
                pressure = np.maximum(pressure, 1e-12)
                density = np.maximum(density, 1e-12)

                a = np.sqrt(self.c.gamma * pressure / density)  # Sound speed
                # Alfven speed squared components
                ca_sq_x = Bx**2 / density
                ca_sq_y = By**2 / density
                ca_sq_tot = ca_sq_x + ca_sq_y

                # Fast magnetosonic speed squared (cf^2)
                # cf^2 = 0.5 * ( (a^2 + ca_tot^2) + sqrt( max( (a^2 + ca_tot^2)^2 - 4*a^2*ca_x^2 , 0.0 ) ) )
                term_under_sqrt = (a**2 + ca_sq_tot)**2 - 4 * a**2 * ca_sq_x
                cf_sq_x = 0.5 * ((a**2 + ca_sq_tot) +
                                 np.sqrt(np.maximum(term_under_sqrt, 0.0)))
                cf_x = np.sqrt(cf_sq_x)

                # Use ca_sq_y for y-direction cf
                term_under_sqrt = (a**2 + ca_sq_tot)**2 - 4 * a**2 * ca_sq_y
                cf_sq_y = 0.5 * ((a**2 + ca_sq_tot) +
                                 np.sqrt(np.maximum(term_under_sqrt, 0.0)))
                cf_y = np.sqrt(cf_sq_y)

                # Max signal speed is max(|u|+cf_x, |v|+cf_y)
                max_signal_x = np.max(np.abs(u) + cf_x)
                max_signal_y = np.max(np.abs(v) + cf_y)
                max_speed = max(max_signal_x, max_signal_y)

            else:
                raise RuntimeError(
                    f"System {self.inp.system} not supported for dt calculation.")

            # Calculate dt, ensuring it doesn't overshoot t_finish
            dt = min(self.inp.cfl * min(self.grid.dx, self.grid.dy) /
                     max_speed, self.inp.t_finish - self.t)
            if dt <= 0:
                raise ValueError(
                    f"Calculated dt is zero or negative ({dt}). Check simulation parameters or state.")

            if self.inp.timeStepper == "RK1":
                # returns numerical flux and conservative variables at interface
                self.grid.assert_variable_type("prim")
                
                consU = self.euler.prim_to_cons(self.grid.grid)
                
                # consU, numFluxX_plus, numFluxX_minus, numFluxY_plus, numFluxY_minus = self.flux.getFlux(
                #     self.grid.grid, self.grid.Nx, self.grid.Ny, self.grid.Nghost)

                # Start with the current conservative variables
                U_new = np.copy(consU)
                # print("TYPE OF U_new!!!!", type(U_new))
                primU = self.euler.cons_to_prim(U_new)
                
                print(self.grid.grid.shape)
                grid_shape_2d = self.grid.grid.shape[1:]
                print(grid_shape_2d)

                # FLUID UPDATE
                
                print("called ebs!")
                self.ebs.apply_embedded_boundary_conditions(U_new, primU, self.inp.system, vertices, inside_polygon, near_polygon, near_polygon_points, i_start, i_end, j_start, j_end)
                                
                # TODO: is it better for this to use consU since we're directly modifying U_new?
                _, numFluxX_plus, numFluxX_minus, numFluxY_plus, numFluxY_minus = self.flux.getFlux(
                    self.grid.grid, self.grid.Nx, self.grid.Ny, self.grid.Nghost)

                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        for icomp in range(self.c.NUMQ):
                            if icomp in (self.c.MUCOMP, self.c.MVCOMP) and near_polygon[i, j] == True:
                                continue
                            elif inside_polygon[i, j] == True:
                                continue
                            else:
                                # Fallback: apply finite volume update if not inside/near
                                delta = (
                                    (dt / self.grid.dx) * (numFluxX_plus[icomp, i, j] - numFluxX_minus[icomp, i, j]) +
                                    (dt / self.grid.dy) * (numFluxY_plus[icomp, i, j] - numFluxY_minus[icomp, i, j])
                                )
                                
                                updated_value = consU[icomp, i, j] - delta

                                # Apply floor where needed
                                # TODO: floors too high?
                                # floor_values = {self.c.RHOCOMP: 0.01, self.c.PCOMP: 0.01}
                                U_new[icomp, i, j] = updated_value
                                

                # Powell divergence cleaning for MHD
                if self.inp.system == "mhd2d":
                    # Calculate div(B) using central differences on consU
                    divB = np.zeros_like(consU[0])
                    # Need to calculate divB over the domain where U_new is updated + 1 layer for central diff
                    # However, we only apply the source term within the main update loop domain.
                    # Note: Using consU which contains Bx, By directly.
                    for i in range(i_start, i_end):
                        for j in range(j_start, j_end):
                            # Central difference requires i-1, i+1, j-1, j+1
                            # Ensure indices are within the bounds where consU is valid (including ghosts)
                            divB_x = (
                                consU[self.c.BXCOMP, i + 1, j] - consU[self.c.BXCOMP, i - 1, j]) / (2.0 * self.grid.dx)
                            divB_y = (
                                consU[self.c.BYCOMP, i, j + 1] - consU[self.c.BYCOMP, i, j - 1]) / (2.0 * self.grid.dy)
                            divB[i, j] = divB_x + divB_y

                    # Calculate Powell source terms using primU_n and consU
                    powell_source = calculate_powell_source(
                        consU, primU_n, divB, self.c)

                    # Apply Powell source terms to U_new
                    for i in range(i_start, i_end):
                        for j in range(j_start, j_end):
                            for icomp in range(self.c.NUMQ):
                                U_new[icomp, i, j] += dt * \
                                    powell_source[icomp, i, j]

                # TODO: particle step

            else:
                raise RuntimeError("Timestepping method not supported.")

            # self.grid.plot()
            # Update the grid with the new conservative variables
            # self.grid.set(U_new)
            self.grid.grid = self.euler.cons_to_prim(U_new)
            self.grid.variables = "prim"

            # Convert back to primitive variables
            # self.grid.transform(self.euler.cons_to_prim, "prim")

            self.bcs.apply_bcs()

            # assert np.all(np.isfinite(self.grid.grid)), f"Invalid values in grid at timestep {self.timestepNum}"
            # assert np.all(self.grid.grid[self.c.PCOMP] > 0), f"Negative pressure detected at timestep {self.timestepNum}"

            self.timestepNum += 1
            self.t += dt

            # always output the last timestep.
            if (self.timestepNum % self.inp.output_freq == 0) or (self.timestepNum == self.inp.nt-1):
                self.output()

            # DEBUG
            # self.grid.plot()
            self.grid.check_grid(self.c)

        if self.inp.make_movie:
            self.generate_movie()

        print("SUCCESS!")
        return self.grid

    # TODO: why does this unused plot function exist?
    
    # def plot(self):
    #     if not os.path.exists(self.inp.output_dir):
    #         os.makedirs(self.inp.output_dir)

    #     if self.inp.system == "euler2D":
    #         fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    #         axs[0].scatter(
    #             self.grid.x, self.grid.grid[self.c.RHOCOMP, :], c="black")
    #         axs[0].set_ylabel("Density")

    #         axs[1].scatter(
    #             self.grid.x, self.grid.grid[self.c.UCOMP, :],  c="black")
    #         axs[1].set_ylabel("Velocity")

    #         axs[2].scatter(
    #             self.grid.x, self.grid.grid[self.c.PCOMP, :],  c="black")
    #         axs[2].set_ylabel("Pressure")

    #     elif self.inp.system == "mhd2d":
    #         fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    #         axs[0].scatter(
    #             self.grid.x, self.grid.grid[self.c.RHOCOMP, :], c="black")
    #         axs[0].set_ylabel("Density")

    #         axs[1].scatter(
    #             self.grid.x, self.grid.grid[self.c.UCOMP, :],  c="black")
    #         axs[1].set_ylabel("Velocity")

    #         axs[2].scatter(
    #             self.grid.x, self.grid.grid[self.c.PCOMP, :],  c="black")
    #         axs[2].set_ylabel("Pressure")

    #         axs[3].scatter(
    #             self.grid.x, self.grid.grid[self.c.B_XCOMP, :],  c="black")
    #         axs[3].set_ylabel("Magnetic Field")

    #     axs[0].set_title(f"Time: {self.t:.4f}")
    #     plt.savefig(
    #         f"{self.inp.output_dir}/plot_dt{str(self.timestepNum).zfill(6)}")
    #     plt.close()

    def apply_ics(self):
        if self.inp.system == "euler2D":
            if self.inp.ics == "diagonal_advection":
                print("diagt??")
                self.grid.fill_grid(ics.diagonal_advection_2d)
            elif self.inp.ics == "kelvin_helmholtz":
                self.grid.fill_grid(ics.kelvin_helmholtz_2d)
            elif self.inp.ics == "double_mach_reflection":
                self.grid.fill_grid(ics.double_mach_reflection_2d)
            elif self.inp.ics == "riemann_problem":
                self.grid.fill_grid(ics.riemann_2d)
            elif self.inp.ics == "static":
                print("static!")
                self.grid.fill_grid(ics.static_2d)
            else:
                raise RuntimeError("[FLUID] ICS not valid.")

        elif self.inp.system == "mhd2d":
            if self.inp.ics == "orszag_tang":
                self.grid.fill_grid(ics.orszag_tang_2d)
            elif self.inp.ics == "field_loop":
                self.grid.fill_grid(ics.field_loop_2d)
            elif self.inp.ics == "rotor":
                print("before filling grid")
                self.grid.fill_grid(ics.rotor_2d)
                print("after filling grid")
            elif self.inp.ics == "staticmhd":
                print("static mhd!")
                self.grid.fill_grid(ics.static_mhd_2d)

        else:
            raise RuntimeError("[FLUID] ICS not valid.")

    def applyParticles(self):
        """Particle Setup"""
        # I assume this should function similar to apply ics?
        # check if self.inp.particle_ic = ...
        pass

    def output(self):
        # Ensure the base output directory exists
        os.makedirs(self.inp.output_dir, exist_ok=True)

        # Ensure the frames subdirectory exists
        frames_dir = os.path.join(self.inp.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        data_dir = os.path.join(self.inp.output_dir, "raw data")
        os.makedirs(data_dir, exist_ok=True)

        # File naming convention: output_timestepNum.txt
        output_filename = os.path.join(
            data_dir, f"output_{str(self.timestepNum).zfill(6)}.txt")
        output_plotname = os.path.join(
            frames_dir, f"output_{str(self.timestepNum).zfill(6)}.png")

        with open(output_filename, 'w') as f:
            # Write header
            f.write(f"# Time: {self.t:.4f}\n")
            f.write("# x, y, density, x-velocity, y-velocity, pressure\n")

            for i in range(self.grid.Nghost, self.grid.Nx - self.grid.Nghost):
                for j in range(self.grid.Nghost, self.grid.Ny - self.grid.Nghost):
                    x = self.grid.x[i]
                    y = self.grid.y[j]
                    components = [self.grid.grid[q, i, j] for q in range(self.c.NUMQ)]
                    f.write(f"{x:.12f}, {y:.12f}, " + ", ".join(f"{comp:.8f}" for comp in components) + "\n")

        if self.inp.system == "euler2D":
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        elif self.inp.system == "mhd2d":
            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        axs = axs.ravel()  # Flatten the array to index by i

        for q in range(self.c.NUMQ):
            # Exclude ghost cells from the plot
            plot_data = self.grid.grid[q, self.grid.Nghost:-self.grid.Nghost, self.grid.Nghost:-self.grid.Nghost].T # TODO: why transpose?
            extent = [self.grid.x[self.grid.Nghost], self.grid.x[-self.grid.Nghost-1],
                      self.grid.y[self.grid.Nghost], self.grid.y[-self.grid.Nghost-1]]

            im = axs[q].imshow(plot_data, origin='lower', extent=extent, cmap='magma')
            print("shape", plot_data.shape)
            
            plt.colorbar(im, ax=axs[q])
            
            # lots of cool tricks to make a transparent mask from here: https://stackoverflow.com/questions/10127284/overlay-imshow-plots-in-matplotlib
            transparent = cm.get_cmap("binary")
            alphas = np.linspace(0.5, 0, transparent.N+3)
            transparent._init()
            transparent._lut[:,-1] = alphas
            
            highlight_data = (self.ebs.highlight_near_polygon[q, self.grid.Nghost:-self.grid.Nghost, self.grid.Nghost:-self.grid.Nghost]).T # TODO: why transpose?
            axs[q].imshow(highlight_data, origin='lower', extent=extent, alpha=0.5, cmap=transparent)
            
            axs[q].set_title(self.c.variable_names[q])
            axs[q].set_xlabel('x')
            axs[q].set_ylabel('y')

        fig.suptitle(f"Time: {self.t:.4f}, Timestep: {self.timestepNum}")
        plt.tight_layout()
        fig.savefig(output_plotname)
        plt.close()

    def generate_movie(self):
        # Create a directory for the frames if it doesn't exist
        frames_dir = os.path.join(self.inp.output_dir, "frames")
        # if not os.path.exists(frames_dir):
        #     os.makedirs(frames_dir)

        # List all the output files and sort them
        # output_files = sorted(glob.glob(os.path.join(self.inp.output_dir, "output_*.png")))

        # movie_filename = os.path.join(self.inp.output_dir, "simulation_movie.mp4")

        ffmpeg_command = f"ffmpeg -y -framerate 24 -i {frames_dir}/output_%06d.png -c:v libx264 -pix_fmt yuv420p {self.inp.output_dir}/movie.mp4"
        os.system(ffmpeg_command)

        # if self.c.NUMQ == 3:
        #     fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        # elif self.c.NUMQ == 4:
        #     fig, axs = plt.subplots(2,2, figsize=(10, 15))
        # else:
        #     print("self.c.NUMQ: ", self.c.NUMQ)
        #     raise RuntimeError("System not implemented for movie.")

        # axs = axs.ravel()  # Flatten the array to index by i

        # def update_plot(file):
        #     data = np.loadtxt(file, delimiter=',', skiprows=2)
        #     x = data[:, 0]

        #     #data is indexed by
        #     # data[:,i] where i = 0 for x, i = 1 for icomp1, i=2 for icomp2

        #     with open(file, 'r') as f:
        #         lines = f.readlines()
        #         time_line = lines[0]
        #         time = float(time_line.split(':')[1].strip())

        #     timestep = int(file.split('_')[-1].split('.')[0])

        #     for i in range(self.c.NUMQ):
        #         axs[i].clear()

        #         axs[i].scatter(x, data[:,i+1], c="black")
        #         axs[i].set_ylabel(self.c.variable_names[i])

        #     axs[0].set_title(f"Time: {time:.4f}, Timestep: {timestep}")

        # # Create an animation by updating the plot for each output file
        # ani = animation.FuncAnimation(fig, update_plot, frames=output_files, repeat=False)

        # # Save the animation as a movie file using ffmpeg
        # movie_filename = os.path.join(self.inp.output_dir, "simulation_movie.mp4")
        # ani.save(movie_filename, writer='ffmpeg', fps=10)

        # print(f"Movie saved as {movie_filename}")

# Define a helper function to calculate the Powell source term outside the main loop for clarity


def calculate_powell_source(consU, primU, divB, c):
    """Calculates the Powell et al. (1999) source terms."""
    Bx = consU[c.BXCOMP]
    By = consU[c.BYCOMP]
    u = primU[c.UCOMP]
    v = primU[c.VCOMP]

    powell_source = np.zeros_like(consU)
    # S_rho = 0
    powell_source[c.MUCOMP] = -Bx * divB
    powell_source[c.MVCOMP] = -By * divB
    # S_MWCOMP = 0 in 2D
    powell_source[c.ECOMP] = -(u * Bx + v * By) * divB
    powell_source[c.BXCOMP] = -u * divB
    powell_source[c.BYCOMP] = -v * divB
    # S_BZCOMP = 0 in 2D
    return powell_source
