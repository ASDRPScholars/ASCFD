from ascfd.grid import Grid2D
from ascfd.constants import Constants
import ascfd.ics as ics
import glob
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import copy

import sys
from ascfd.flux import Flux


from ascfd.euler import Euler

from ascfd.bcs import BoundaryConditions

import numpy as np
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

            # EMBEDDED BOUNDARIES — MODULARIZE LATER

            lower_bound = 0
            upper_bound = 1
            step = 0.01

            # TODO: none of these three versions of vertices draws a circle in the center:

            # old_vertices = [(0.4990133642141358, 0.03139525976465669), (0.49605735065723894, 0.06266661678215213), (0.49114362536434436, 0.09369065729286231), (0.48429158056431554, 0.1243449435824274), (0.47552825814757677, 0.1545084971874737), (0.4648882429441257, 0.184062276342339), (0.45241352623300973, 0.21288964578253636), (0.4381533400219318, 0.24087683705085766), (0.42216396275100754, 0.26791339748949833), (0.4045084971874737, 0.29389262614623657), (0.38525662138789457, 0.3187119948743449), (0.3644843137107058, 0.34227355296434436), (0.3422735529643443, 0.3644843137107058), (0.3187119948743448, 0.3852566213878946), (0.2938926261462365, 0.4045084971874737), (0.2679133974894983, 0.42216396275100754), (0.24087683705085758, 0.43815334002193185), (0.21288964578253633, 0.4524135262330098), (0.18406227634233893, 0.46488824294412573), (0.15450849718747373, 0.47552825814757677), (0.12434494358242737, 0.48429158056431554), (0.09369065729286226, 0.49114362536434436), (0.06266661678215213, 0.49605735065723894), (0.03139525976465665, 0.4990133642141358), (-8.040613248383183e-17, 0.5), (-0.0313952597646567, 0.4990133642141358), (-0.06266661678215218, 0.4960573506572389), (-0.09369065729286241, 0.4911436253643443), (-0.12434494358242743, 0.48429158056431554), (-0.15450849718747378, 0.47552825814757677), (-0.184062276342339, 0.4648882429441257), (-0.21288964578253636, 0.45241352623300973), (-0.24087683705085772, 0.43815334002193174), (-0.26791339748949844, 0.4221639627510075), (-0.2938926261462365, 0.4045084971874737), (-0.3187119948743449, 0.3852566213878946), (-0.34227355296434436, 0.3644843137107057), (-0.36448431371070583, 0.34227355296434425), (-0.3852566213878947, 0.31871199487434476), (-0.40450849718747367, 0.2938926261462366), (-0.42216396275100754, 0.26791339748949833), (-0.4381533400219318, 0.2408768370508576), (-0.4524135262330098, 0.21288964578253625), (-0.46488824294412573, 0.18406227634233888), (-0.4755282581475768, 0.15450849718747356), (-0.48429158056431554, 0.12434494358242741), (-0.49114362536434436, 0.09369065729286229), (-0.49605735065723894, 0.06266661678215205), (-0.4990133642141358, 0.03139525976465657), (-0.5, -1.6081226496766366e-16),
            #                 (-0.4990133642141358, -0.03139525976465667), (-0.49605735065723894, -0.06266661678215214), (-0.4911436253643443, -0.09369065729286238), (-0.48429158056431554, -0.12434494358242751), (-0.47552825814757677, -0.15450849718747386), (-0.4648882429441256, -0.18406227634233915), (-0.45241352623300973, -0.21288964578253633), (-0.43815334002193174, -0.2408768370508577), (-0.4221639627510075, -0.2679133974894984), (-0.4045084971874736, -0.2938926261462367), (-0.3852566213878945, -0.318711994874345), (-0.3644843137107058, -0.34227355296434436), (-0.3422735529643443, -0.36448431371070583), (-0.31871199487434476, -0.3852566213878947), (-0.2938926261462366, -0.40450849718747367), (-0.26791339748949816, -0.42216396275100765), (-0.24087683705085763, -0.4381533400219318), (-0.21288964578253608, -0.4524135262330099), (-0.1840622763423389, -0.46488824294412573), (-0.15450849718747378, -0.47552825814757677), (-0.12434494358242722, -0.4842915805643156), (-0.09369065729286231, -0.49114362536434436), (-0.06266661678215187, -0.49605735065723894), (-0.031395259764656604, -0.4990133642141358), (-9.184850993605148e-17, -0.5), (0.03139525976465686, -0.4990133642141358), (0.06266661678215212, -0.49605735065723894), (0.09369065729286256, -0.4911436253643443), (0.12434494358242747, -0.48429158056431554), (0.15450849718747361, -0.4755282581475768), (0.18406227634233913, -0.4648882429441256), (0.2128896457825363, -0.4524135262330098), (0.24087683705085786, -0.4381533400219317), (0.2679133974894984, -0.4221639627510075), (0.29389262614623685, -0.4045084971874735), (0.318711994874345, -0.3852566213878945), (0.3422735529643443, -0.3644843137107058), (0.36448431371070594, -0.34227355296434414), (0.3852566213878946, -0.3187119948743448), (0.4045084971874739, -0.2938926261462363), (0.42216396275100765, -0.26791339748949816), (0.4381533400219318, -0.24087683705085766), (0.45241352623300984, -0.2128896457825361), (0.46488824294412573, -0.18406227634233893), (0.4755282581475769, -0.1545084971874734), (0.4842915805643156, -0.12434494358242724), (0.49114362536434436, -0.09369065729286234), (0.49605735065723894, -0.0626666167821519), (0.4990133642141358, -0.03139525976465663)]

            # vertices = [(0.749506682107068, 0.5156976298823284), (0.7480286753286195, 0.5313333083910761), (0.7455718126821722, 0.5468453286464312), (0.7421457902821578, 0.5621724717912137), (0.7377641290737884, 0.5772542485937369), (0.7324441214720628, 0.5920311381711695), (0.7262067631165049, 0.6064448228912682), (0.7190766700109659, 0.6204384185254288), (0.7110819813755038, 0.6339566987447491), (0.7022542485937369, 0.6469463130731183), (0.6926283106939473, 0.6593559974371724), (0.6822421568553529, 0.6711367764821722), (0.6711367764821722, 0.6822421568553529), (0.6593559974371724, 0.6926283106939473), (0.6469463130731182, 0.7022542485937369), (0.6339566987447491, 0.7110819813755038), (0.6204384185254288, 0.7190766700109659), (0.6064448228912682, 0.7262067631165049), (0.5920311381711695, 0.7324441214720628), (0.5772542485937369, 0.7377641290737884), (0.5621724717912137, 0.7421457902821578), (0.5468453286464311, 0.7455718126821722), (0.5313333083910761, 0.7480286753286195), (0.5156976298823284, 0.749506682107068), (0.49999999999999994, 0.75), (0.48430237011767163, 0.749506682107068), (0.46866669160892394, 0.7480286753286194), (0.4531546713535688, 0.7455718126821722), (0.4378275282087863, 0.7421457902821578), (0.42274575140626314, 0.7377641290737884), (0.4079688618288305, 0.7324441214720628), (0.39355517710873184, 0.7262067631165049), (0.37956158147457114, 0.7190766700109659), (0.36604330125525075, 0.7110819813755037), (0.35305368692688177, 0.7022542485937369), (0.34064400256282756, 0.6926283106939473), (0.3288632235178278, 0.6822421568553528), (0.31775784314464706, 0.6711367764821721), (0.30737168930605263, 0.6593559974371723), (0.29774575140626314, 0.6469463130731183), (0.28891801862449623, 0.6339566987447491), (0.28092332998903413, 0.6204384185254288), (0.2737932368834951, 0.6064448228912681), (0.26755587852793716, 0.5920311381711695), (0.2622358709262116, 0.5772542485937368), (0.25785420971784223, 0.5621724717912137), (0.2544281873178278, 0.5468453286464311), (0.2519713246713805, 0.5313333083910761), (0.2504933178929321, 0.5156976298823283), (0.25, 0.49999999999999994),
            #             (0.2504933178929321, 0.4843023701176717), (0.2519713246713805, 0.46866669160892394), (0.25442818731782785, 0.4531546713535688), (0.25785420971784223, 0.43782752820878623), (0.2622358709262116, 0.4227457514062631), (0.26755587852793716, 0.4079688618288304), (0.2737932368834951, 0.39355517710873184), (0.28092332998903413, 0.37956158147457114), (0.2889180186244963, 0.3660433012552508), (0.2977457514062632, 0.35305368692688166), (0.30737168930605274, 0.3406440025628275), (0.3177578431446471, 0.3288632235178278), (0.3288632235178278, 0.31775784314464706), (0.3406440025628276, 0.30737168930605263), (0.35305368692688166, 0.29774575140626314), (0.3660433012552509, 0.2889180186244962), (0.3795615814745712, 0.28092332998903413), (0.39355517710873195, 0.27379323688349505), (0.4079688618288305, 0.26755587852793716), (0.42274575140626314, 0.2622358709262116), (0.4378275282087864, 0.25785420971784223), (0.45315467135356885, 0.2544281873178278), (0.46866669160892405, 0.2519713246713805), (0.4843023701176717, 0.2504933178929321), (0.49999999999999994, 0.25), (0.5156976298823285, 0.2504933178929321), (0.5313333083910761, 0.2519713246713805), (0.5468453286464313, 0.25442818731782785), (0.5621724717912138, 0.25785420971784223), (0.5772542485937369, 0.2622358709262116), (0.5920311381711696, 0.26755587852793716), (0.6064448228912681, 0.2737932368834951), (0.6204384185254289, 0.28092332998903413), (0.6339566987447491, 0.2889180186244963), (0.6469463130731185, 0.29774575140626325), (0.6593559974371725, 0.30737168930605274), (0.6711367764821722, 0.3177578431446471), (0.6822421568553529, 0.32886322351782793), (0.6926283106939473, 0.34064400256282756), (0.702254248593737, 0.3530536869268819), (0.7110819813755038, 0.3660433012552509), (0.7190766700109659, 0.3795615814745712), (0.7262067631165049, 0.39355517710873195), (0.7324441214720628, 0.4079688618288305), (0.7377641290737884, 0.4227457514062633), (0.7421457902821578, 0.4378275282087864), (0.7455718126821722, 0.45315467135356885), (0.7480286753286195, 0.46866669160892405), (0.749506682107068, 0.4843023701176717)]

            # vertices = [(0.75, 0.5), (0.7487510413195064, 0.524958354161707), (0.7450166444603104, 0.5496673326987653), (0.7388341222814014, 0.5738800516653348), (0.7302652485007213, 0.5973545855771626), (0.7193956404725932, 0.6198563846510508), (0.7063339037274196, 0.6411606183487588), (0.6912105468211222, 0.6610544218094228), (0.6741766773367914, 0.6793390227248807), (0.6554024920676661, 0.6958317274068708), (0.6350755764670349, 0.7103677462019741), (0.6133990303563943, 0.7228018400153589), (0.5905894386191683, 0.7330097714918066), (0.5668747071561469, 0.7408895463542983), (0.5424917857250602, 0.7463624324971151), (0.5176843004169257, 0.7493737466510136), (0.49270011942467773, 0.7498934007603762), (0.4677888764261187, 0.7479162026131172), (0.4431994763267281, 0.7434619077195488), (0.419177608284124, 0.7365750219218535), (0.3959632908632143, 0.7273243567064204), (0.37378847385003555, 0.7158023416622183), (0.35287472068616343, 0.7021241009548974), (0.3334309946800438, 0.68642630304418), (0.3156515711146885, 0.6688657951377877), (0.29971409611326644, 0.6496180360259889), (0.28577781165776306, 0.6288753429553658), (0.27398196449573464, 0.6068449700584573), (0.2644444148328354, 0.5837470375389759), (0.2572604587126023, 0.5598123323034953), (0.2525018758498886, 0.5352800020149665), (0.25021621243168013, 0.5103951656083223), (
            #     0.2504263060513118, 0.4854064641431046), (0.25313005752278384, 0.4605635764641875), (0.25830045185513484, 0.43611472449329175), (0.2658858281773011, 0.41230419307759464), (0.27581039591646345, 0.38936988917628645), (0.2879749920723982, 0.3675409647728763), (0.30225807202139615, 0.34703552726431985), (0.31851692394996534, 0.3280584602040062), (0.3365890947840974, 0.31079937617301767), (0.3562940133666831, 0.29543072223389716), (0.3774347946648253, 0.28210605689660284), (0.39980020698000635, 0.2709585158126362), (0.4231667825053952, 0.26209948152762097), (0.4473010501423051, 0.25561747058372575), (0.47196186826623626, 0.2515772490916339), (0.49690283413427716, 0.2500191856089748), (0.5218747458598614, 0.25095884779103983), (0.5466280923556435, 0.25438684684391677), (0.5709155463658061, 0.26026893133421525), (0.5944944356782447, 0.2685463294180667), (0.6171291678250936, 0.27913633606996135), (0.6385935840447896, 0.2919331394440243), (0.658673218985658, 0.30680887811100266), (0.6771674435728143, 0.3236149186074014), (0.6938914696275619, 0.3421833405319189), (0.7086781962097893, 0.36232861435058966), (0.7213798792353292, 0.38384945514655966), (0.7318696076860085, 0.40653083379243976), (0.7400425716625911, 0.43014612545026726), (0.7458171096106458, 0.4544593739319747), (0.7491355242558042, 0.47922764929562434)]

            vertices = [(0.55, 0.5), (0.5487510413195065, 0.524958354161707), (0.5450166444603104, 0.5496673326987653), (0.5388341222814015, 0.5738800516653348), (0.5302652485007213, 0.5973545855771626), (0.5193956404725932, 0.6198563846510508), (0.5063339037274196, 0.6411606183487588), (0.4912105468211221, 0.6610544218094228), (0.47417667733679136, 0.6793390227248807), (0.45540249206766614, 0.6958317274068708), (0.43507557646703493, 0.7103677462019741), (0.41339903035639436, 0.7228018400153589), (0.3905894386191684, 0.7330097714918066), (0.36687470715614684, 0.7408895463542983), (0.34249178572506017, 0.7463624324971151), (0.31768430041692564, 0.7493737466510136), (0.2927001194246777, 0.7498934007603762), (0.2677888764261187, 0.7479162026131172), (0.24319947632672811, 0.7434619077195488), (0.219177608284124, 0.7365750219218535), (0.1959632908632143, 0.7273243567064204), (0.1737884738500355, 0.7158023416622183), (0.15287472068616342, 0.7021241009548974), (0.1334309946800438, 0.68642630304418), (0.11565157111468849, 0.6688657951377877), (0.09971409611326643, 0.6496180360259889), (0.08577781165776305, 0.6288753429553658), (0.0739819644957346, 0.6068449700584573), (0.06444441483283536, 0.5837470375389759), (0.0572604587126023, 0.5598123323034953), (0.05250187584988858, 0.5352800020149665), (0.05021621243168012, 0.5103951656083223), (0.05042630605131174, 0.4854064641431046), (0.053130057522783825, 0.4605635764641875), (0.058300451855134855, 0.43611472449329175), (0.06588582817730107, 0.41230419307759464), (0.07581039591646344, 0.38936988917628645), (0.0879749920723982, 0.3675409647728763), (0.10225807202139614, 0.34703552726431985), (0.11851692394996532, 0.3280584602040062), (0.13658909478409734, 0.31079937617301767), (0.15629401336668305, 0.29543072223389716), (0.1774347946648253, 0.28210605689660284), (0.19980020698000633, 0.2709585158126362), (0.22316678250539515, 0.26209948152762097), (0.24730105014230508, 0.25561747058372575), (0.27196186826623625, 0.2515772490916339), (0.29690283413427715, 0.2500191856089748), (0.32187474585986137, 0.25095884779103983), (0.3466280923556435, 0.25438684684391677), (0.37091554636580615, 0.26026893133421525), (0.3944944356782446, 0.2685463294180667), (0.4171291678250937, 0.27913633606996135), (0.43859358404478965, 0.2919331394440243), (0.45867321898565794, 0.30680887811100266), (0.4771674435728144, 0.3236149186074014), (0.4938914696275618, 0.3421833405319189), (0.5086781962097893, 0.36232861435058966), (0.5213798792353292, 0.38384945514655966), (0.5318696076860084, 0.40653083379243976), (0.5400425716625912, 0.43014612545026726), (0.5458171096106459, 0.4544593739319747), (0.5491355242558043, 0.47922764929562434)]
            # vertices = [(0.4, 0.4), (0.4, 0.6), (0.6, 0.6), (0.6, 0.4)]

            # for vertex in old_vertices:
            #     x = (vertex[0] * 0.5) + 0.5
            #     y = (vertex[1] * 0.5) + 0.5
            #     vertices.append((x, y))

            print(vertices)
            # print(
            #     f"length of old_verticies: {len(old_vertices)}, verticies: {len(vertices)}")

            def is_inside_polygon(polygon_points, point):
                """
                Determine whether a point is inside a polygon using a horizontal ray-casting algorithm. 

                i.e. If we draw a horizontal, right-facing ray from a point and it intersects with the polygon an ODD number of times
                (an odd number of edges), the point is inside the polygon. If that right-facing raw intersects an EVEN number of edges,
                then the point is OUTSIDE the polygon. Draw it for yourself and see!

                Instead of counting even and odd, we'll just alternate between inside = True and inside = False.
                """

                n = len(polygon_points)
                inside = False

                px, py = point
                n = len(polygon_points)
                
                for i in range(n):
                    # we'll use these two points to draw a line/edge of the polygon
                    p1x, p1y = polygon_points[i]
                    p2x, p2y = polygon_points[(i + 1) % n]

                    # check if the point's y-level crosses this edge
                    if (p1y > py) != (p2y > py) and p1y != p2y:
                        # find the x where this edge intersects y = py
                        xint = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if px < xint:
                            inside = not inside

                return inside
            
            def is_near_polygon(polygon_points, point):
                n = len(polygon_points)
                inside = False

                px, py = point
                n = len(polygon_points)
                
                for (px, py) in [(px+0.01, py), (px-0.01, py), (px, py+0.01), (px, py-0.01)]:
                    for i in range(n):
                        # we'll use these two points to draw a line/edge of the polygon
                        p1x, p1y = polygon_points[i]
                        p2x, p2y = polygon_points[(i + 1) % n]

                        # check if the point's y-level crosses this edge
                        if (p1y > py) != (p2y > py) and p1y != p2y:
                            # find the x where this edge intersects y = py
                            xint = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if px < xint:
                                inside = not inside
                                
                    if inside:
                        return inside
                    
                return inside
                
            
            # TODO: find the negative sign
            def fluid_vector_through_time(embedded_boundary_vector, velocity_vector, vertices, dt):
                # TODO: no logic necessary to double-check if its going to head into the embedded boundary?
                
                _, V_parallel = decompose_vector(embedded_boundary_vector, velocity_vector)
                return V_parallel


            def decompose_vector(embedded_boundary_vector, velocity_vector):
                norm_squared = np.dot(embedded_boundary_vector, embedded_boundary_vector)
                if norm_squared == 0:
                    return velocity_vector, (0, 0)  # or return velocity unchanged
                dot_product = np.dot(velocity_vector, embedded_boundary_vector)
                V_parallel = (embedded_boundary_vector * dot_product / norm_squared)
                V_normal = velocity_vector - V_parallel
                return V_normal, V_parallel

            def find_embedded_boundary_vector(vertices, lower_bound, upper_bound, step):
                grid_size = 100
                x_vals = np.linspace(lower_bound, upper_bound, grid_size)
                y_vals = np.linspace(lower_bound, upper_bound, grid_size)
                vector_map = np.zeros((grid_size, grid_size, 2))

                for i, x in enumerate(x_vals):
                    for j, y in enumerate(y_vals):
                        cell_center = np.array([x + step / 2, y + step / 2])
                        min_dist = float('inf')
                        tangent = np.array([1, 0])  # default

                        for k in range(len(vertices)):
                            v1 = np.array(vertices[k])
                            v2 = np.array(vertices[(k + 1) % len(vertices)])
                            edge_center = 0.5 * (v1 + v2)
                            dist = np.linalg.norm(cell_center - edge_center)
                            if dist < min_dist:
                                min_dist = dist
                                tangent = v2 - v1

                        vector_map[i, j] = tangent
                return vector_map

            vector_map = find_embedded_boundary_vector(
                vertices, lower_bound, upper_bound, step)

            if self.inp.timeStepper == "RK1":
                # returns numerical flux and conservative variables at interface
                self.grid.assert_variable_type("prim")
                consU, numFluxX_plus, numFluxX_minus, numFluxY_plus, numFluxY_minus = self.flux.getFlux(
                    self.grid.grid, self.grid.Nx, self.grid.Ny, self.grid.Nghost)

                # Start with the current conservative variables
                U_new = np.copy(consU)
                primU = self.euler.cons_to_prim(U_new)

                i_start, i_end = self.grid.Nghost, self.grid.Nx + self.grid.Nghost
                j_start, j_end = self.grid.Nghost, self.grid.Ny + self.grid.Nghost

                # FLUID UPDATE
                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        point = np.array([0.01 * i, 0.01 * j])
                        
                        velocity_vector = np.array([
                                    primU[self.c.UCOMP, i, j],
                                    primU[self.c.VCOMP, i, j]
                                ])
                        
                        # future_point = point + velocity_vector * dt
                        
                        inside_polygon = is_inside_polygon(vertices, point)
                        
                        #TODO: optimize by bringing all of this logic into inside_polygon — 1 all-in-one call > 5 separate calls
                        near_polygon = is_near_polygon(vertices, point)

                        for icomp in range(self.c.NUMQ):
                            if inside_polygon:
                                break
                            elif near_polygon:
                                fluid_vec = fluid_vector_through_time(
                                    vector_map[i][j],
                                    velocity_vector,
                                    vertices,
                                    dt
                                )
                                
                                # please_be_nonzero = [fluid_vec[0], fluid_vec[1], consU[self.c.MUCOMP, i, j], consU[self.c.MVCOMP, i, j]]
                                
                                # for i in range(len(please_be_nonzero)):
                                #     if please_be_nonzero[i] != 0:
                                #         assert(f"index {i} is not zero!!!!")
                                        
                                # print(("UPPER HALF:" if j >= (j_end/2) else "LOWER HALF:"), fluid_vec[0], fluid_vec[1])
                                # print(("UPPER HALF:" if j >= (j_end/2) else "LOWER HALF:"), consU[self.c.MUCOMP, i, j], consU[self.c.MVCOMP, i, j])
                                
                                if icomp == self.c.MUCOMP:
                                    U_new[icomp, i, j] = U_new[self.c.RHOCOMP, i, j] * fluid_vec[0]
                                elif icomp == self.c.MVCOMP:
                                    U_new[icomp, i, j] = U_new[self.c.RHOCOMP, i, j] * fluid_vec[1]
                                elif icomp == self.c.RHOCOMP:
                                    # TODO: try doing an average instead of no-update
                                    U_new[icomp, i, j] = consU[icomp, i, j]
                                elif icomp == self.c.ECOMP:
                                    U_new[icomp, i, j] = consU[icomp, i, j]  # constant fill or marker
                            else:
                                delta = (
                                    (dt / self.grid.dx) * (numFluxX_plus[icomp, i, j] - numFluxX_minus[icomp, i, j]) +
                                    (dt / self.grid.dy) * (numFluxY_plus[icomp, i, j] - numFluxY_minus[icomp, i, j])
                                )
                                updated_value = consU[icomp, i, j] - delta

                                # Apply floors where appropriate
                                floor_values = {1: 0.01, 3: 0.01}
                                floor_value = floor_values.get(icomp, None)

                                if floor_value is not None:
                                    U_new[icomp, i, j] = max(updated_value, floor_value)
                                else:
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

                # TODO: embedded boundaries

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

    def plot(self):
        if not os.path.exists(self.inp.output_dir):
            os.makedirs(self.inp.output_dir)

        if self.inp.system == "euler2D":
            fig, axs = plt.subplots(3, 1, figsize=(10, 15))
            axs[0].scatter(
                self.grid.x, self.grid.grid[self.c.RHOCOMP, :], c="black")
            axs[0].set_ylabel("Density")

            axs[1].scatter(
                self.grid.x, self.grid.grid[self.c.UCOMP, :],  c="black")
            axs[1].set_ylabel("Velocity")

            axs[2].scatter(
                self.grid.x, self.grid.grid[self.c.PCOMP, :],  c="black")
            axs[2].set_ylabel("Pressure")

        elif self.inp.system == "mhd2d":
            fig, axs = plt.subplots(3, 1, figsize=(10, 15))
            axs[0].scatter(
                self.grid.x, self.grid.grid[self.c.RHOCOMP, :], c="black")
            axs[0].set_ylabel("Density")

            axs[1].scatter(
                self.grid.x, self.grid.grid[self.c.UCOMP, :],  c="black")
            axs[1].set_ylabel("Velocity")

            axs[2].scatter(
                self.grid.x, self.grid.grid[self.c.PCOMP, :],  c="black")
            axs[2].set_ylabel("Pressure")

            axs[3].scatter(
                self.grid.x, self.grid.grid[self.c.B_XCOMP, :],  c="black")
            axs[3].set_ylabel("Magnetic Field")

        axs[0].set_title(f"Time: {self.t:.4f}")
        plt.savefig(
            f"{self.inp.output_dir}/plot_dt{str(self.timestepNum).zfill(6)}")
        plt.close()

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

        # File naming convention: output_timestepNum.txt
        output_filename = os.path.join(
            frames_dir, f"output_{str(self.timestepNum).zfill(6)}.txt")
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
                    components = [self.grid.grid[q, i, j]
                                  for q in range(self.c.NUMQ)]
                    f.write(f"{x:.12f}, {y:.12f}, " +
                            ", ".join(f"{comp:.8f}" for comp in components) + "\n")

        if self.inp.system == "euler2D":
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        elif self.inp.system == "mhd2d":
            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        axs = axs.ravel()  # Flatten the array to index by i

        for i in range(self.c.NUMQ):
            # Exclude ghost cells from the plot
            plot_data = self.grid.grid[i, self.grid.Nghost:-
                                       self.grid.Nghost, self.grid.Nghost:-self.grid.Nghost].T
            extent = [self.grid.x[self.grid.Nghost], self.grid.x[-self.grid.Nghost-1],
                      self.grid.y[self.grid.Nghost], self.grid.y[-self.grid.Nghost-1]]

            im = axs[i].imshow(plot_data, origin='lower',
                               extent=extent, cmap='magma')
            plt.colorbar(im, ax=axs[i])
            axs[i].set_title(self.c.variable_names[i])
            axs[i].set_xlabel('x')
            axs[i].set_ylabel('y')

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
