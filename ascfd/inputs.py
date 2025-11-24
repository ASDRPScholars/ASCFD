import configparser
import numpy as np
import os
import ast

class CaseInsensitiveConfigParser(configparser.ConfigParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optionxform = (
            str.lower
        )  # Transform options to lowercase for case insensitivity


class Inputs:
    def __init__(self, fname):
        config = CaseInsensitiveConfigParser()
        config.read(fname)

        if not os.path.exists(fname):
            raise RuntimeError("Inputs File not found.")

        # Mesh
        self.nx = self.get_config_value(config, "Mesh", "nx", type_func=int)
        self.ny = self.get_config_value(config, "Mesh", "ny", type_func=int)
        self.ng = self.get_config_value(config, "Mesh", "ng", type_func=int)
        self.xlim = self.get_config_value(config, "Mesh", "xlim", type_func=self.parse_bounds)
        self.ylim = self.get_config_value(config, "Mesh", "ylim", type_func=self.parse_bounds)

        ## computed mesh properties
        self.nx_with_ghosts = self.nx + 2 * self.ng
        self.ny_with_ghosts = self.ny + 2 * self.ng
        
        self.L_x = int(self.nx/2)
        
        self.dx = (self.xlim[1] - self.xlim[0]) / (self.nx - 1)
        self.dy = (self.ylim[1] - self.ylim[0]) / (self.ny - 1)
        
        self.grid_x = np.linspace(self.xlim[0] - self.dx * self.ng, self.xlim[1] + self.dx * self.ng, self.nx + 2 * self.ng)
        self.grid_y = np.linspace(self.ylim[0] - self.dy * self.ng, self.ylim[1] + self.dy * self.ng, self.ny + 2 * self.ng)

        # Time
        self.nt = self.get_config_value(
            config, "Time", "time_steps", type_func=int, mandatory=False, default=np.inf
        )
        self.timeStepper = self.get_config_value(config, "Time", "method")
        self.t0 = self.get_config_value(
            config, "Time", "t0", mandatory=False, default=0.0, type_func=float
        )
        self.t_finish = self.get_config_value(
            config, "Time", "t_finish", mandatory=False, default=np.inf, type_func=float
        )

        if (self.t_finish == np.inf) and (self.time_steps == np.inf):
            raise RuntimeError(
                "Must supply either t_finish or time_steps in [Time] section of inputs file."
            )
        self.cfl = self.get_config_value(config, "Time", "cfl", type_func=float)

        if self.t0 >= self.t_finish:
            raise RuntimeError("Initial time is >= to the final time.")
        
        self.dt = self.get_config_value(config, "Time", "dt", type_func=float)

        self.k_B = 8.617e-5

        
        # Electrons
        self.e_system = self.get_config_value(config, "Electrons", "system")
        self.e_ics = self.get_config_value(config, "Electrons", "ics")
        self.rho_e = self.get_config_value(config, "Electrons", "density", type_func = float)
        self.p_e = self.get_config_value(config, "Electrons", "pressure", type_func = float)
        self.T_e = self.get_config_value(config, "Electrons", "T_e", type_func = float)
        self.gammas = self.get_config_value(config, "Electrons", "gammas", type_func=lambda s: [float(item) for item in s.strip('[]').split(',')])
        self.mW = self.get_config_value(config, "Electrons", "mW", type_func=lambda s: [float(item) for item in s.strip('[]').split(',')])

        # Method
        self.flux = self.get_config_value(config, "Method", "flux")
        self.bcs_lo = self.get_config_value(config, "Method", "bcs_lo", type_func=self.parse_bcs)
        self.bcs_hi = self.get_config_value(config, "Method", "bcs_hi", type_func=self.parse_bcs)
        self.collision_type = self.get_config_value(config, "Method", "collision_type", mandatory=False, default="rate_coeffs")

        # Params
        self.propellant = self.get_config_value(config, "Parameters", "propellant", type_func = str)
        self.q = self.get_config_value(config, "Parameters", "charge", type_func = float)
        self.m_e = self.get_config_value(config, "Parameters", "e_mass", type_func = float)
        self.m_i = self.get_config_value(config, "Parameters", "i_mass", type_func = float)
        self.m_n = self.get_config_value(config, "Parameters", "n_mass", type_func = float)
        
        # Ions
        self.i_system = self.get_config_value(config, "Ions", "system")
        self.i_ics = self.get_config_value(config, "Ions", "ics", mandatory = False, default=None)
        self.T_i = self.get_config_value(config, "Ions", "T_i", mandatory = False, default = 0, type_func = int)
        self.n_i = self.get_config_value(config, "Ions", "n_i", mandatory = False, default = 0, type_func = int)
        self.n_ppc = self.get_config_value(config, "Ions", "n_ppc", mandatory = False, default = 0, type_func = int)
        self.particle_flow_type = self.get_config_value(config, "Ions", "flow_type", mandatory = False, default = None)
        self.seeding_per_timestep = self.get_config_value(config, "Ions", "seeding_per_timestep", mandatory = False, default = 0, type_func = int)
        self.bounce_back_multiplier = self.get_config_value(config, "Ions", "bounce_back_multiplier", mandatory = False, default = 1.0, type_func = float)
        self.max_number_of_bounces = self.get_config_value(config, "Ions", "max_number_of_bounces", mandatory = False, default = 3, type_func = int)
        
        # self.n_n = self.n_flow_rate / (self.m_n * self.ylim[1] * self.v_n)
        
        self.n_particles = self.n_ppc * self.nx * self.ny

        # Neutrals
        self.n_system = self.get_config_value(config, "Neutrals", "system", default = "advection1d")
        self.n_n = self.get_config_value(config, "Neutrals", "n_n", mandatory = False, default = 0, type_func = int)
        self.T_n = self.get_config_value(config, "Neutrals", "T_n", mandatory = False, default = 0, type_func = int)
        self.flux_n = self.get_config_value(config, "Neutrals", "flux_n", mandatory = False, default = 5e-6, type_func = float)
        self.v_n = self.get_config_value(config, "Neutrals", "v_n", mandatory = False, default = 150, type_func = float)
        
        self.n_particles = self.n_ppc * self.nx_with_ghosts * self.ny_with_ghosts
        
        # Electric Field
        self.B_ics = self.get_config_value(config, "Fields", "B_ics")
        
        self.internal_grid = np.zeros((self.nx, self.ny))
        
        # TODO
        self.B_max = self.get_config_value(config, "Fields", "B_max", type_func = float)
        self.V_anode = self.get_config_value(config, "Fields", "V_anode", type_func = int)
        self.V_cathode = self.get_config_value(config, "Fields", "V_cathode", type_func = int)

        # RF Discharge parameters (optional)
        self.rf_frequency = self.get_config_value(config, "Fields", "rf_frequency", type_func = float, mandatory=False, default=None)  # Hz
        self.rf_amplitude = self.get_config_value(config, "Fields", "rf_amplitude", type_func = float, mandatory=False, default=None)  # V
        self.rf_boundary = self.get_config_value(config, "Fields", "rf_boundary", type_func = str, mandatory=False, default=None)  # which boundary oscillates

        # Output
        self.output_freq = self.get_config_value(
            config, "Output", "output_freq", type_func=int, mandatory=False, default=1
        )
        self.output_dir = self.get_config_value(
            config, "Output", "output_dir", mandatory=False, default="output/"
        )
        self.make_movie = self.get_config_value(
            config,
            "Output",
            "make_movie",
            type_func=lambda x: x.lower() == 'true',
            mandatory=False,
            default=True,
        )

        # Add containers for plotting data
        self.data_2d = self.get_config_value(
            config,
            "Output",
            "2d_data",
            type_func=lambda x: [item.strip() for item in x.split(",")]
        )
        # self.data_1d = [] 


    def get_config_value(self, config, section, option, type_func=str, mandatory=True, default=None):
        try:
            # Get raw value from config
            value = config.get(
                section, option, fallback=default if not mandatory else None
            )

            # If the value is literally the string "None", treat it as Python None
            if isinstance(value, str) and value.strip().lower() == "none":
                value = None

            # Raise error if still None and mandatory
            if value is None and mandatory:
                raise ValueError(
                    f"Missing mandatory argument: '{option}' in section '{section}'"
                )

            # Apply type conversion
            return type_func(value) if value is not None else None

        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            if mandatory:
                raise ValueError(
                    f"Missing mandatory argument: '{option}' in section '{section}'"
                ) from e
            return default

        except ValueError as e:
            raise ValueError(
                f"Type conversion error for '{option}' in section '{section}': {e}"
            ) from e
        
    def parse_bounds(self, bounds_str):
        try:
            bounds = ast.literal_eval(bounds_str)
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError("Bounds must be a tuple of two numbers")
            return (float(bounds[0]), float(bounds[1]))
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid bounds format. Expected (min, max), got {bounds_str}") from e

    def parse_bcs(self, bcs_str):
        try:
            # Remove parentheses and split by comma
            bcs = bcs_str.strip('()').split(',')
            if len(bcs) != 2:
                raise ValueError("Boundary conditions must be a tuple of two strings")
            # Strip whitespace from each boundary condition
            return tuple(bc.strip().lower() for bc in bcs)
        except Exception as e:
            raise ValueError(f"Invalid boundary conditions format. Expected (bc_x, bc_y), got {bcs_str}") from e

