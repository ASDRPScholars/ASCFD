class ParticleConstants:
    def __init__(self, dimensions=2):
        self.XCOMP = 0
        self.YCOMP = 1
        self.UCOMP = 2  # vx
        self.VCOMP = 3  # vy
        self.WCOMP = 4  # vz (even in 2D, particles can have z-velocity)
        if dimensions == 3:
            self.ZCOMP = 5  # z position
            self.NUMQ = 6
        else:
            self.NUMQ = 5  # 2D position + 3D velocity

        E_CHARGE = 1.602176e-19  
        ELECTRON_MASS = 9.1093837e-31
