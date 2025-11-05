class ParticleConstants:
    def __init__(self, dimensions=2):
        self.XCOMP = 0
        self.YCOMP = 1
        self.UCOMP = 2  # vx
        self.VCOMP = 3  # vy
        self.WCOMP = 4
        
        self.NUMQ = 5 if dimensions == 3 else 4

        self.WEIGHT = self.NUMQ
        
        self.k_B = 8.617e-5