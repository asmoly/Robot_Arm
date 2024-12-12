from math import cos, sin, pi

class Matrix:
    def __init__(self, col_a, col_b, col_c) -> None:
        self.col_a = col_a
        self.col_b = col_b
        self.col_c = col_c

    def __mul__(self, b):
        if type(b) == Matrix:
            col_a = Vector(self.col_a.x*b.col_a.x + self.col_b.x*b.col_a.y + self.col_c.x*b.col_a.z,
                           self.col_a.y*b.col_a.x + self.col_b.y*b.col_a.y + self.col_c.y*b.col_a.z,
                           self.col_a.z*b.col_a.x + self.col_b.z*b.col_a.y + self.col_c.z*b.col_a.z)
            
            col_b = Vector(self.col_a.x*b.col_b.x + self.col_b.x*b.col_b.y + self.col_c.x*b.col_b.z,
                           self.col_a.y*b.col_b.x + self.col_b.y*b.col_b.y + self.col_c.y*b.col_b.z,
                           self.col_a.z*b.col_b.x + self.col_b.z*b.col_b.y + self.col_c.z*b.col_b.z)
            
            col_c = Vector(self.col_a.x*b.col_c.x + self.col_b.x*b.col_c.y + self.col_c.x*b.col_c.z,
                           self.col_a.y*b.col_c.x + self.col_b.y*b.col_c.y + self.col_c.y*b.col_c.z,
                           self.col_a.z*b.col_c.x + self.col_b.z*b.col_c.y + self.col_c.z*b.col_c.z)
            
            return Matrix(col_a, col_b, col_c)
        if type(b) == Vector:
            return Vector(b.x*self.col_a.x + b.y*self.col_b.x + b.z*self.col_c.x,
                          b.x*self.col_a.y + b.y*self.col_b.y + b.z*self.col_c.y,
                          b.x*self.col_a.z + b.y*self.col_b.z + b.z*self.col_c.z)

    def __str__(self):
        return f"[{self.col_a.x}, {self.col_b.x}, {self.col_c.x}]\n[{self.col_a.y}, {self.col_b.y}, {self.col_c.y}]\n[{self.col_a.z}, {self.col_b.z}, {self.col_c.z}]"

    @staticmethod
    def get_rot_mat(x, y, z, rad=True):
        if rad == False:
            x = x*pi/180
            y = y*pi/180
            z = z*pi/180

        x_rot_mat = Matrix(Vector(1, 0, 0), Vector(0, cos(x), sin(x)), Vector(0, -sin(x), cos(x)))
        y_rot_mat = Matrix(Vector(cos(y), 0, -sin(y)), Vector(0, 1, 0), Vector(sin(y), 0, cos(y)))
        z_rot_mat = Matrix(Vector(cos(z), sin(z), 0), Vector(-sin(z), cos(z), 0), Vector(0, 0, 1))

        return z_rot_mat*y_rot_mat*x_rot_mat

class Vector:
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, b):
        if type(b) == Vector:
            return Vector(self.x + b.x, self.y + b.y, self.z + b.z)

    def __mul__(self, b):
        if type(b) == int:
            return Vector(self.x*b, self.y*b, self.z*b)
        
    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
    def as_list(self):
        return [self.x, self.y, self.z]