import numpy as np


OUTPUT = ['B', 'C', 'N', 'O', 'Mg', 'Al', 'Si', 'P', 'S', 'K', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'As', 'Se', 'Mo', 'Nb', 'Zr', 'Ce', 'W', 'Pb', 'Sb', 'Bi', 'Cd', 'Te', 'Sn', 'Ta']

class ModelPipeline:
    def __init__(self, reg, clas):
        self.reg = reg
        self.clas = clas

    def arch(self, x):
        x1 = self.reg.predict(x)
        x2 = np.hstack((x, x1))
        x2 = self.clas.predict(x2)
        y = np.multiply(x1, x2)
        return y

    def __call__(self, x):
        input_x = []
        for obj in x:
            exmpl = []
            exmpl.append(float(obj['HB']))
            exmpl.append(float(obj['Ultimate_strength']))
            exmpl.append(float(obj['E']))
            exmpl.append(float(obj['ro']))
            exmpl.append(float(obj['c']))
            input_x.append(exmpl)
        input_x = np.array(input_x)
        output_y = self.arch(input_x)
        interpretate_output = []
        for i in range(output_y.shape[0]):
            obj_dict = {}
            for j in range(output_y.shape[1]):
                if output_y[i, j] > 0.0:
                    obj_dict[OUTPUT[j]] = round(output_y[i, j], 3)
            interpretate_output.append(obj_dict)
        return interpretate_output
    
