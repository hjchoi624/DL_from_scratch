import numpy as np
import weakref
import numpy as np


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
    
    
class Variable:
    def __init__(self, data):
        # if data is not None:
        #     if not isinstance(data,np.ndarray):
        #         raise TypeError(f'{type(data)} 은 지원하지 않습니다.')
        
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 
        
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self, retain_grad=False):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad  = f.backward(self.grad)
            x.backward()
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = []
        seen_set = set()
        
        def __mul__(self, other):
            return mul(self, other)
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.append(f)
                funcs.sort(key=lambda x:x.generation)
                
        add_func(self.creator)
        
        while funcs:
            f =  funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
                
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                    
                else:
                    x.grad = x.grad + gx
                    
                if x.creator is not None:
                    add_func(x.creator)
                    
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None
         
           
class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        
        xs = [x.data for x in inputs]
        ys = [self.forward(*xs)]
        if not isinstance(ys, tuple):
            ys = (ys,)
            
        outputs = [Variable(as_array(y)) for y in ys]
        
        self.generation = max([x.generation for x in inputs])
        
        for output in outputs:
            output.set_creator(self)
            
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
        
    def forward(self, x):
        pass
    
    def backward(self, gy):
        pass
    
# class Function:
#     def __call__(self, *inputs):
#         inputs = [as_variable(x) for x in inputs]
        
#         xs = [x.data for x in inputs]
#         ys = [self.forward(*xs)]
        
#     def forward(self, x):
#         pass
    
#     def backward(self, gy):
#         pass
    
def add(x0, x1) :
    return Add()(x0, x1)
 
        
if __name__ == "__main__":
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(x0, x1)
    print(y)
    

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y 

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        
        return gy * x1, gy * x0


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y 
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
    
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp((x)) * gy
        return gx
    
    
class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy
    
class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

class Pow(Function) :
    def __init__(self, c):
        self.c = c
        
    def forward(self,x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c + x ** (c -1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv

import unittest

'''
유닛테스트
'''


#기울기 확인을 이용한 자동 테스트
def numeric_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (t1.data - y0.data) / (2 * eps)

class SquareTest(unittest.TestCase):
    ...
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        y.backward()
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
        num_grad = numeric_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
        
        
'''
python -m unittest discover tests
'''