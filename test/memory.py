# import numpy as np
# #from calc import Mul

# def mul(x0, x1):
#     return Mul()(x0, x1)

# def as_variable(obj):
#     if isinstance(obj, Variable):
#         return obj
#     return Variable(obj)


# class Variable:
#     def backward(self, retain_grad=False):
#         if self.grad is None:
#             self.grad = np.ones_like(self.data)
        
#         funcs = []
#         seen_set = set()
        
#         def __mul__(self, other):
#             return mul(self, other)
#         def add_func(f):
#             if f not in seen_set:
#                 funcs.append(f)
#                 seen_set.append(f)
#                 funcs.sort(key=lambda x:x.generation)
                
#         add_func(self.creator)
        
#         while funcs:
#             f =  funcs.pop()
#             gys = [output().grad for output in f.outputs]
#             gxs = f.backward(*gys)
            
#             if not isinstance(gxs, tuple):
#                 gxs = (gxs, )
                
#             for x, gx in zip(f.inputs, gxs):
#                 if x.grad is None:
#                     x.grad = gx
                    
#                 else:
#                     x.grad = x.grad + gx
                    
#                 if x.creator is not None:
#                     add_func(x.creator)
                    
#             if not retain_grad:
#                 for y in f.outputs:
#                     y().grad = None
                    
                    
# class Function:
#     def __call__(self, *inputs):
#         inputs = [as_variable(x) for x in inputs] #ndarray 인스턴스와 사용
#         xs = [x.data for x in inputs]
#         ys = self.forward(*xs)
#         if not isinstance(ys, tuple):
#             ys = (ys,)
            
#         outputs = [Variable(as_array(y)) for y in ys]
        
#         if Config.enable_backprop:
#             self.generation = max([x.generation for x in inputs])
            
#             for output in outputs:
#                 output.set_creator(self)
                
#             self.inputs = inputs
#             self.inputs = inputs
#             self.outputs = [weakref.ref(output) for output in outputs]
        
#         return outputs if len(outputs)>1 else outputs[0]
    
# class Config:
#     enable_backprop = True
    

        
        