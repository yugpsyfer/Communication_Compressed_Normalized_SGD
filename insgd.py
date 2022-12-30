import torch
from torch.optim.optimizer import Optimizer, required
import torch.functional as F

class INSGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                    weight_decay=0, nesterov=False, memory=False):
            
            self.P = 1
            self.Q = 10
            self.beta = 0.9
            self.rho=1

            if lr is not required and lr < 0.0:
                raise ValueError("Invalid learning rate: {}".format(lr))
            if momentum < 0.0:
                raise ValueError("Invalid momentum value: {}".format(momentum))
            if weight_decay < 0.0:
                raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

            defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                            weight_decay=weight_decay, nesterov=nesterov,
                            memory=memory)

            if nesterov and (momentum <= 0 or dampening != 0):
                raise ValueError("Nesterov momentum requires a momentum and zero dampening")
            super().__init__(params, defaults)

            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['M'] = torch.zeros_like(p.data, device=torch.device('cuda'))



    def step(self,closure=None):
            

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                param_state = self.state[p]
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data, device=torch.device('cuda'))
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                

                param_state['M'] = self.beta*param_state['M'] + (1-self.beta)*d_p

                neta = 1/(self.P*F.norm(param_state['M']) + self.Q)
                
                x_t_1 = p.data - neta*param_state['M']
                r = self.__randomize__()
                p.data = p.data * r + (1-r)*x_t_1
                
                
        return loss

    
    def __randomize__(self):
        return torch.rand(1, device=torch.device('cuda'))
    


