import torch
from torch.optim.optimizer import Optimizer, required
import torch.functional as F

class NSGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                    weight_decay=0, nesterov=False, memory=False):
            
            self.P = 1
            self.Q = 1
            self.beta = 0.5
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
                    param_state['memory'] = torch.zeros_like(p.data, device=torch.device('cuda'))

        

    def step(self, closure=None):
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

                # d_p corresponds to g in alg. 1 from the paper.
                # param_state['gradient'] = d_p  # Save the gradient so its norm can be computed later

                # d_p = group['lr'] * d_p
                
                corrected_gradient = param_state['memory'] + d_p
                
                corrected_gradient = self.__compress__(corrected_gradient)

                
                param_state['memory'] = param_state['memory'] + d_p - corrected_gradient
                
                d_p_1 = self.beta * d_p  + (1-self.beta)*corrected_gradient/self.rho

                neta = 1/(self.P*F.norm(d_p_1) + self.Q)
                
                v_t_1 = p.data - neta*d_p_1
                rand_ = self.__randomize__()

                p.data = p.data * (1-rand_) + rand_*v_t_1

        return loss

    
    def __randomize__(self):
        return torch.rand(1)
    

    def __compress__(self,x,input_compress_settings={}):
        max_iteration=10000
        compress_settings={'p':0.8}
        compress_settings.update(input_compress_settings)
        #p=compress_settings['p']
        #vec_x=x.flatten()
        #out=torch.dropout(vec_x,1-p,train=True)
        #out=out/p
        vec_x=x.flatten()
        d = int(len(vec_x))
        p=compress_settings['p']
        
        abs_x=torch.abs(vec_x)
        #d=torch.prod(torch.Tensor(x.size()))
        out=torch.min(p*d*abs_x/torch.sum(abs_x),torch.ones_like(abs_x, device=torch.device('cuda')))
        i=0
        while True:
            i+=1
            #print(i)
            if i>=max_iteration:
                raise ValueError('Too much operations!')
            temp=out.detach()
                
            cI=1-torch.eq(out,1).float()
            c=(p*d-d+torch.sum(cI))/torch.sum(out*cI)
            if c<=1:
                break
            out=torch.min(c*out,torch.ones_like(out, device=torch.device('cuda')))
            if torch.sum(~torch.eq(out,temp)):
                break
        
        z=torch.rand_like(out, device=torch.device('cuda'))
        out=vec_x*(z<out).float()/out

        out=out.reshape(x.shape)

        #out=out.reshape(x.shape)
        return out


