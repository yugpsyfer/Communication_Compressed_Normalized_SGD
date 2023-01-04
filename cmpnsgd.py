import torch
from torch.optim.optimizer import Optimizer, required
import torch.functional as F
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class NSGD(Optimizer):
    def __init__(self, params, p=1, q=10, beta=0.9, rho=1):
            defaults = {"p":p,"q":q, "beta":beta, "rho":rho}

            super().__init__(params, defaults)

            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['memory'] = torch.zeros_like(p.data, device=torch.device('cuda'))
                    param_state['M'] = torch.zeros_like(p.data, device=torch.device('cuda'))


    def step(self, closure=None):
        loss = None
        
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            P = group['p']
            Q = group['q']
            Beta = group['beta']
            Rho = group['rho']

            for p in group['params']:
                param_state = self.state[p]
                if p.grad is None:
                    continue
                d_p = p.grad.data
                m_prev = param_state['M']
                
                corrected_gradient = param_state['memory'] + d_p
                corrected_gradient = self.__compress__(corrected_gradient)
                # logging.warning(corrected_gradient)

                param_state['memory'] = param_state['memory'] + d_p - corrected_gradient

                m_t = Beta * m_prev  + (1-Beta)*corrected_gradient/Rho
                param_state['M'] = m_t
                
                neta = 1/(P*F.norm(m_t) + Q)

                v_t_1 = p.data - neta*m_t
                rand_ = self.__randomize__()

                p.data = p.data * (1-rand_) + rand_*v_t_1

        return loss

    
    def __randomize__(self):
        return torch.rand(1, device=torch.device('cuda'))


    def __compress__(self, x):
        
        x_norm=torch.norm(x,p=float('inf'))
        sgn_x=((x>0).float()-0.5)*2
        
        compressed_x=x_norm*sgn_x
        
        return compressed_x    

    def __compress1__(self,x,input_compress_settings={}):
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


