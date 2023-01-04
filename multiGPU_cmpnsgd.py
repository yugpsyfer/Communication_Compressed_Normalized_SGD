import torch
import torch.functional as F
import torch.distributed as dist


class CNSGD():
    def __init__(self, params, p=1, q=10, beta=0.9, rho=1):
        """
        Default values have been taken from the paper https://arxiv.org/pdf/2002.04130.pdf
        Rho is number of workers
        By default I am taking a single parameter group
        """
        defaults = {"P":p, "Q":q, "Beta":beta, "Rho":rho}
        
        self.param_groups = [{"params":params, "memory": [], "M":[]} + defaults]
        
        for g in self.param_groups:
            for pars in g['params']:
                g['memory'].append(torch.zeros_like(pars.data, device=torch.device('cuda')))
                g['M'].append(torch.zeros_like(pars.data, device=torch.device('cuda')))
                
        
    def step(self):
        for groups in self.param_groups:
            P = groups['P']
            Q = groups['Q']
            Beta = groups['Beta']
            Rho = groups['Rho']

            for par in range(len(groups['params'])):
                
                grad = groups['params'][par].grad.data  #worker
                m_prev =  groups['M'][par]

                corrected_grad, groups['memory'][par] = self.correct_gradients(grad, groups['memory'][par]) #worker
                
                dist.barrier()
                corrected_grad = dist.all_reduce(corrected_grad)

                m_t = Beta * m_prev  + (1-Beta)*corrected_grad/Rho
                groups['M'][par] = m_t    
                
                neta = 1/(P*F.norm(m_t) + Q)

                v_t = groups['params'][par].data - neta*m_t
                
                rand_ = self.randomize()
                
                groups['params'][par].data = groups['params'][par].data * (1-rand_) + rand_*v_t


    def compress(self, x):  #worker
        
        x_norm=torch.norm(x,p=float('inf'))
        sgn_x=((x>0).float()-0.5)*2
        compressed_x=x_norm*sgn_x
        
        return compressed_x 
    
    
    def randomize(): #worker
        return torch.rand(1, device=torch.device('cuda'))
    
    
    def correct_gradients(self, x, mem): #worker
        
        corrected_gradient = self.compress(mem + x)
        mem = mem + x - corrected_gradient

        return corrected_gradient, mem