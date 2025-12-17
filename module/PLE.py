import torch
from torch import nn

class Customized_Gate_Control(nn.Module):
    def __init__(self, in_dim, Expert_dim, drop_out, Tasknum): 
        super(Customized_Gate_Control, self).__init__()
        self.Tasknum = Tasknum

        #Task_specific
        self.specific_experts = nn.ModuleList()
        for _ in range(Tasknum):
            expert = nn.Sequential(
                nn.Linear(in_dim, Expert_dim),
                nn.GELU(),
                nn.Dropout(drop_out)
            )
            self.specific_experts.append(expert)
        
        #Task_shared
        self.Shared_experts = nn.Sequential(nn.Linear(in_dim, Expert_dim), nn.GELU(), nn.Dropout(drop_out))

        # Integrate_gate
        self.Gates = nn.ModuleList()
        for _ in range(Tasknum):
            gate_liner = nn.Sequential(nn.Linear(in_dim, 2), nn.Softmax(dim=1))
            self.Gates.append(gate_liner)
                

    def forward(self, X_tasks, X_s):
        """
        X_tasks dim: (task_num, batch, embeding_dim*n_graph)
        """
        #Specific_expert_outs
        Specific_expert_outs = [] #(Tasknum, batch, embedding_dim*n_graph)
        for task_index in range(self.Tasknum):
            x_task = X_tasks[task_index] #(batch, embedding_dim*n_graph)
            specific_model = self.specific_experts[task_index]
            specific_out = specific_model(x_task)
            Specific_expert_outs.append(specific_out)
        
        #Shared_expert_out
        Shared_expert_out = self.Shared_experts(X_s) #(batch, embeding_dim*n_graph)

        #Integrate_gate
        Gate_task_out = [] # (Tasknum, batch, 2)
        for gate_model, x_task in zip(self.Gates, X_tasks):
            gate_weight = gate_model(x_task) # (batch, 2)
            Gate_task_out.append(gate_weight)

        #Integrate_out
        Integrate_outs = [] #(Tasknum, batch, embeding_dim*n_graph)
        for gate, specific_expert in zip(Gate_task_out, Specific_expert_outs):
            gate = gate.unsqueeze(2) #(batch_size, 2, 1)
            specific_shared = torch.stack([specific_expert, Shared_expert_out], dim = 1) #(batch_size, 2, embeding_dim*n_graph)
            task_out = torch.matmul(specific_shared.transpose(1,2), gate)
            task_out = task_out.squeeze(2) #(batch_size, embeding_dim*n_graph)
            Integrate_outs.append(task_out)
        return Integrate_outs


class PLE(nn.Module):
    def __init__(self, in_dim, Expert_dim, tower_hidden, Tasknum, drop_out=0.2):
        super(PLE , self).__init__()
        self.Tasknum = Tasknum
        self.cgc = Customized_Gate_Control(in_dim, Expert_dim, drop_out, Tasknum)
        self.clfs = nn.ModuleList([nn.Sequential(
            nn.Linear(Expert_dim, tower_hidden), nn.GELU(),
            nn.Linear(tower_hidden, 1))  
            for _ in range(Tasknum)])    
        
    def forward(self , x):
        """
        x dim: (batch, embeding_dim*n_graph)
        """
        integrate_outs = self.cgc([x for _ in range(self.Tasknum)], x)
        x_clf = [clf_model(integrate_out) for clf_model, integrate_out in zip(self.clfs, integrate_outs)]
        out = torch.cat(x_clf , dim = 1)
        return out