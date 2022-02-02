import torch
import torch.nn as nn
import torch.optim as optim


def gamma_initial_grad(net, train_dataloader, lr, device='cpu'):
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)

    for tmp in train_dataloader:
        for batch_idx, (x, target) in enumerate(tmp):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            loss.backward()

    g_i = []
    for p in net.parameters():
        if p.requires_grad:
            g_i.append(torch.clone(p.grad /len(train_dataloader[0])))
        else:
            g_i.append(torch.zeros(p.size(), dtype=torch.float).to(device))
    net.zero_grad()
    optimizer.zero_grad()
    #copy and return the gradient

    return g_i

def get_gamma(net, train_dataloader, g_i, global_weight_collector, lr, mu, device='cpu'):
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)

    trained_model = list(net.to(device).parameters())
    for tmp in train_dataloader:
        for batch_idx, (x, target) in enumerate(tmp):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            loss.backward()

    inexact_g_i = []
    for p in net.parameters():
        if p.requires_grad:
            inexact_g_i.append(torch.clone(p.grad /len(train_dataloader[0])))
        else:
            inexact_g_i.append(torch.zeros(p.size(), dtype=torch.float).to(device))

    #caculate gamma
    h_inexa = 0
    h_wt = 0
    for indx in range(len(g_i)):
        h_inexa += torch.sum((g_i[indx] + mu * (trained_model[indx] - global_weight_collector[indx]) )**2 ).to(torch.float)
        h_wt += torch.sum(g_i[indx] **2)

    gamma = torch.sqrt(h_inexa) / torch.sqrt(h_wt)
    return gamma


def dynfedprox_mu(args, stage, round_gamma_list):
    # print("Inside gamma_based_linear_three_stage")
    #if we want to discount the mu base on something else just change the if below
    if stage == 'init':
        if len(round_gamma_list)>=3 and list_less_than(round_gamma_list[-3:], 1):
            stage = 'exp'
        #if gamma keep larger than 1, we increase the epoch for the gamma decreasing.
        elif round_gamma_list[-1]>=1:
            args.epochs += args.delta_epoch
    
    elif stage == 'exp':
        if round_gamma_list[-1]>=1:
            stage = 'linear'
            args.mu *= args.discount_rate
            # if args.dataset == 'Shakespeare':
            #     args.delta_mu = 0.0001
            # else:
            #     args.delta_mu = 0.1
        else:
            args.mu *= args.exponential_growth_rate
    elif stage == 'linear': 
        if round_gamma_list[-1]>=1:
            args.mu *= args.discount_rate
        else:
            args.mu += args.linear_growth_rate

def list_less_than(l, value=1):
    flag = True
    for i in l:
        if(i>=value):
            flag = False
    return flag