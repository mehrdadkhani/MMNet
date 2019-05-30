from gurobipy import *
import cvxpy as cp
import numpy as np

class solvers:
    def ampSolver(self, hBatch, yBatch, Symb, noise_sigma):
        def F(x_in, tau_l, Symb):
            arg = -(x_in - Symb.reshape((1,1,-1))) ** 2 / 2. / tau_l
            exp_arg = np.exp(arg - np.max(arg, axis=2, keepdims=True))
            prob = exp_arg / np.sum(exp_arg, axis=2, keepdims=True)
            f = np.matmul(prob, Symb.reshape((1,-1,1)))
            return f

        def G(x_in, tau_l, Symb):
            arg = -(x_in - Symb.reshape((1,1,-1))) ** 2 / 2. / tau_l
            exp_arg = np.exp(arg - np.max(arg, axis=2, keepdims=True))
            prob = exp_arg / np.sum(exp_arg, axis=2, keepdims=True)
            g = np.matmul(prob, Symb.reshape((1,-1,1)) ** 2) - F(x_in, tau_l, Symb) ** 2
            return g

        numIterations = 50
        NT = hBatch.shape[2]
        NR = hBatch.shape[1]
        N0 = noise_sigma ** 2 / 2.
        xhat = np.zeros((numIterations, hBatch.shape[0], hBatch.shape[2], 1))
        z = np.zeros((numIterations, hBatch.shape[0], hBatch.shape[2], 1))
        r = np.zeros((numIterations, hBatch.shape[0], hBatch.shape[1], 1))
        tau = np.zeros((numIterations, hBatch.shape[0], 1, 1))
        r[0] = yBatch
        for l in range(numIterations-1):
            z[l] = xhat[l] + np.matmul(hBatch.transpose((0,2,1)), r[l])
            xhat[l+1] = F(z[l], N0 * (1.+tau[l]), Symb)
            tau[l+1] = float(NT) / NR / N0 * np.mean(G(z[l], N0 * (1. + tau[l]), Symb), axis=1, keepdims=True)
            r[l+1] = yBatch - np.matmul(hBatch, xhat[l+1]) + tau[l+1]/(1.+tau[l]) * r[l]

        return xhat[l+1]
    def sdrSolver(self, hBatch, yBatch, constellation, NT):
        results = []
        for i, H in enumerate(hBatch):
            y = yBatch[i]
            s = cp.Variable((2*NT,1))
            S = cp.Variable((2*NT, 2*NT))
            objective = cp.Minimize(cp.trace(H.T @ H @ S) - 2. * y.T @ H @ s)
            constraints = [S[i,i] <= (constellation**2).max() for i in range(2*NT)]
            constraints += [S[i,i] >= (constellation**2).min() for i in range(2*NT)]
            constraints.append(cp.vstack([cp.hstack([S,s]), cp.hstack([s.T,[[1]]])]) >> 0)
            prob = cp.Problem(objective, constraints)
            result = prob.solve()
            results.append(s.value)
        results = np.array(results)
        print(results.shape)
        return results
            
    def vaSolver(self, hBatch, hthBatch, yBatch, NT):
        constellation = np.array([-3,-1,1,3])
        alpha = np.sqrt(np.mean(constellation**2))
        # [-1,1] maps to [-3,-1,
        results = []
        NT *= 2
        W = np.concatenate([np.eye(NT), 2 * np.eye(NT)], axis=1)
        print(W.shape)
        for idx_, Y in enumerate(yBatch):
            if idx_ % 10 == 0:
                print(idx_ / float(len(yBatch)) * 100. ,"% completed")
            hth = hthBatch[idx_]
            H = hBatch[idx_]
            H2 = np.matmul(W.transpose(), hth)
            H2 = np.matmul(H2, W)
            G = np.matmul(H,W)
            print(G.shape)
            model = Model('VA')
            B = model.addVars(2*NT, 2*NT, name='B')
            b = model.addVars(2*NT, vtype=GRB.BINARY, name='b')
          
            DC1 = model.addVars(2*NT, name='DC1')
            for k in DC1:
                model.addConstr(DC1[k]==sum(H2[int(k),j] * B[j,int(k)] for j in range(2*NT)))
            
            traceH2B = model.addVar(name='traceH2B')
            model.addConstr(traceH2B==quicksum(DC1[i] for i in range(2*NT)))
        
            rhs = np.matmul(np.matmul(W.transpose(),H.transpose()), Y)
            #obj = traceH2B - 2 * sum([b[i]*rhs[i] for i in range(2*NT)])
            HtG = np.matmul(H.T, G)
            print(HtG.shape)
            obj = (4./alpha**2) * traceH2B - 4/alpha * sum([b[i]*rhs[i] for i in range(2*NT)]) - 12./alpha * sum([sum([HtG[i,j]*b[j] for j in range(2*NT)]) for i in range(HtG.shape[0])])
            model.setObjective(obj, GRB.MINIMIZE)
           
            for i in range(2*NT):
                for j in range(2*NT):
                    model.addConstr(B[i,j] >= b[i] * b[j])
           
            model.addConstrs(B[i,i]==1. for i in range(2*NT))
           
            model.Params.logToConsole = 0
            model.update()
            model.optimize()      
            solution = model.getAttr('X', b)
            x_est = []
            for k in solution:
                x_est.append(k)
            x_est = np.array(x_est)
            results.append(x_est[:NT]+2.*x_est[NT:])
        results = np.array(results)
        results = (2. * results - 3.)/alpha
        
        return results

    #def mlSolver(self, hBatch, yBatch, m, n, x_baseleline, Symb):
    def mlSolver(self, hBatch, yBatch, Symb):
        results = []
        status = []
        m = len(hBatch[0,0,:])
        n = len(hBatch[0,:,0])
        print(m, n)
        #Symb = list(Symb)
        #print Symb
        for idx_, Y in enumerate(yBatch):
            if idx_ % 10 == 0:
                print(idx_ / float(len(yBatch)) * 100. ,"% completed")
        #Y = list(Y)
            #H = list(hBatch[idx_])
            H = hBatch[idx_]
            model = Model('mimo')
            k = len(Symb)
            Z = model.addVars(m, k, vtype=GRB.BINARY, name='Z')
            S = model.addVars(m, ub=max(Symb)+.1, lb=min(Symb)-0.1,  name='S')
            E = model.addVars(n, ub=200.0, lb=-200.0, vtype=GRB.CONTINUOUS, name='E')
            model.update() 
            
            # Defining S[i]
            for i in range(m):
                model.addConstr(S[i] == quicksum(Z[i,j] * Symb[j] for j in range(k)))
                #S[i] == quicksum(Z[i,j] * Symb[j] for j in range(k))
            
            # Constraint on Z[i,j]
            model.addConstrs((Z.sum(j,'*') == 1
                             for j in range(m)), name='Const1')
            
            
            # Defining E
            for i in range(n):
                E[i] = quicksum(H[i][j] * S[j] for j in range(m)) - Y[i][0]
            #for i in range(m):
            #    E[i] = quicksum(quicksum(H[l,i]*H[l,j] for l in range(n)) * S[j] 
            #            - quicksum(H[ll,i]*Y[ll] for ll in range(n))  
            #            for j in range(m)) 
            
            # Defining the objective function
            obj = E.prod(E)  
            model.setObjective(obj, GRB.MINIMIZE)
            model.Params.logToConsole = 0
            model.setParam('TimeLimit', 100)
            model.update()
             
            model.optimize()
            
            #model.write('MIMO_BPSK.txt')
            
            #print('')
            #print('Solution:')
            #print('')
            
            # Retrieve optimization result
            solution = model.getAttr('X', S)
            status.append(model.getAttr(GRB.Attr.Status) == GRB.OPTIMAL)
            print(GRB.OPTIMAL, model.getAttr(GRB.Attr.Status))
            if model.getAttr(GRB.Attr.Status)==9:
                print(np.linalg.cond(H))
            x_est = []
            for nnn in solution:
                 x_est.append(solution[nnn])
            results.append(x_est)
        return results, np.array(status)
    
#    def BQP_solver_mosek(self, H, b, m, n, x_baseline):
#        results = []
#        cnt_ = 0
#        for ii, y in enumerate(b):
#            cnt_ += 1
#            y = np.reshape(y, (n,1))
#            y2 = y + np.matmul(H, np.ones((m,1)))
#            H2 = 2 * H
#            Q11 = np.matmul(np.matrix.transpose(H2), H2)
#            Q12 = -1. *  np.matmul(np.matrix.transpose(H2), y2)
#            Q21 = np.matrix.transpose(Q12)
#            Q22 = np.matmul(np.matrix.transpose(y), y)
#            
#            Q1 = np.concatenate([Q11, Q12], axis= 1)
#            Q2 = np.concatenate([Q21, Q22], axis= 1)
#            
#            Q = np.concatenate([Q1,Q2], axis= 0) 
#            P = np.zeros(m+1)
#            R = 0.0
#            
#            solver = BB(Q, P, R)
#            solver.solve()
#            
#            t_star = solver.xx[-1]
#            if t_star==1:
#                s_star = solver.xx[0:-1]
#            else:
#                s_star = 1 - solver.xx[0:-1]
#            results.append(s_star)
#            if cnt_ % 100 == 0:
#                print cnt_
#            if not (s_star == x_baseline[ii]).all():
#                print s_star, x_baseline[ii]
#                solver.summary()
#                return
#        return results
#    def PAM4_solver(self, H, b, m, n):
#        #m = 30
#        #n = 60
#        #H = np.random.normal(0.0, 1.0, (n,m))
#        #x_labels = np.random.randint(0, 2, size =(m,1))
#        #x = 2 * x_labels - 1
#        #y = np.matmul(H,x) + np.random.normal(0, 1., size=(n, 1))
#        #print y.shape
#        W = np.concatenate([np.eye(m), 2*np.eye(m)], axis=1)
#        H = np.matmul(H,W)
#
#        results = []
#        cnt_ = 0
#        for y in b:
#            cnt_ += 1
#            y = np.reshape(y, (n,1))
#            y2 = y + np.matmul(H, np.ones((2*m,1)))
#            H2 = 2 * H
#            Q11 = np.matmul(np.matrix.transpose(H2), H2)
#            Q12 = -1. *  np.matmul(np.matrix.transpose(H2), y2)
#            Q21 = np.matrix.transpose(Q12)
#            Q22 = np.matmul(np.matrix.transpose(y), y)
#            
#            Q1 = np.concatenate([Q11, Q12], axis= 1)
#            Q2 = np.concatenate([Q21, Q22], axis= 1)
#            
#            Q = np.concatenate([Q1,Q2], axis= 0) 
#            P = np.zeros(2*m+1)
#            R = 0.0
#            print Q.shape, P.shape
#            solver = BB(Q, P, R)
#            solver.solve()
#            
#            t_star = solver.xx[-1]
#            if t_star==1:
#                s_star = solver.xx[0:-1]
#            else:
#                s_star = 1 - solver.xx[0:-1]
#            
#            nonbool_s = 2 * s_star - 1
#            s1 = nonbool_s[0:m]
#            s2 = nonbool_s[m::]
#            S = s1 + 2 * s2
#            results.append(S)
#            if cnt_ % 100 == 0:
#                print cnt_
#        return results
