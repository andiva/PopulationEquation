import numpy as np
import time
import pandas as pd


def main():
    # initi conditions---------------------------------------------------------
    A=2
    B=0.01
    t0=0.0
    t_end=5
    #--------------------------------------------------------------------------

    # Analytic solution
    analytic = lambda t_ini, y_ini, t, a, b: 1/(b/a+(1/y_ini-b/a)*np.exp(-a*(t-t_ini)))
    #--------------------------------------------------------------------------
    # ODE
    f = lambda y: A*y-B*y*y
    #--------------------------------------------------------------------------
    # initial valuse
    y0_list = np.array([8, 10, 12])
    #y0_list = np.array([100, 190, 250])
    #--------------------------------------------------------------------------

    timesteps = np.array([0.05, 0.025, 0.0125, 0.00625, 0.003125])

    # maps of 4th order for different time steps (A=2, B=0.01)
    # M(0,:) is map for dt=0.05     (N=100)
    # M(1,:) is map for dt=0.025    (N=200)
    # M(2,:) is map for dt=0.0125   (N=400)
    # M(3,:) is map for dt=0.00625  (N=800)
    # M(4,:) is map for dt=0.003125 (N=1600)

    Maps = np.array([[-9.74718510761581e-11,2.82830795154941e-7,-0.00057683855758959,1.1047522448889,0.0164019262867156],
                    [-1.379332373516e-11,6.75987803436825e-8,-0.00026920890283235,1.05124251835198,0.00113151812515],
                    [-1.83493448551926e-12,1.63316814751669e-8,-0.00012976106224663,1.0253132530535,7.43225401875509e-5],
                    [-2.36633829318494e-13,3.99916101441112e-9,-6.36821468173527e-5,1.01257833218163,4.76237954022738e-6],
                    [-3.00445936310731e-14,9.88470260129351e-10,-3.15443221011727e-5,1.00626956445964,3.0138672729979e-7]
                    ])

    #--------------------------------------------------------------------------
    # resulting table
    # result(:, 0) is N
    # result(:, 1) is err_RK4
    # result(:, 2) is err_TM4
    # result(:, 3) is time_RK4
    # result(:, 4) is time_TM4
    # result(:, 5) is time_ratio = time_RK4/time_TM4
    result = np.zeros((len(timesteps), 6))


    for k, dt in enumerate(timesteps):  # for each time stemp dt
        M = Maps[k, :]                 # get TM for this dt
        N = int((t_end-t0)/dt)
        t = np.arange(t0, t_end, dt)

        result[k, 0] = N

        for y0 in y0_list: # for each y0
            # analytic solution------------------------------------------------
            y_sol = analytic(t0, y0, t, A, B)
            #------------------------------------------------------------------

            # RK4 integration--------------------------------------------------
            y_rk4 = np.zeros(N)

            y_rk4[0]=y0
            tic = time.time()
            for i in range(N-1):
                y = y_rk4[i]
                k1 = f(y)
                k2 = f(y+dt*k1/2)
                k3 = f(y+dt*k2/2)
                k4 = f(y+dt*k3)
                y_rk4[i+1] = y + dt*(k1+2*k2+2*k3+k4)/6

            elapsed_time = time.time()-tic
            result[k, 3] += elapsed_time # time_RK4
            #------------------------------------------------------------------

            # Mapping----------------------------------------------------------
            y_map = np.zeros(N)
            y_map[0] = y0
            tic = time.time()
            for i in range(N-1):
                y=y_map[i]
                y2 = y*y
                y_map[i+1] = (y2*(M[0]*y2 + M[2]) +
                            y *(M[1]*y2 + M[3]) +
                            M[4])

            elapsed_time = time.time()-tic
            result[k, 4] += elapsed_time # time_RK4
            #------------------------------------------------------------------
            result[k, 1] += np.abs(y_rk4 - y_sol).max() # err_RK4
            result[k, 2] += np.abs(y_map - y_sol).max() # err_TM4

    result[:, 1:] /= len(y0_list) # get average results
    result[:, 5] = result[:, 3]/result[:, 4] # get time_ratio


    result = pd.DataFrame(data=result[:,1:], index=result[:,0], columns=np.array(['err_RK4', 'err_TM4', 'time_RK4', 'time_TM4', 'time_ratio']))
    return result

if __name__ == "__main__":
    print(main())