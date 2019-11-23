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

    # maps of 6th order for different time steps (A=2, B=0.01)
    # M(0,:) is map for dt=0.05     (N=100)
    # M(1,:) is map for dt=0.025    (N=200)
    # M(2,:) is map for dt=0.0125   (N=400)
    # M(3,:) is map for dt=0.00625  (N=800)
    # M(4,:) is map for dt=0.003125 (N=1600)
    Maps = np.array([[-2.20674242163663e-17,7.2859273622565e-14,-1.57090670168903e-10,3.0491292885473e-7,-0.000581078444935817,1.10516564125989,0.000148534476733103],
                    [-8.20209311569985e-19,4.34779284638878e-15,-1.76489909946068e-11,6.90754305025356e-8,-0.000269497671189628,1.05127100148724,2.69139400711076e-6],
                    [-2.79644218719739e-20,2.60080775407912e-16,-2.07823660780399e-12,1.6426765170331e-8,-0.000129779855270288,1.02531511893295,4.53070556313535e-8],
                    [-9.12880385587037e-22,1.57930196254673e-17,-2.51879120712609e-13,4.00518610053792e-9,-6.3683344531535e-5,1.01257845151487,7.34983595234415e-10],
                    [-2.91577138323977e-23,9.70954867859659e-19,-3.09980538706333e-14,9.88849311608068e-10,-3.15443976782067e-5,1.00626957200335,1.16379709756459e-11]
                    ])

    #--------------------------------------------------------------------------
    # resulting table
    # result(:, 0) is N
    # result(:, 1) is err_RK4
    # result(:, 2) is err_TM6
    # result(:, 3) is time_RK4
    # result(:, 4) is time_TM6
    # result(:, 5) is time_ratio = time_RK4/time_TM6
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
                y3 = y2*y
                y_map[i+1] = (y3*(M[0]*y3 + M[3]) +
                            y2*(M[1]*y3 + M[4]) +
                            y *(M[2]*y3 + M[5]) +
                            M[6])

            elapsed_time = time.time()-tic
            result[k, 4] += elapsed_time # time_RK4
            #------------------------------------------------------------------
            result[k, 1] += np.abs(y_rk4 - y_sol).max() # err_RK4
            result[k, 2] += np.abs(y_map - y_sol).max() # err_TM6

    result[:, 1:] /= len(y0_list) # get average results
    result[:, 5] = result[:, 3]/result[:, 4] # get time_ratio


    result = pd.DataFrame(data=result[:,1:], index=result[:,0], columns=np.array(['err_RK4', 'err_TM6', 'time_RK4', 'time_TM6', 'time_ratio']))
    return result


if __name__ == "__main__":
    print(main())