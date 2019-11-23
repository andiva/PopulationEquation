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

    # maps of 8th order for different time steps (A=2, B=0.01)
    # M(0,:) is map for dt=0.05     (N=100)
    # M(1,:) is map for dt=0.025    (N=200)
    # M(2,:) is map for dt=0.0125   (N=400)
    # M(3,:) is map for dt=0.00625  (N=800)
    # M(4,:) is map for dt=0.003125 (N=1600)
    Maps = np.array([[-4.99601921667887e-24,1.84935964133497e-20,-4.23629183829682e-17,8.39174629410321e-14,-1.60590216195728e-10,3.05590456029527e-7,-0.000581157957616179,1.10517085683809,1.34511594564678e-6],
                    [-4.87731111635685e-26,2.78047377145932e-22,-1.15484975507109e-18,4.5376519355888e-15,-1.77104564948567e-11,6.90875050990476e-8,-0.000269499102660927,1.05127109608626,6.4017323733666e-9],
                    [-4.26178099855912e-28,4.13410513256993e-24,-3.32748495857332e-20,2.631715681418e-16,-2.07925096139956e-12,1.64269661317723e-8,-0.000129779879232919,1.02531512052318,2.76214379044979e-11],
                    [-3.52165980224463e-30,6.23344675198481e-26,-9.96204381136311e-22,1.58422251710012e-17,-2.51895391085821e-13,4.00518933883553e-9,-6.36833449188686e-5,1.01257845154063,2.10469969587637e-13],
                    [-2.82967746195301e-32,9.53610182697698e-28,-3.04610757006006e-23,9.71730546503066e-19,-3.09983113737682e-14,9.88849362981926e-10,-3.15443976843614e-5,1.00626957200376,-6.11683690894885e-14],
                    ])

    #--------------------------------------------------------------------------
    # resulting table
    # result(:, 0) is N
    # result(:, 1) is err_RK4
    # result(:, 2) is err_TM8
    # result(:, 3) is time_RK4
    # result(:, 4) is time_TM8
    # result(:, 5) is time_ratio = time_RK4/time_TM8
    result = np.zeros((len(timesteps), 6))


    for k, dt in enumerate(timesteps):  # for each time stemp dt
        M = Maps[k, :]                 # get TM8 for this dt
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
                #y_map[i+1] = M[0]*y**8+M[1]*y**7+M[2]*y**6+M[3]*y**5+M[4]*y**4+M[5]*y**3+M[6]*y**2+M[7]*y+M[8]
                #y_map[i+1] = +M[8]
                y2 = y*y
                y3 = y2*y
                y4 = y3*y
                y_map[i+1] = (y4*(M[0]*y4 + M[4]) +
                            y3*(M[1]*y4 + M[5]) +
                            y2*(M[2]*y4 + M[6]) +
                            y *(M[3]*y4 + M[7]) +
                            M[8]
                            )

            elapsed_time = time.time()-tic
            result[k, 4] += elapsed_time # time_RK4
            #------------------------------------------------------------------
            result[k, 1] += np.abs(y_rk4 - y_sol).max() # err_RK4
            result[k, 2] += np.abs(y_map - y_sol).max() # err_TM8

    result[:, 1:] /= len(y0_list) # get average results
    result[:, 5] = result[:, 3]/result[:, 4] # get time_ratio


    result = pd.DataFrame(data=result[:,1:], index=result[:,0], columns=np.array(['err_RK4', 'err_TM8', 'time_RK4', 'time_TM8', 'time_ratio']))
    return result


if __name__ == "__main__":
    print(main())
