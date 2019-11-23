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

    # RK4 maps of 16th order for different time steps (A=2, B=0.01)
    # M(0,:) is map for dt=0.05     (N=100)
    # M(1,:) is map for dt=0.025    (N=200)
    # M(2,:) is map for dt=0.0125   (N=400)
    # M(3,:) is map for dt=0.00625  (N=800)
    # M(4,:) is map for dt=0.003125 (N=1600)
    Maps = np.array([[-1.24176343282064e-54, 4.17232513427735e-50, -6.17305437723796e-46, 5.58670361836752e-42, -3.65673502286275e-38, 1.86983289718628e-34, -7.61456981341044e-31, 2.55201158536275e-27, -7.29182852147738e-24, 1.79585005859884e-20, -4.05253082868449e-17, 8.30971451424154e-14, -1.60413792806803e-10, 3.05568265755208e-7, -0.000581156515625000, 1.10517083333333, 0],
                    [-3.78956125738720e-59, 2.48595218484600e-54, -7.15893596255531e-50, 1.26161952114974e-45, -1.61626321641961e-41, 1.62417739144682e-37, -1.30012025170921e-33, 8.59902605996855e-30, -4.87811601234756e-26, 2.39285907946033e-22, -1.08922760954950e-18, 4.50707976967116e-15, -1.77032732017839e-11, 6.90865114748637e-8, -0.000269499027547200, 1.05127109375000, 0],
                    [-1.15648231731787e-63, 1.49880108324396e-58, -8.51300511565493e-54, 2.95968212900751e-49, -7.50094577191737e-45, 1.49426335029901e-40, -2.37104463179320e-36, 3.11554262291481e-32, -3.52255796638899e-28, 3.44882666909471e-24, -3.15548898298620e-20, 2.62289183471847e-16, -2.07903166284304e-12, 1.64269345936780e-8, -0.000129779876772881, 1.02531512044271, 0],
                    [-3.52930394689292e-68, 9.09148696719616e-63, -1.02551408301341e-57, 7.08145000068103e-53, -3.56976641206297e-48, 1.41596658769409e-43, -4.47335003020908e-39, 1.17167674089703e-34, -2.64503266559424e-30, 5.17411803479936e-26, -9.49353677268027e-22, 1.58164405586617e-17, -2.51888764652252e-13, 4.00518836603873e-9, -6.36833448419993e-5, 1.01257845153809, 0],
                    [-1.07705808926176e-72, 5.53177034644837e-67, -1.24354025058865e-61, 1.71141357456727e-56, -1.72071128136625e-51, 1.36202572421837e-46, -8.58630390761042e-42, 4.49042483608656e-37, -2.02572795133709e-32, 7.92132749991183e-28, -2.91088534993033e-23, 9.70955007019360e-19, -3.09981084252993e-14, 9.88849332868609e-10, -3.15443976819671e-5, 1.00626957200368, 0]
                    ])

    #--------------------------------------------------------------------------
    # resulting table
    # result(:, 0) is N
    # result(:, 1) is err_RK4
    # result(:, 2) is err_TM16_RK4
    # result(:, 3) is time_RK4
    # result(:, 4) is time_TM16_RK4
    # result(:, 5) is time_ratio = time_RK4/time_TM16_RK4
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
                y_map[i+1] = (M[0]*y**16+M[1]*y**15+M[2]*y**14+M[3]*y**13+M[4]*y**12+M[5]*y**11+M[6]*y**10+M[7]*y**9+
                            M[8]*y**8+M[9]*y**7+M[10]*y**6+M[11]*y**5+M[12]*y**4+M[13]*y**3+M[14]*y**2+M[15]*y+M[16])


            elapsed_time = time.time()-tic
            result[k, 4] += elapsed_time # time_RK4
            #------------------------------------------------------------------
            result[k, 1] += np.abs(y_rk4 - y_sol).max() # err_RK4
            result[k, 2] += np.abs(y_map - y_sol).max() # err_TM8

    result[:, 1:] /= len(y0_list) # get average results
    result[:, 5] = result[:, 3]/result[:, 4] # get time_ratio


    result = pd.DataFrame(data=result[:,1:], index=result[:,0], columns=np.array(['err_RK4', 'err_TM16_RK4', 'time_RK4', 'time_TM16_RK4', 'time_ratio']))
    return result


if __name__ == "__main__":
    print(main())