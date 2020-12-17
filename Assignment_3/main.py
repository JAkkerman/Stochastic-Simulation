import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random
import csv
import pandas as pd
import copy



def open_data():
    """
    Opens data from csv
    """
    with open('predator-prey-data.csv', newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        t = []
        y = []
        x = []

        for i,row in enumerate(reader):
            if i != 0:
                t += [np.double(row[0])]
                x += [np.double(row[1])]
                y += [np.double(row[2])]

    return t,x,y


# def save_to_csv(res):
#     """
#     Saves results to csv
#     """
#     print(res)
#     data = {}
#     lastsols = [list(exp[0][0]) for exp in res]
#     bestsols = [list(exp[1][0]) for exp in res]
#     errors   = [list(exp[2]) for exp in res]

#     data = {'lastsols': lastsols, 'bestsols': bestsols, 'errors': errors}

#     # for i,exp in enumerate(res):
#     #     data = {'last': last_sol[0], 'best': best_sol[0], 'errors': errors}
#     #     data[i] = []
    
#     df = pd.DataFrame(data)
#     df.to_csv(path_or_buf='SA_exp_fulldataset.csv')


def print_to_csv(results, filename):
    """
    Prints results to csv
    """
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(results)
    
    # for i in results:
    #     string = str(i)+','
    #     for j,c in enumerate(results[i]):
    #         string += str(c)+','

    #     f = open(filename, "a")
    #     f.write(string + '\n')
    #     f.close()


def integrate(param, t, x0, y0, x_keys, y_keys):
    """
    Integrates ODE based on simulated parameter values
    """
    a = param[0]
    b = param[1]
    g = param[2]
    d = param[3]
    time = np.linspace(0, 20, 100)

    def eq(begin_pop,t,a,b,g,d):
        x,y = begin_pop
        dxdt = a*x - b*x*y
        dydt = d*x*y - g*y
        return dxdt, dydt

    begin_pop = x0, y0
    numint = odeint(eq, begin_pop, time, args=(a,b,g,d))

    x_val, y_val = numint.T

    # if x_keys.any():
    # x_val = [x_val[int(i)] for i in x_keys]
    # if y_keys.any():

    # y_val = [y_val[int(i)] for i in y_keys]

    return x_val, y_val


def error(x_prime, y_prime, x_val, y_val, x_keys, y_keys, error_method='mean squared'):
    '''
    Computes the loss between data and predictions
    x, y : data
    x_val, y_val: predictions
    method: which loss function to use 'mean squared' or 'absolute'
    '''
    # if x_keys.any():
    #     x = [x[i] for i in x_keys]
    # if y_keys.any():
    #     y = [y[i] for i in y_keys]

    x_val = [x_val[int(i)] for i in x_keys]
    y_val = [y_val[int(i)] for i in y_keys]

    if error_method == 'mean squared':
        error_x = np.average([(np.array(x_prime)-np.array(x_val))**2])
        error_y = np.average([(np.array(y_prime)-np.array(y_val))**2])

    elif error_method == 'absolute':
        error_x = np.mean(np.abs([np.array(x_prime) - np.array(x_val)]))
        error_y = np.mean(np.abs([np.array(y_prime) - np.array(y_val)]))

    return error_x + error_y


def hillclimber(t,x,y, error_method='mean squared', plot_error=False, plot_fit=False, steps=1000, 
                n_runs=4):
    '''
    t, x, y: time, predator, prey data
    error: 'absolute'or 'mean squared', defines which loss function to use
    plot_error: boolean, whether or not to plot the evolution of the error
    plot_fit: boolean, whether or not to plot the final fit
    steps: int, how often the algorithm is executed
    n_runs: int, how often the algorithm starts again from a random point
    ''' 

    # dictionary with key, value = run, (errors, params)
    output = {}

    for run in range(n_runs):

        param = np.random.uniform(0, 2, size=4).tolist()
        x_est, y_est = integrate(param,t,x[0],y[0])

        error_xy = error(x,y,x_est,y_est, error_method=error_method)

        all_error_xy = []

        for i in range(steps):
            change = np.random.choice(param)
            new_param = param
            new_param[param.index(change)] = np.abs(change + np.random.normal(0,0.1))
            
            x_est, y_est = integrate(param,t,x[0],y[0])
            new_error_xy = error(x,y,x_est,y_est, error_method=error_method)


            if new_error_xy < error_xy:
                param = new_param
                error_xy = new_error_xy

        all_error_xy += [error_xy]
        output[run] = [error_xy, new_param]

    # select best fit
    min_error = min([val[0] for key, val in output.items()])
    best_param = [val[1] for key, val in output.items() if val[0] == min_error][0]

    if plot_error ==  True:
        plt.plot(range(steps), all_error_xy)
        plt.show()

    # select best fit
    min_error = min([val[0] for key, val in output.items()])
    best_param = [val[1] for key, val in output.items() if val[0] == min_error][0]
    x_est, y_est = integrate(best_param,t,x[0],y[0])

    if plot_fit == True:
        fig , ax = plt.subplots(1,2, figsize=(7, 4))
        ax[0].plot(t, x_est, label='est')
        ax[0].plot(t, x, label='real')
        ax[0].set_title('predator')
        ax[0].legend()
        # for i,tval in enumerate(t):
        #     plt.arrow(tval, min(x[i], x_val[i]), 0, max(x[i],x_val[i]))
        ax[1].plot(t, y_est, label='est')
        ax[1].plot(t,y, label='real')
        ax[1].set_title('prey')
        ax[1].legend()

        plt.show()

    return best_param


def sigmoid_linmap(step, steps):
    """
    Returns a linear map to the inverse sigmoid function
    """
    def S(x):
        return (1-np.exp(x)/(np.exp(x)+1))

    linmap = step*(8/steps)-4

    return S(linmap)

def calc_T0(n_samp, T, error_method='mean squared'):
    # add al neighbours to a set
    S = [np.random.uniform(0, 2, size=4)]
    for i in range(n_samp):
        noise = np.array([np.random.normal(0, 0.1), 0, 0, 0])
        np.random.shuffle(noise)
        S.append(np.abs(np.array(S[-1]) + noise))

    t, x, y = open_data()
    epsilon = 1e-3
    p = 1.5
    chi = []
    x_val, y_val = integrate(S[0], t, x, y)

    for i in range(1000):
        E_max = []
        E_min = []
        for j in range(len(S)-1):
            # before transition
            x_val, y_val = integrate(S[j], t, x, y)
            E_min += [error(x, y, x_val, y_val, error_method=error_method)]

            # after transition
            x_val, y_val = integrate(S[j+1], t, x, y)
            E_max += [error(x, y, x_val, y_val, error_method=error_method)]

        chi += [np.sum(np.exp(-np.array(E_max)/T)) / np.sum(np.exp(-np.array(E_min)/T))]

        if np.abs(chi[-1] - chi[0]) <= epsilon:
            return T

        else:
            T = T * (np.log(chi[-1]) / np.log(chi[0])) ** (1/p)


def SA(t, x, y, run, error_method='mean squared', cooling='linear', reducerand=False, x_keys=np.linspace(0,99,100), y_keys=np.linspace(0,99,100)):
    '''
    Simulated annealing algorithm
    t, x, y: time, predator, prey data
    p0: array_like, initial parameters
    error: 'absolute'or 'mean squared', defines which loss function to use
    '''

    all_errors = []

    # initializations
    steps = 10e3
    # steps = 10
    # steps = 100
    Tc = 1        # current temperature
    dT = Tc/steps # step size, scaled to amount of steps

    # dT = 10e-5   # step size

    Tf = 10e-5    # final temperature
    if cooling == 'logarithmic':
        Tf = 0.07
    # elif cooling == 'geometric':
    #     Tf = 0.01

    count = 1

    x0 = x[0]
    y0 = y[0]

    # param = list(np.random.uniform(0, 1, size=2)) + list(np.random.uniform(2, 4, size=2))
    param = np.random.uniform(0, 2, size=4)
    x_current, y_current = integrate(param, t, x0, y0, x_keys, y_keys)
    

    # if x_keys.any():
    x_prime = [x[int(i)] for i in x_keys]
    
    # if y_keys.any():
    y_prime = [y[int(i)] for i in y_keys]

    error_current = error(x_prime, y_prime, x_current, y_current, x_keys, y_keys, error_method=error_method)
    all_errors += [error_current]
    best_sol = [param, error_current]

    all_Tc = [Tc]

    while Tc > Tf:
        # choose new parameters
        noise = np.array([np.random.normal(0, 0.1), 0, 0, 0])
        np.random.shuffle(noise)
        # print(noise)
        param_neighbour = np.abs(np.array(param) + noise)
        x_neighbour, y_neighbour = integrate(param_neighbour, t, x0, y0, x_keys, y_keys)
        error_neighbour = error(x_prime, y_prime, x_neighbour, y_neighbour, x_keys, y_keys, error_method=error_method)

        diff = error_current - error_neighbour

        if error_neighbour < error_current:
            error_current = error_neighbour
            param = param_neighbour

        else:
            if np.random.uniform(0, 1) < np.exp(diff / Tc):
                error_current = error_neighbour
                param = param_neighbour

        all_errors += [error_current]

        if error_current < best_sol[1]:
            best_sol = [param, error_current]

        if cooling =='linear':
            Tc -= dT
        elif cooling == 'geometric':
            Tc *= 0.999
        elif cooling == 'sigmoid':
            Tc = sigmoid_linmap(count, steps)
        elif cooling == 'logarithmic':
            Tc = 0.7/np.log(1+count)

        all_Tc += [Tc]

        # print(Tc)
        count += 1

        if cooling == 'sigmoid' and count==steps:
            break

    print('final parameters: \na: ',param[0],'\nb: ',param[1],'\ng: ',param[2],'\nd: ',param[3])
    print('best parameters: \na: ',best_sol[0][0],'\nb: ',best_sol[0][1],'\ng: ',best_sol[0][2],'\nd: ',best_sol[0][3])
    print('final ratio a/b: ', round(param[0]/param[1],2), ',final ratio g/d:', round(param[2]/param[3],2))
    print('best ratio a/b: ', round(best_sol[0][0]/best_sol[0][1],2), ',best ratio g/d:', round(best_sol[0][2]/best_sol[0][3],2))

    # print(x)
    # x_current, y_current = integrate(param, t, x0, y0, x_keys, y_keys)
    x_bestsol, y_bestsol = integrate(best_sol[0], t, x0, y0, x_keys, y_keys)

    results = []
    results += list(param)
    results += list(best_sol[0])
    results += [all_errors[-1]]
    results += [best_sol[1]]

    filename = 'SA_'+cooling+'_'+error_method+'.csv'
    if reducerand:
        filename = 'SA_'+cooling+'_'+error_method+'_removed_data_'+reducerand+'.csv'

    print_to_csv(results, filename)

    # fig , ax = plt.subplots(1,2, figsize=(7, 4))
    # # ax[0].plot(t, x_current, label='last est')
    # ax[0].plot(t, x_bestsol, label='best est')
    # ax[0].plot(t, x, label='real')
    # ax[0].set_title('predator')
    # ax[0].legend()
    # # ax[1].plot(t, y_current, label='last est')
    # ax[1].plot(t, y_bestsol, label='best est')
    # ax[1].plot(t,y, label='real')
    # ax[1].set_title('prey')
    # ax[1].legend()

    # plt.show()

    # plt.plot(np.arange(0, len(all_errors), 1), all_errors)
    # plt.show()

    # plt.plot(range(len(all_Tc)), all_Tc)
    # plt.show()

    # return [param, error_current], best_sol, all_errors


def reduce_random(t,x,y):
    """
    Randomly reduces 
    """
    n_experiments = 20

    # reduce x
    # for perc in [0.8, 0.6, 0.4, 0.2, 0]:
    #     for i in range(n_experiments):
    #         x_keys = list(np.random.choice(range(len(x)),size=int(perc*len(x))))
    #         while not 0 in x_keys:
    #             if perc == 0:
    #                 x_keys += [0]
    #                 break
    #             del x_keys[-1]
    #             x_keys += [0]

    #         SA(t, x, y, i, error_method='mean squared', cooling='linear', reducerand='x', x_keys=x_keys)

    # reduce y
    for perc in [0.8, 0.6, 0.4, 0.2, 0]:
        for i in range(n_experiments):
            y_keys = list(np.random.choice(range(len(y)),size=int(perc*len(y))))
            while not 0 in y_keys:
                if perc == 0:
                    y_keys += [0]
                    break
                del y_keys[-1]
                y_keys += [0]

            SA(t, x, y, i, error_method='mean squared', cooling='linear', reducerand='y', y_keys=y_keys)


def remove_data(n_remove):

    for j in range(30):
        t, x, y = open_data()
        x_copy = copy.copy(x)
        y_copy = copy.copy(y)

        mid_x = np.mean(x)
        mid_y = np.mean(y)
        tbx_removed = []
        tby_removed = []

        for i in range(n_remove):
            close_x = min(range(len(x_copy)), key=lambda i: abs(x_copy[i]-mid_x))
            tbx_removed += [close_x]
            del x_copy[close_x]
        
            close_y = min(range(len(y_copy)), key=lambda i: abs(y_copy[i]-mid_y))
            tby_removed += [close_y]
            del y_copy[close_y]

        x_keys = np.sort([x.index(ele) for ele in x_copy])
        y_keys = np.sort([y.index(ele) for ele in y_copy])

        SA(t, x, y, 1, x_keys=x_keys, y_keys=y_keys, reducerand=f'removemean_{n_remove}')

        
        x_copy = copy.copy(x)
        y_copy = copy.copy(y)

        max_x = max(x)
        max_y = max(y)
        tbx_removed = []
        tby_removed = []

        for i in range(n_remove):
            close_x = min(range(len(x_copy)), key=lambda i: abs(x_copy[i]-max_x))
            tbx_removed += [close_x]
            del x_copy[close_x]
        
            close_y = min(range(len(y_copy)), key=lambda i: abs(y_copy[i]-max_y))
            tby_removed += [close_y]
            del y_copy[close_y]

        x_keys = np.sort([x.index(ele) for ele in x_copy])
        y_keys = np.sort([y.index(ele) for ele in y_copy])

        SA(t, x, y, 1, x_keys=x_keys, y_keys=y_keys, reducerand=f'removemax_{n_remove}')


        x_copy = copy.copy(x)
        y_copy = copy.copy(y)

        min_x = min(x)
        min_y = min(y)
        tbx_removed = []
        tby_removed = []

        for i in range(n_remove):
            close_x = min(range(len(x_copy)), key=lambda i: abs(x_copy[i]-min_x))
            tbx_removed += [close_x]
            del x_copy[close_x]
        
            close_y = min(range(len(y_copy)), key=lambda i: abs(y_copy[i]-min_y))
            tby_removed += [close_y]
            del y_copy[close_y]

        x_keys = np.sort([x.index(ele) for ele in x_copy])
        y_keys = np.sort([y.index(ele) for ele in y_copy])

        SA(t, x, y, 1, x_keys=x_keys, y_keys=y_keys, reducerand=f'removemin_{n_remove}')



if __name__ == "__main__":

    error_type = 'means_sq'

    t,x,y = open_data()

    n_experiments = 15
    # param=[0.3,0.3,0.3,0.3]
    # x_val, y_val = integrate(param,t,x,y)

    # if error_type == 'means_sq':
    #     error_x, error_y = means_sq(x,y,x_val,y_val)
    #     print(error_x, error_y)

    # params = hillclimber(t,x,y, plot_fit=True, n_runs=4, steps=2000)
    # res = []
    # for i in range(n_experiments):
        # last_sol, best_sol, errors = SA(t, x, y, i, error_method='absolute', cooling='linear')
        # res += [[last_sol, best_sol, errors]]
    # save_to_csv(res)

    # reduce_random(t,x,y)
    for n in [10, 20]:
        remove_data(n)