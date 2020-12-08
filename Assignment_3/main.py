import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import csv



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

    # plt.scatter(t,x)
    # plt.scatter(t,y)
    # plt.xlabel('t')
    # plt.ylabel('x and y')
    # plt.show()

    return t,x,y


def integrate(param,t,x,y):
    a = param[0]
    b = param[1]
    g = param[2]
    d = param[3]
    time = np.arange(0, len(t))

    def eq(begin_pop,t,a,b,g,d):
        x,y = begin_pop
        dxdt = a*x - b*x*y
        dydt = d*x*y - g*y
        return dxdt, dydt

    begin_pop = x[0],y[0]
    numint = odeint(eq, begin_pop, time, args=(a,b,g,d))

    x_val, y_val = numint.T

    return x_val, y_val

    # plt.scatter(t, x_val)
    # plt.scatter(t, x)
    # for i,tval in enumerate(t):
    #     plt.arrow(tval, min(x[i], x_val[i]), 0, max(x[i],x_val[i]))
    # # plt.scatter(t, y_val[1:])

    # plt.show()

    # print(len(x))
    # print(len(x_val))


def means_sq(x,y,x_val,y_val):
    """
    Computes means squared error
    """
    error_x = np.average([(np.array(x)-np.array(x_val))**2])
    error_y = np.average([(np.array(y)-np.array(y_val))**2])
    return error_x, error_y


def hillclimber(t,x,y):

    # param = [np.abs(np.random.normal(0,0.5)) for i in range(4)]
    param = [0.3,0.3,0,3,0.3]
    # print(param)
    x_est, y_est = integrate(param,t,x,y)
    error_x, error_y = means_sq(x,y,x_est,y_est)

    all_error_x = []

    steps = 4000

    for i in range(steps):
        change = np.random.choice(param)
        new_param = param
        new_param[param.index(change)] = np.abs(change + np.random.normal(0,0.1))
        
        x_est, y_est = integrate(param,t,x,y)
        new_error_x, new_error_y = means_sq(x,y,x_est,y_est)

        # print(new_error_x)

        if new_error_x < error_x:
            param = new_param
            error_x = new_error_x

        all_error_x += [error_x]

    plt.plot(range(steps), all_error_x)
    plt.show()

    fig , ax = plt.subplots(1,2)
    ax[0].scatter(t, x_est, label='est')
    ax[0].scatter(t, x, label='real')
    ax[0].legend()
    # for i,tval in enumerate(t):
    #     plt.arrow(tval, min(x[i], x_val[i]), 0, max(x[i],x_val[i]))
    ax[1].scatter(t, y_est, label='est')
    ax[1].scatter(t,y, label='real')
    ax[1].legend()

    plt.show()


if __name__ == "__main__":

    error_type = 'means_sq'

    t,x,y, = open_data()
    # param=[0.3,0.3,0.3,0.3]
    # x_val, y_val = integrate(param,t,x,y)

    # if error_type == 'means_sq':
    #     error_x, error_y = means_sq(x,y,x_val,y_val)
    #     print(error_x, error_y)

    hillclimber(t,x,y)