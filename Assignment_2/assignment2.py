import simpy
import numpy as np
import matplotlib.pyplot as plt
import plot


class Client():

    def __init__(self, id, env, bcs, time_of_arr, time_in_service):
        self.id  = id
        self.env = env
        self.bcs = bcs
        self.time_of_arr = time_of_arr
        self.time_in_service = time_in_service

        self.waiting_time = 0


    def service(self):

        yield self.env.timeout(self.time_of_arr)
        # print(self.id, 'started waiting at', self.env.now)

        # wait until place is ready, then get service
        with self.bcs.request() as req:
            yield req

            self.waiting_time = self.env.now - self.time_of_arr

            # print(self.id, 'get serviced at', self.env.now)
            yield self.env.timeout(self.time_in_service)

            # print(self.id, 'left at', self.env.now)

def experiment(n_clients, n_servers, max_time, arr_time, ser_time):

    clients = []
    time_of_arr = 0
    for client in range(n_clients):
        time_of_arr = time_of_arr + np.random.exponential(scale=1/arr_time)
        time_in_ser = np.random.exponential(scale=1/ser_time)
        clients += [[time_of_arr, time_in_ser]]

    env = simpy.Environment()
    bcs = simpy.Resource(env, capacity=n_servers)

    for i,client in enumerate(clients):
        c = Client(i+1, env, bcs, client[0], client[1])
        env.process(c.service())
        clients[i] += [c]

    env.run(until=max_time)
    waiting_times = [clients[i][2].waiting_time for i in range(n_clients)]

    return waiting_times


def print_to_csv(results):
    """
    Prints results to csv
    """
    
    for i in results:
        string = str(i)+','
        for i,c in enumerate(results[i][1]):
            string += str(c)+','

        f = open('res_n_servers.csv', "a")
        f.write(string + '\n')
        f.close()

def plot_and_test():
    """
    Plots results for different n servers and performs statistical test
    """
    # open results

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[10,4])
    # ax1.hist(results[1][1], label='1 server', alpha=0.5, bins=25)
    # ax2.hist(results[2][1], label='2 servers', alpha=0.5, bins=25)
    # ax3.hist(results[4][1], label='4 servers', alpha=0.5, bins=25)
    # # plt.legend()
    
    # ax1.set_ylabel('number of experiments')
    # ax1.set_xlabel('average waiting time')
    # ax1.set_title('1 server')
    # ax2.set_xlabel('average waiting time')
    # ax2.set_title('2 servers')
    # ax3.set_xlabel('average waiting time')
    # ax3.set_title('4 servers')

    # plt.tight_layout()
    # plt.savefig('hist_servers_apart.pdf')




if __name__ == '__main__':

    n_clients       = 100
    n_experiments   = 10000
    max_time        = 500
    arr_time        = 1
    ser_time        = 0.7

    # rate of arival lambda
    # lamb = 10
    # # capacity of each server mu
    # mu   = 1
    # # system load rho
    # rho  = lamb/(mu*n_servers)

    results = {i:[] for i in [1,2,4]}
    for n_servers in [1,2,4]:
        all_waiting_times = []
        for i in range(n_experiments):
            all_waiting_times += [experiment(n_clients, n_servers, max_time, arr_time*n_servers, ser_time)]
        results[n_servers] = [all_waiting_times, [np.average(i) for i in all_waiting_times]]

    print(results)

    print_to_csv(results)

    plot.plot_nservers()

