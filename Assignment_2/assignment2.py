import simpy
import numpy as np
import matplotlib.pyplot as plt
import plot


class Client():

    def __init__(self, id, env, bcs, prior, time_of_arr, time_in_service):
        self.id  = id
        self.env = env
        self.bcs = bcs
        self.time_of_arr = time_of_arr
        self.time_in_service = time_in_service
        self.key = time_in_service
        self.prior = prior

        self.waiting_time = 0


    def service(self):

        yield self.env.timeout(self.time_of_arr)
        # print(self.id, 'started waiting at', self.env.now)

        # wait until place is ready, then get service
        if not self.prior:
            with self.bcs.request() as req:
                yield req

                self.waiting_time = self.env.now - self.time_of_arr

                # print(self.id, 'get serviced at', self.env.now)
                yield self.env.timeout(self.time_in_service)

                # print(self.id, 'left at', self.env.now)
        else:
            with self.bcs.request(priority=self.prior) as req:
                yield req

                self.waiting_time = self.env.now - self.time_of_arr

                # print(self.id, 'service time', self.time_in_service)
                # print(self.id, 'get serviced at', self.env.now)
                yield self.env.timeout(self.time_in_service)

                # print(self.id, 'left at', self.env.now)

def experiment1(n_clients, n_servers, max_time, arr_time, ser_time, sorted):

    clients = []
    time_of_arr = 0
    for client in range(n_clients):
        time_of_arr = time_of_arr + np.random.exponential(scale=1/arr_time)
        time_in_ser = np.random.exponential(scale=1/ser_time)
        clients += [[time_of_arr, time_in_ser]]

    env = simpy.Environment()
    bcs = simpy.Resource(env, capacity=n_servers)
    if sorted:
        bcs = simpy.PriorityResource(env)
        # prior = simpy.resources.resource.PriorityRequest(bcs)
    
    for i,client in enumerate(clients):
        c = Client(i+1, env, bcs, client[1], client[0], client[1])
        env.process(c.service())
        clients[i] += [c]

    env.run(until=max_time)
    waiting_times = [clients[i][2].waiting_time for i in range(n_clients)]

    return waiting_times


def print_to_csv(results, n_experiments, type, prior):
    """
    Prints results to csv
    """
    
    for i,d in enumerate(results):
        string = str(i)+','
        for j,c in enumerate(results[i]):
            string += str(c)+','

        f = open('res_n_servers_'+str(n_experiments)+'_'+str(type)+'_'+prior+'.csv', "a")
        f.write(string + '\n')
        f.close()



if __name__ == '__main__':

    n_clients       = 100
    n_experiments   = 10000
    max_time        = 500
    arr_time        = 1
    ser_time        = 0.7
    sorted = False

    # rate of arival lambda
    # lamb = 10
    # # capacity of each server mu
    # mu   = 1
    # # system load rho
    # rho  = lamb/(mu*n_servers)

    all_n_experiments = np.array([int(i) for i in np.logspace(1,4,10)])

    # for n_experiments in all_n_experiments:
    #     print(n_experiments)
    #     # if n_experiments != 10000:

    #     results = {i:[] for i in [1,2,4]}
    #     for n_servers in [1,2,4]:
    #         all_waiting_times = []
    #         for i in range(n_experiments):
    #             all_waiting_times += [experiment1(n_clients, n_servers, max_time, arr_time*n_servers, ser_time, sorted)]
    #         results[n_servers] = [all_waiting_times, [np.average(i) for i in all_waiting_times]]

    #     print(results)

    #     print_to_csv(results, n_experiments, 1, 'non_prior')
    #     print_to_csv(results, n_experiments, 0, 'non_prior')

        # plot.plot_nservers()

    sorted = True
    n_servers = 1
    experiment1(n_clients, n_servers, max_time, arr_time, ser_time, sorted)

    for n_experiments in all_n_experiments:
        print(n_experiments)

        results = []
        # for n_servers in [1,2,4]:
        all_waiting_times = []
        for i in range(n_experiments):
            all_waiting_times += [experiment1(n_clients, n_servers, max_time, arr_time*n_servers, ser_time, sorted)]
        results += [all_waiting_times, [np.average(i) for i in all_waiting_times]]

        print(results)

        print_to_csv(results, n_experiments, 1, 'prior')
        print_to_csv(results, n_experiments, 0, 'prior')

