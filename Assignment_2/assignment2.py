import simpy
import numpy as np
import matplotlib.pyplot as plt
import plot
import scipy.stats as stats


class Client():

    def __init__(self, id, env, bcs, prior, time_of_arr, time_in_service):
        self.id  = id
        self.env = env
        self.bcs = bcs
        self.time_of_arr = time_of_arr
        self.time_in_service = time_in_service
        self.key = time_in_service
        self.prior = prior

        self.waiting_time = None


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
            with self.bcs.request(priority=self.time_in_service) as req:
                yield req

                self.waiting_time = self.env.now - self.time_of_arr

                # print(self.id, 'service time', self.time_in_service)
                # print(self.id, 'get serviced at', self.env.now)
                yield self.env.timeout(self.time_in_service)

                # print(self.id, 'left at', self.env.now)

def run_queue(n_clients, n_servers, max_time, lamb, mu, priority, ser_exp):
    """
    Runs one queue simulation
    """

    clients = []
    time_of_arr = 0
    for client in range(n_clients):
        # sample arrival time and service time
        time_of_arr = time_of_arr + np.random.exponential(scale=1/lamb)
        time_in_ser = 1/mu
        if ser_exp:
            time_in_ser = np.random.exponential(scale=1/mu)
        clients += [[time_of_arr, time_in_ser]]

    # initiate environment and resource(s) (=server(s))
    env = simpy.Environment()
    bcs = simpy.Resource(env, capacity=n_servers)

    # chet for priority, then change resource
    if priority:
        bcs = simpy.PriorityResource(env)
    
    # run experiment for all clients
    for i,client in enumerate(clients):
        c = Client(i+1, env, bcs, priority, client[0], client[1])
        env.process(c.service())
        clients[i] += [c]

    env.run(until=max_time)
    waiting_times = [clients[i][2].waiting_time for i in range(n_clients)]

    return waiting_times


def print_to_csv(results, filename):
    """
    Prints results to csv
    """
    
    for i in results:
        string = str(i)+','
        for j,c in enumerate(results[i]):
            string += str(c)+','

        f = open(filename, "a")
        f.write(string + '\n')
        f.close()


def rho_experiment(n_servers=1, max_time=1000, priority=False, ser_exp=True):
    """
    Estimates how the amount of measurements required depends on rho
    """
    
    rhos = np.linspace(0.1, 0.9, 5)
    mus  = [1/i for i in rhos]
    rho_results = {10:[], 100:[], 1000:[]}

    for i,rho in enumerate(rhos):
        expected = rho/((1-rho)*mus[i])
        print('exp', expected)
        lamb = 1
        mu = mus[i]

        for n_clients in [10, 100, 1000]:
            all_waiting_times = []
                
            # perform 1000 experiments for each amount of clients
            for exp in range(1000):
                all_waiting_times += [run_queue(n_clients, n_servers, max_time, lamb*n_servers, mu, priority, ser_exp)]

            for i,c in enumerate(all_waiting_times):
                all_waiting_times[i] = [x for x in c if x is not None]

            sign = stats.ttest_1samp([np.average(l) for l in all_waiting_times],expected)[1]

            rho_results[n_clients] += [sign]

    fig = plt.subplots(figsize=(7,4))
    for m in rho_results:
        plt.plot(rhos, rho_results[m], label='$m$='+str(m))
    plt.ylabel('significance level',fontsize=12)
    plt.xlabel(r'$\rho$',fontsize=12)
    plt.legend(fontsize=12)
    plt.xticks(list(rhos), [round(i,1) for i in list(rhos)])
    plt.tight_layout()
    plt.savefig('sign_frac_rho.pdf')


def rho_hist():
    """
    Plots two example histograms for rho=0.1 and rho=0.9
    """

    rhos = [0.5,0.9]
    n_clients = 1000
    n_servers = 1
    max_time  = 500
    priority  = False
    ser_exp   = True
    lamb = 1

    fig = plt.subplots(figsize=(7,4))

    for rho in rhos:
        mu   = 1/rho
        all_waiting_times = []

        col = 'red'
        bins = 10
        if rho == 0.9:
            col = 'blue'
            bins = 100

        # run experiment 1000 times
        for exp in range(1000):
                all_waiting_times += [run_queue(n_clients, n_servers, max_time, lamb*n_servers, mu, priority, ser_exp)]

        for i,c in enumerate(all_waiting_times):
                all_waiting_times[i] = [x for x in c if x is not None]

        # compute averages
        all_avg = [np.average(l) for l in all_waiting_times]

        # plot histograms
        plt.hist(all_avg, label=r'$\rho=$'+str(rho), bins=bins, color=col, alpha=0.3)
        expected = rho/((1-rho)*mu)
        plt.arrow(expected, 0, 0, 500, head_width=4, linestyle='dotted', color=col)

    plt.legend()
    plt.xlabel('average Waiting time')
    plt.ylabel('density')
    plt.tight_layout()
    # plt.show()
    plt.savefig('rho_hist_example.pdf')


def experiment1(n_clients, max_time, lamb, mu, all_n_exp, priority, ser_exp, ser_type):
    """
    Runs M/M/1 and M/M/n experiments
    """

    for n_exp in all_n_exp:
        results = {i:[] for i in [1,2,4]}
        # run experiment for different amounts of servers
        for n_servers in [1,2,4]:
            all_waiting_times = []

            # run experiment for n_exp times, save waiting times
            for i in range(n_exp):
                all_waiting_times += [run_queue(n_clients, n_servers, max_time, lamb*n_servers, mu, priority, ser_exp)]

            # filter out None values
            for i,c in enumerate(all_waiting_times):
                all_waiting_times[i] = [x for x in c if x is not None]

            # compute averages
            results[n_servers] = [np.average(i) for i in all_waiting_times]

        # print results to csv
        file_name = 'Data/'+str(n_clients)+'_cl_'+str(n_exp)+'_exper_'+ser_type+'_ser.csv'
        print_to_csv(results, file_name)


def experiment2(n_clients, n_servers, max_time, all_n_exp, lamb, mu, priority, ser_exp, ser_type):
    """
    Runs M/M/1 experiments with priority for short service time
    """
    for n_exp in all_n_exp:

        results = {1:[]}
        all_waiting_times = []

        # run experiment for n_exp times, save waiting times
        for i in range(n_exp):
            all_waiting_times += [run_queue(n_clients, n_servers, max_time, lamb*n_servers, mu, priority, ser_exp)]

        # filter out None values
        for i,c in enumerate(all_waiting_times):
                all_waiting_times[i] = [x for x in c if x is not None]

        # compute averages
        results[1] = [np.average(i) for i in all_waiting_times]

        # print results to csv
        file_name = 'Data/'+str(n_clients)+'_cl_'+str(n_exp)+'_exper_'+ser_type+'_ser_prior.csv'
        print_to_csv(results, file_name)


if __name__ == '__main__':

    n_clients       = 1000
    n_exp           = 1000
    max_time        = 500

    lamb            = 1
    mu              = 4/3

    # booleans for priority selection and exponential drawing of service time.
    priority  = False
    ser_exp   = True

    all_n_exp = np.array([int(i) for i in np.logspace(1,3,8)])

    # perform experiment for different values of rho:
    # rho_experiment()
    # rho_hist()

    # M/M/n experiment for n = 1, 2 and 4.
    # experiment1(n_clients, max_time, lamb, mu, all_n_exp, priority, ser_exp, 'exp')

    # M/M/1 experiment with sorting
    priority  = True
    n_servers = 1
    # experiment2(n_clients, n_servers, max_time, all_n_exp, lamb, mu, priority, ser_exp, 'exp')

    # experiment for M/D/1 (deterministic service time)
    priority = False
    ser_exp  = False
    experiment1(n_clients, max_time, lamb, mu, all_n_exp, priority, ser_exp, 'det')