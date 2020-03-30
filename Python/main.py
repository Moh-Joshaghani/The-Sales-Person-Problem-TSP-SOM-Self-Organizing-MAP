"""
This module initializes and solves a TSP problem, and the steps are illustrated in a graphical way


"""
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')
import numpy as np


#               FUNCTIONS
def gaussian(sgm, j, jstar, clus_num):
    """
    gaussian function
    :param sgm:
    :param j:
    :param jstar:
    :param clus_num:
    :return:
    """
    dist = min(abs(j - jstar), clus_num - abs(j-jstar))
    return np.exp(-(dist / sgm) ** 2)




#               CLASSES
class TSP:
    def __init__(self,
                 city_count,
                 factor=2.4,
                 eta=0.2,
                 sigma_decay=0.015,
                 max_iter=100,
                 min_change_to_terminate=1e-5
                 ):

        # the problem settings
        self.city_count = city_count
        self.cluster_count = int(factor*city_count)

        # initializing cities and normalizing them
        self.cities = np.random.rand(self.city_count, 2)
        self.cities[0, :] = (self.cities[0, :] - np.min(self.cities[0, :]))/(np.max(self.cities[0, :]) - np.min(self.cities[0, :]))
        self.cities[1, :] = (self.cities[1, :] - np.min(self.cities[1, :]))/(np.max(self.cities[1, :]) - np.min(self.cities[1, :]))

        # initializing the clusters on a ring
        t = np.linspace(0, 2*np.pi, self.cluster_count)
        self.clusters = np.concatenate((0.5+0.5*np.cos(t)[:, np.newaxis], 0.5+0.5*np.sin(t)[:, np.newaxis]), axis=1)

        # for graphical output
        self.cluster_history = [np.copy(self.clusters)]

        # initializing the plots
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.state = 'initial'
        self._update_plot()

        # simulation parameters
        self.eta = eta
        self.sigma_decay = sigma_decay
        self.max_iter = max_iter
        self.sigma = self.cluster_count
        self.epoch = 0
        self.distance_to_all = np.zeros((self.cluster_count, 1))
        self.min_change_to_terminate = min_change_to_terminate
        self.delta_weights = 100



    # ===============================================================================================
    # ===============================================================================================
    def _update_plot(self, cluster_idx=-1, delay=0.01, clear_figure=True):
        """
        updates the plot of cities and clusters in the current state
        :return:
        """

        # clearing the plot
        if clear_figure:
            plt.cla()

        # plotting the initial state
        self.ax.scatter(self.cities[:, 0], self.cities[:, 1], c='r', marker='*', label='Cities')

        # making ring location
        clusters = np.vstack((self.cluster_history[cluster_idx], self.cluster_history[cluster_idx][0]))
        self.ax.plot(clusters[:, 0], clusters[:, 1], c='b', marker='o', linestyle='--', fillstyle='none',
                     label="Traveler's Path")

        self.ax.set_title('state: {}'.format(self.state))
        self.ax.legend()
        plt.show(False)
        plt.draw()
        plt.pause(delay)

    # ===============================================================================================
    # ===============================================================================================
    def single_iteration(self):
        """

        :return:
        """
        # updating the state
        self.state = 'iteration = {}'.format(self.epoch)
        self.previous_clusters = np.copy(self.clusters)

        # decaying sigma
        self.sigma *= np.exp(-self.epoch * self.sigma_decay)
        self.eta = 1 / (np.sqrt(np.sqrt(self.epoch)))

        # iterating over cities
        for city in self.cities:

            # finding the closest cluster
            self.distance_to_all = np.sum((self.clusters - city)**2, axis=1)
            id_closest = np.argmin(self.distance_to_all)

            # updating the cluster centers
            g = self.eta * np.asarray([gaussian(self.sigma, id_clus, id_closest, self.cluster_count)
                                       for id_clus in range(self.cluster_count)])[:, np.newaxis]

            dist_idx = city - self.clusters

            self.clusters += np.repeat(g, 2, 1) * dist_idx

        # saving the new cluster in history
        self.cluster_history.append(np.copy(self.clusters))
        self.delta_weights = np.mean(np.sqrt(np.sum((self.clusters - self.cluster_history[-2])**2, 1)))

    # ===============================================================================================
    # ===============================================================================================
    def TSP_loop(self):
        """
        the main loop in which the problem is being solved in some iterations
        :return:
        """

        while self.epoch < self.max_iter and self.delta_weights > self.min_change_to_terminate:
            self.epoch += 1
            # performing one update on the parameters
            self.single_iteration()
            # visualizing the effect
            self._update_plot(delay=0.01)

        # saving the animation as a gif file
        self.save_animation()

    # =================================================================================================
    # =================================================================================================
    def save_animation(self):

        plt.close(fig=self.fig)
        self.fig = plt.figure()
        self.ax = plt.axes()

        animation = FuncAnimation(self.fig, self.animate, init_func=self.init_animation,
                                  frames=len(self.cluster_history),
                                  repeat_delay=2,
                                  interval=400)
        animation.save('./animation.gif', writer='imagemagick', fps=6)

    def init_animation(self):
        self.ax.scatter(self.cities[:, 0], self.cities[:, 1], c='r', marker='*', label='Cities')


    def animate(self, i):
        plt.cla()
        # plotting the initial state
        self.ax.scatter(self.cities[:, 0], self.cities[:, 1], c='r', marker='*', label='Cities')

        # making ring location
        clusters = np.vstack((self.cluster_history[i], self.cluster_history[i][0]))
        self.ax.plot(clusters[:, 0], clusters[:, 1], c='b', marker='o', linestyle='--', fillstyle='none',
                     label="Traveler's Path")

        self.ax.set_title('Iteration: {}'.format(i))


        return self.ax

# _____________________________________________________________________________________________________
#                   TEST
tsp = TSP(10)
tsp.TSP_loop()

a = 1



