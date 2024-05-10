"""
This module initializes and solves a TSP problem, and the steps are illustrated in a graphical way


"""
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import imageio
import os
plt.style.use('seaborn-pastel')
import numpy as np
import math

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
    dist = min(abs(j - jstar), clus_num - abs(j - jstar))
    return np.exp(-(dist / sgm) ** 2)

# ================================================================
def plot_cities_on_map(coordinates, clusters=None, fig=None, ax=None, show_plot=False):
    # Create a new map plot
    if fig is None:
        # fig = plt.figure(figsize=(10, 7))
        fig, ax = plt.subplots(figsize=(10, 7))

    min_lat = min(coord[0] for coord in coordinates) - 5  # Adding a small buffer to ensure visibility on the map
    max_lat = max(coord[0] for coord in coordinates) + 5
    min_lon = min(coord[1] for coord in coordinates) - 5
    max_lon = max(coord[1] for coord in coordinates) + 5

    m = Basemap(projection='mill', # 'mill',  # Miller cylindrical projection
                llcrnrlat=min_lat,  # Lower latitude limit
                urcrnrlat=max_lat,  # Upper latitude limit
                llcrnrlon=min_lon,  # Left longitude limit
                urcrnrlon=max_lon,  # Right longitude limit
                resolution='c')

    m.drawcoastlines(linewidth=0.5, color='#cccccc')  # Draw coastlines
    m.drawcountries(linewidth=0.5, color='#cccccc')  # Draw countries
    # m.fillcontinents(color='palegreen', lake_color='lightblue')
    m.fillcontinents(color='#f4e8c1', lake_color='#a9c0cb')
    m.drawmapboundary(fill_color='#a9c0cb')
    # m.shadedrelief()  # m.shadedrelief() m.bluemarble()
    # Plot each city on the ma p
    for coord in coordinates:
        x, y = m(coord[1], coord[0])  # Convert longitude and latitude to the map projection coordinates
        ax.plot(x, y, markersize=3,  c='#808000', marker='*')

    # plotting the commute if set
    if clusters is not None:
        previous_coords = []
        for idx, coord in enumerate(clusters):
            x, y = m(coord[1], coord[0])  # Convert longitude and latitude to the map projection coordinates
            if idx == 0:
                previous_coords = [x, y]
                continue
            ax.plot([previous_coords[0], x], [previous_coords[1], y], c='#A0522D', linewidth=1, marker='o',
                    linestyle='--', fillstyle='none')
            previous_coords = [x, y]

    if show_plot:
        ax.show()


# ================================================================
def haversine_distance(coords1, coords2):
    """
    Calculate haversince distance between two longitude and latitude points.
    :param coords1:
    :param coords2:
    :return:
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, np.hstack((coords1, coords2)))

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of Earth in kilometers.
    r = 6371.0

    return c * r


#               CLASSES
class animate_cities:
    def __init__(self, cities, show_world=True):
        self.cities_coordinates = cities
        self.show_world = show_world
        min_lat = min(coord[0] for coord in self.cities_coordinates) - 5
        max_lat = max(coord[0] for coord in self.cities_coordinates) + 5
        min_lon = min(coord[1] for coord in self.cities_coordinates) - 5
        max_lon = max(coord[1] for coord in self.cities_coordinates) + 5

        self.basemap = Basemap(projection='mill', llcrnrlat=min_lat, urcrnrlat=max_lat, llcrnrlon=min_lon,
                               urcrnrlon=max_lon, resolution='c')
        self.iteration = 0
        self.figure_frames = []
        self.fig = plt.figure()
        self.ax = plt.axes()

        self._init_animation(self.cities_coordinates)

    # ===================================================================================
    def _init_animation(self, cities_coordinates):
        plot_cities_on_map(cities_coordinates, fig=self.fig, ax=self.ax)

    # ===================================================================================
    def plot_single_iteration(self, cluster_history, cluster_idx=-1, delay=0.01, clear_figure=True, save=False):
        # clearing the plot
        if clear_figure:
            self.ax.cla()

        self.iteration += 1

        # making ring location
        clusters = np.vstack((cluster_history[cluster_idx], cluster_history[cluster_idx][0]))
        plot_cities_on_map(self.cities_coordinates, clusters, fig=self.fig, ax=self.ax)
        self.ax.set_title(f'Iteration: {self.iteration}')
        plt.show(block=False)
        plt.draw()
        plt.pause(delay)



        # returning the figure to make an animation at the end of all the loops
        self.fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.figure_frames.append(image)
        if save:
            plt.savefig(f'./output/Iteration_{self.iteration}_output.png')

    # =================================================================================================
    def save_animation(self, save_dir='output/animation.gif'):
        fig_frames = np.array(self.figure_frames)
        dur = fig_frames.shape[0] / 5
        print(f'fig_frames shape: {fig_frames.shape}')
        print(f'saving the animation in: {save_dir}')
        imageio.mimwrite(save_dir, fig_frames, format='GIF', duration=dur)


# ==================================================================
class TSP:
    def __init__(self,
                 city_count=None,
                 cities_coordinates_list=None,
                 factor=2.4,
                 eta=0.2,
                 sigma_decay=0.015,
                 max_iter=100,
                 min_change_to_terminate=1e-5
                 ):
        self.show_world = True
        if city_count is None and cities_coordinates_list is None:
            raise ValueError('Both city_count and cities_coordinates_list cannot be None!')
        # reading the cities name if given
        if cities_coordinates_list is not None:
            self.city_count = len(cities_coordinates_list)
            self.cities = np.array(cities_coordinates_list)
        else:
            self.show_world = False
            self.city_count = city_count
            self.cities = np.random.rand(self.city_count, 2)
            # initializing cities and normalizing them
            self.cities[0, :] = (self.cities[0, :] - np.min(self.cities[0, :])) / (
                    np.max(self.cities[0, :]) - np.min(self.cities[0, :]))
            self.cities[1, :] = (self.cities[1, :] - np.min(self.cities[1, :])) / (
                    np.max(self.cities[1, :]) - np.min(self.cities[1, :]))

        self.cluster_count = int(factor * self.city_count)

        # initializing the clusters on a ring
        if cities_coordinates_list is None:
            t = np.linspace(0, 2 * np.pi, self.cluster_count)
            self.clusters = np.concatenate((0.5 + 0.5 * np.cos(t)[:, np.newaxis], 0.5 + 0.5 * np.sin(t)[:, np.newaxis]),
                                           axis=1)
        else:
            lat_grid = np.linspace(np.min(self.cities[:, 0]), np.max(self.cities[:, 0]), self.cluster_count)[:, np.newaxis]
            long_grid = np.linspace(np.min(self.cities[:, 1]), np.max(self.cities[:, 1]), self.cluster_count)[:, np.newaxis]

            lat_range = [np.min(self.cities, 0)[0], np.max(self.cities, 0)[0]]
            long_range = [np.min(self.cities, 0)[1], np.max(self.cities, 0)[1]]

            # lat_grid = np.random.uniform(lat_range[0], lat_range[1], [self.cluster_count, 1])
            # long_grid = np.random.uniform(long_range[0], long_range[1], [self.cluster_count, 1])
            # self.clusters = np.concatenate((lat_grid, long_grid), axis=1)
            self.clusters = self.__initialize_clusters(lat_range[0], lat_range[1], long_range[0], long_range[1], self.cluster_count)

        # for graphical output
        self.cluster_history = [np.copy(self.clusters)]

        # simulation parameters
        self.eta = eta
        self.sigma_decay = sigma_decay
        self.max_iter = max_iter
        self.sigma = self.cluster_count
        self.epoch = 0
        self.distance_to_all = np.zeros((self.cluster_count, 1))
        self.min_change_to_terminate = min_change_to_terminate
        self.delta_weights = 100

        # initializing the plots
        self.animate_cities = animate_cities(self.cities, self.show_world)
        # self.animate_cities.plot_single_iteration(self.cluster_history, delay=2)

    # ===============================================================================================
    # ===============================================================================================
    def __initialize_clusters(self, lat_min, lat_max, long_min, long_max, N):

        center_lat = (lat_min + lat_max) / 2
        center_long = (long_min + long_max) / 2

        radius_lat = (lat_max - lat_min) / 2
        radius_long = (long_max - long_min) / 2
        radius = min(radius_lat, radius_long) * 0.9

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        points = [(center_long + radius * np.cos(angle), center_lat + radius * np.sin(angle)) for angle in angles]

        return points

    # ===============================================================================================
    # ===============================================================================================
    def _update_plot(self, cluster_idx=-1, delay=0.01, clear_figure=True, save=False):
        # clearing the plot
        self.animate_cities.plot_single_iteration(self.cluster_history, cluster_idx=cluster_idx,
                                                  delay=delay, clear_figure=clear_figure, save=save)

    # ===============================================================================================
    # ===============================================================================================
    def tsp_single_iteration(self):
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
            # self.distance_to_all = np.sum((self.clusters - city) ** 2, axis=1)
            self.distance_to_all = np.array([haversine_distance(cluster, city) for cluster in self.clusters])
            id_closest = np.argmin(self.distance_to_all)

            # updating the cluster centers
            g = self.eta * np.asarray([gaussian(self.sigma, id_clus, id_closest, self.cluster_count)
                                       for id_clus in range(self.cluster_count)])[:, np.newaxis]

            dist_idx = city - self.clusters

            self.clusters += np.repeat(g, 2, 1) * dist_idx

        # saving the new cluster in history
        self.cluster_history.append(np.copy(self.clusters))
        self.delta_weights = np.mean(np.sqrt(np.sum((self.clusters - self.cluster_history[-2]) ** 2, 1)))

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
            self.tsp_single_iteration()
            # visualizing the effect
            self._update_plot(delay=0.01)

        # removing the un-assigned clusters
        self.remove_extra_clusters()
        # updating the plot
        self._update_plot(delay=0.01, save=True)
        # saving the animation as a gif file
        self.save_animation(os.path.join('output', 'animation.gif'))

    # ================================================================================================
    # ================================================================================================
    def remove_extra_clusters(self):
        """

        :return:
        """
        # flag indicating if the cluster is assigned to a city or not
        cluster_flag = self.cluster_count * [False]

        # assigning cities to clusters
        for city in self.cities:

            distance_from_city = np.sum((self.cluster_history[-1] - city) ** 2, axis=1)
            closest_free_cluster_id = int(np.argmin(distance_from_city))

            # checking if the cluster was free
            while cluster_flag[closest_free_cluster_id]:
                distance_from_city[closest_free_cluster_id] = 100
                closest_free_cluster_id = int(np.argmin(distance_from_city))

            # updating the flag
            cluster_flag[closest_free_cluster_id] = True

        clean_clusters = self.cluster_history[-1][cluster_flag]
        # saving the clean cluster
        self.cluster_history.append(clean_clusters)

    # =================================================================================================
    # =================================================================================================
    def save_animation(self, save_dir='./output/animation.gif'):
        self.animate_cities.save_animation(save_dir=save_dir)

# _____________________________________________________________________________________________________
#                   TEST
if __name__ == '__main__':
    tsp = TSP(25)
    tsp.TSP_loop()

    a = 1
