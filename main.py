import argparse
import os, sys
print(f'sys path: {sys.path}')
from lib.database_library import cities_db as cities_db, filter_countries
from lib import tsp

# ============================================================
def main(db_dir, cities_list=None, countries_list=None, num_cities_per_country=5):
    cities_database = cities_db(db_dir)
    # print(f"db: {cities_database}")

    if countries_list is not None:
        db_new = filter_countries(cities_database, countries_list, num_cities_per_country)

    for row in db_new:
        print(row)
    cities_coords = [[row['lat'], row['lng']] for row in db_new]
    cities_list = [row['city'] for row in db_new]
    print(f'cities: {cities_list}')
    print(f'coordinates: {cities_coords}')

    # tsp.plot_cities_on_map(coordinates=lat_lng_list, show_plot=True)
    tsp_solver = tsp.TSP(cities_coordinates_list=cities_coords)
    tsp_solver.TSP_loop()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', default='worldcities', type=str)
    parser.add_argument('--countries', nargs='+', required=False, type=str)
    parser.add_argument('--cities', nargs='+', required=False, type=str)
    parser.add_argument('--max_cities_per_country', default=5, type=int)
    args = parser.parse_args()
    print(f'running for the country list: {args.countries}')
    if (args.countries is None) and (args.cities is None):
        raise ValueError('Both countries and cities cannot be None! You should set one of them!')
    if (args.countries is not None) and (args.cities is not None):
        raise ValueError('You should only use either cities or countries!')

    main(args.db_name, args.cities, args.countries, args.max_cities_per_country)