import yaml


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Example usage
if __name__ == "__main__":
    config_file = "configuration_deboarding.yaml"
    config = load_config(config_file)

    simulation_params = config['simulation_parameters']
    aircraft_config = config['aircraft_configuration']
    passenger_characteristics = config['passenger_flight_characteristics']

    print(f"Buffer time for gate connections: {simulation_params['buffer_time_gate_connecting']} seconds")
    print(f"Number of rows in the aircraft: {aircraft_config['nb_rows']}")
    print(f"Percentage of passengers with luggage: "
          f"{simulation_params['luggage_collection']['min_percentage_has_luggage']}% to "
          f"{simulation_params['luggage_collection']['max_percentage_has_luggage']}%")
