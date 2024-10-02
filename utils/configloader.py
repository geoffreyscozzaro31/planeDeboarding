import os

import yaml


class ConfigLoader:
    def __init__(self, config_file_path):
        configuration = read_yml(config_file_path)

        # Simulation Parameters
        sim_params = configuration['simulation_parameters']
        self.buffer_time_gate_connecting = sim_params['buffer_time_gate_connecting_seconds']
        self.time_step_duration = sim_params['time_step_duration_seconds']
        self.walk_duration = sim_params['walk_duration_step']
        self.stand_up_duration = sim_params['stand_up_duration_step']
        self.move_seat_duration = sim_params['move_seat_duration_step']
        self.t_max_simulation = sim_params['t_max_simulation_step']
        self.nb_simulation = sim_params['nb_simulation']
        self.gate_close_time = sim_params['gate_close_time_seconds']

        luggage_collection = sim_params['luggage_collection']
        self.alpha_weibull = luggage_collection['alpha_weibull']
        self.beta_weibull = luggage_collection['beta_weibull']
        self.min_percentage_has_luggage = luggage_collection['min_percentage_has_luggage']
        self.max_percentage_has_luggage = luggage_collection['max_percentage_has_luggage']

        aircraft_config = configuration['aircraft_configuration']
        self.nb_seat_left = aircraft_config['nb_seat_left']
        self.nb_seat_right = aircraft_config['nb_seat_right']
        self.nb_rows = aircraft_config['nb_rows']
        self.nb_dummy_rows = aircraft_config['nb_dummy_rows']

        passenger_char = configuration['passenger_flight_characteristics']
        self.min_load_factor = passenger_char['min_load_factor']
        self.max_load_factor = passenger_char['max_load_factor']
        self.min_percentage_prereserved_seats = passenger_char['min_percentage_prebooked_seats']
        self.max_percentage_prereserved_seats = passenger_char['max_percentage_prebooked_seats']
        self.day_label = passenger_char['day_label']

    def display(self):
        print("Simulation Parameters:")
        print(f"Buffer Time for Gate Connecting (seconds): {self.buffer_time_gate_connecting}")
        print(f"Time Step Duration (seconds): {self.time_step_duration}")
        print(f"Walk Duration (steps): {self.walk_duration}")
        print(f"Stand Up Duration (steps): {self.stand_up_duration}")
        print(f"Move Seat Duration (steps): {self.move_seat_duration}")
        print(f"T Max Simulation (steps): {self.t_max_simulation}")
        print(f"Number of Simulations: {self.nb_simulation}")
        print(f"Gate Close Time (seconds): {self.gate_close_time}")

        print("\nLuggage Collection Parameters:")
        print(f"Alpha Weibull: {self.alpha_weibull}")
        print(f"Beta Weibull: {self.beta_weibull}")
        print(f"Min Percentage with Luggage: {self.min_percentage_has_luggage}%")
        print(f"Max Percentage with Luggage: {self.max_percentage_has_luggage}%")

        print("\nAircraft Configuration:")
        print(f"Number of Seats Left: {self.nb_seat_left}")
        print(f"Number of Seats Right: {self.nb_seat_right}")
        print(f"Number of Rows: {self.nb_rows}")
        print(f"Number of Dummy Rows: {self.nb_dummy_rows}")

        print("\nPassenger Flight Characteristics:")
        print(f"Min Load Factor: {self.min_load_factor}")
        print(f"Max Load Factor: {self.max_load_factor}")
        print(f"Min Percentage Prebooked Seats: {self.min_percentage_prereserved_seats}%")
        print(f"Max Percentage Prebooked Seats: {self.max_percentage_prereserved_seats}%")
        print(f"Day Label: {self.day_label}")


def read_yml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    dir_config_file = os.path.dirname(os.getcwd())
    config_file_path = os.path.join(dir_config_file, "configuration_deboarding.yaml")
    config = ConfigLoader(config_file_path)
    config.display()
