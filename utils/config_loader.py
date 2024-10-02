import yaml



class ConfigLoader:
    def __init__(self):
        config = read_yml(f"../configuration_deboarding.yaml")

        self.buffer_time_gate_connecting = config['simulation_parameters']['buffer_time_gate_connecting']
        self.time_step_duration = config['simulation_parameters']['time_step_duration']
        self.walk_duration = config['simulation_parameters']['walk_duration']
        self.stand_up_duration = config['simulation_parameters']['stand_up_duration']
        self.move_seat_duration = config['simulation_parameters']['move_seat_duration']
        self.t_max_simulation = config['simulation_parameters']['t_max_simulation']
        self.nb_simulation = config['simulation_parameters']['nb_simulation']
        self.gate_close_time = config['simulation_parameters']['gate_close_time']

        self.alpha_weibull = config['simulation_parameters']['luggage_collection']['alpha_weibull']
        self.beta_weibull = config['simulation_parameters']['luggage_collection']['beta_weibull']
        self.min_percentage_has_luggage = config['simulation_parameters']['luggage_collection'][
            'min_percentage_has_luggage']
        self.max_percentage_has_luggage = config['simulation_parameters']['luggage_collection'][
            'max_percentage_has_luggage']

        self.nb_seat_left = config['aircraft_configuration']['nb_seat_left']
        self.nb_seat_right = config['aircraft_configuration']['nb_seat_right']
        self.nb_rows = config['aircraft_configuration']['nb_rows']

        self.min_load_factor = config['passenger_flight_characteristics']['min_load_factor']
        self.max_load_factor = config['passenger_flight_characteristics']['max_load_factor']
        self.min_percentage_prereserved_seats = config['passenger_flight_characteristics'][
            'min_percentage_prebooked_seats']
        self.max_percentage_prereserved_seats = config['passenger_flight_characteristics'][
            'max_percentage_prebooked_seats']

    def display(self):
        print("Simulation Parameters:")
        print(f"Buffer Time for Gate Connecting: {self.buffer_time_gate_connecting}")
        print(f"Time Step Duration: {self.time_step_duration}")
        print(f"Walk Duration: {self.walk_duration}")
        print(f"Stand Up Duration: {self.stand_up_duration}")
        print(f"Move Seat Duration: {self.move_seat_duration}")
        print(f"T Max Simulation: {self.t_max_simulation}")
        print(f"Number of Simulations: {self.nb_simulation}")
        print(f"Gate Close Time: {self.gate_close_time}")

        print("\nLuggage Collection Parameters:")
        print(f"Alpha Weibull: {self.alpha_weibull}")
        print(f"Beta Weibull: {self.beta_weibull}")
        print(f"Min Percentage with Luggage: {self.min_percentage_has_luggage}%")
        print(f"Max Percentage with Luggage: {self.max_percentage_has_luggage}%")

        print("\nAircraft Configuration:")
        print(f"Number of Seats Left: {self.nb_seat_left}")
        print(f"Number of Seats Right: {self.nb_seat_right}")
        print(f"Number of Rows: {self.nb_rows}")

        print("\nPassenger Flight Characteristics:")
        print(f"Min Load Factor: {self.min_load_factor}")
        print(f"Max Load Factor: {self.max_load_factor}")
        print(f"Min Percentage Prebooked Seats: {self.min_percentage_prereserved_seats}%")
        print(f"Max Percentage Prebooked Seats: {self.max_percentage_prereserved_seats}%")


def read_yml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)  # Utiliser safe_load pour éviter d'exécuter du code arbitraire
    return config


if __name__ == "__main__":
    config_loader = ConfigLoader()
    config_loader.display()