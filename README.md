## Aircraft Deboarding and Connecting Passengers

### Introduction

This project is based on research work described in the following paper:  
```
```

The aim of this project is to study the impact of passenger seat allocation strategies on the transfer times of connecting passengers. During the check-in process, passengers may pre-reserve seats for a fee. Once these reservations are made, the remaining passengers are assigned seats by the airline, typically in a random fashion.

This project explores the potential benefits of a more strategic approach, where connecting passengers with tight transfer windows are allocated seats closer to the aircraft exit.

The project consists of two main components:
1. **Deboarding Simulation**: Simulates the deboarding process of passengers from an aircraft.
2. **Connecting Simulation**: Simulates the connections of passengers to their onward flights throughout a day of airport operations.

### Deboarding Simulation

We use a cellular automata approach to model the deboarding process. The aircraft is represented as a grid where each cell is either occupied by a passenger or empty. Passengers move to adjacent cells at each time step based on specific rules.

Two disembarkation strategies are simulated:
1. **Courtesy Rule**: Disembarkation proceeds row by row, starting from the front of the aircraft. If a passenger wants to move to the aisle but there is another passenger behind them, the one at the front takes priority while the other waits.
2. **Aisle Priority Rule**: Passengers closer to the aisle take priority. If a passenger in the aisle wants to move forward and there is another passenger in the row who wants to access the aisle, the passenger already in the aisle moves first, and the other must wait.

### Connecting Simulation

We used a data-driven approach to simulate connecting passengers, based on historical flight schedules at Paris Charles de Gaulle (CDG) Airport. Due to confidentiality agreements, the original data cannot be provided, but preprocessed files that include connecting passenger information are included.

The simulation works by:
- Reviewing the flight schedule to identify potential connecting flights, based on arrival and departure times.
- Selecting flights with acceptable connection windows (45 minutes to 3 hours).
- Assigning a predefined proportion of arriving passengers as connecting passengers, and distributing them among different departing flights.

Simulated passengers are generated using scheduled arrival and departure times, while actual times (accounting for delays) are used in the simulation.

### Project Architecture
*To be written.*

### Installation
*To be written.*

### Running the Project
*To be written.*

### Authors
Geoffrey Scozzaro

### Acknowledgements
The code for the aircraft disembarkation simulation is inspired by a GitHub project that models the boarding process: [https://github.com/pszemsza/plane_boarding](https://github.com/pszemsza/plane_boarding).


