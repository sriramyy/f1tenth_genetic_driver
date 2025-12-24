# Follow the Gap + Genetic ML

An autonomous racing algorithm that combines the **Follow the Gap** (FTG) with **Genetic Algorithm optimization** to evolve optimal driving parameters for F1TENTH race cars in simulation.

*Note: This Algorithm requires using the `f1tenth_gym_ros` simulator*

## Prerequisites

- `f1tenth_gym_ros` simulator installed and running
- ROS2 Foxy
- Python 3.8+
- Docker

## Installation

```shell
# First clone into your local workspace

# Build the package
colcon build --packages-select genetic_driver
source install/local_setup.bash
# + source any other local files required
```

## Running the Algorithm

### 1. Start the F1TENTH Simulator
```shell
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

### 2. Open New Terminals and Source Setup

For each new terminal:
```shell
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
```

### 3. Run the Algorithm

**Terminal 1 - Follow the Gap Driver:**
```shell
ros2 run genetic_driver ftg
```

**Terminal 2 - Genetic Supervisor:**
```shell
ros2 run genetic_driver supervisor
```

The supervisor will automatically start evolving after a couple seconds.

## Interactive Controls

While the algorithm is running, you can interact via the supervisor terminal:

- **Press `f` + Enter**: View current champion genome stats (fastest lap)
- **Press `r` + Enter**: Manually reset the current simulation run

Example output:
```
 🏆 CURRENT CHAMPION (As of Generation 5)
     GenomeID : 12
     Time     : 24.37s
     Score    : 31.55
     Genes    : [1.234, 5.67, 2.1, ...]
```

## Implementation Details

### Architecture

```
GeneticSupervisor (ROS2 Node)
├── Manages simulation lifecycle
├── Tracks lap times & crashes
├── Publishes evolved parameters
└── Handles user input

GeneticML (Core Algorithm)
├── Maintains population of genomes
├── Calculates fitness scores
├── Selects & breeds best performers
└── Mutates genes for diversity

FollowTheGap (FTG Driver)
├── Processes LIDAR scans
├── Implements gap detection
├── Publishes steering/speed commands
└── Reports crash detection
```

### Configuration

Edit parameters in `genetic_ml.py`:

```python
self.population_size = 20      # Number of genomes per generation
self.max_generations = 50       # Total generations to evolve
self.elitism_percent = 0.2      # Top 20% always survive
```

### Genetic Parameters

Each genome contains these evolving parameters (in `GeneticParameters`):

- `MAX_LIDAR_DIST` - Maximum LIDAR range for gap detection
- `STRAIGHT_SPEED` - Target velocity on straights
- `TURN_SPEED` - Target velocity during turns
- `STEERING_SCALE` - Steering sensitivity multiplier
- *...and more* (defined in `parameters.py`)

### Fitness Function

```python
fitness_score = (1000 / lap_time) - (crash_count * 500)
```

- **Positive side**: Rewards faster lap times
- **Negative side**: Heavy penalty (500 points) for each crash
- **Result**: Evolves safe, fast driving strategies

### Selection & Breeding

1. **Elitism**: Top 20% of genomes automatically advance
2. **Crossover**: Random parents breed offspring
3. **Mutation**: Offspring genes randomly mutate for diversity
4. **Replacement**: New generation replaces worst performers

## Project Structure

```
genetic_driver/
├── genetic_ml.py           # Core genetic algorithm
├── parameters.py           # GeneticParameters class
├── supervisor.py           # ROS2 supervisor node
├── ftg_driver.py          # Follow the Gap implementation
└── README.md
```
