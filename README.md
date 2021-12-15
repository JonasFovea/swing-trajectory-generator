# swing-trajectory-generator
Swing trajectory generation for walking robots.


## Motivation
To enable walking robots to walk, their legs must follow trajectories in a coordinated manner to simultaneously move and maintain their balance.
Such a leg trajectory is divided into two sections: stance and swing phase.
In the stance phase, the leg is moved while it is placed on the ground, which results in a force being exerted on the rest of the robot, causing it to move.
In the swing phase, on the other hand, the leg is lifted and moved forward in the direction of travel to its new stance point, around which the following stance phase then takes place.

Since the swinging movement takes place freely in space and is initially defined only by its start and end point, it is necessary to adapt this to the conditions of the robot and, if necessary, to define further desired parameters of the trajectory.
These further parameters can be, for example, speeds, torques, minimum and maximum leg heights, impact angles or distances to obstacles.
A trajectory that meets these requirements must then be defined in a mathematically describable way and calculated for each individual movement.

If the leg hits an obstacle during the swing phase, it must be possible to calculate another trajectory for this incident, which also moves within the specified framework conditions and can then avoid the obstacle.

Another crucial requirement for the calculation of these trajectories is the computing power needed for this purpose. On the one hand, it is absolutely necessary for such a mobile system that these calculations take place in real time, and on the other hand, it can happen that these calculations have to take place internally in the robot system, in which the cooling and computing power can be severely limited by the spatial conditions of the robot itself.

It is therefore necessary to find a method that generates an swing trajectory in real time with the lowest possible computational effort and taking into account the requirements specified by the robot and additional own requirements. 

## Installation
Run the following to install:

```shell
pip install swinggen
```

### Development installation
To install `swinggen` for development, along with tools you need to develop and run tests, run the following in your virtualenv:
```shell
$ pip install -e .[dev]
```


## Usage
To use the spline trajectory visualization with the AdjusterGui simply run the following command in your terminal:

```shell
$ python3 -m swinggen
``` 
