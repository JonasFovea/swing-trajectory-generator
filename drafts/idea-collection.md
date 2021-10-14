# Idea Collection

## The task
+ Move a foot from it's position to a desired position by lifting it up and then moving it
+ If a colision is detected, calculate a higher trajectory to step over the obstacle


## Different path planning methods

*see [1] p.203f*
+ Cubic or higher order polynomials
  - Position and speed (and further derivatives for higher order polyn of a joint can be defined for each point
  - Inverse Kinematics is used to convert from cartesian to angular coordinates
  - Velocities can alsoomials) be chosen automatically by using a simple heuristic or setting a fixed linear acceleration
+ Linear paths with parabolic blends
  - to ensure a constant acceleration, linear joint movements are blended with parabolas near the start and goal position
  - therefore very simple to calculate
  - controlled acceleration

All of these methods are generating joint angle functions $\Theta(t)$ describing a path between two points in space.

## Trajecory shapes
+ Elliptical trajectory *(see [2] p.103f)*
  - simple calculation of the cartesian path
  - smooth path in the cartesian space
  - perpendicular liftoff and approach angle
  - high number of via points necessary to maintain shape
+ Triangular trajectory
  - very simple shape
  - not smooth
  - short time at maximum height -> collision risk increased
+ Rectangular trajectory
  - simple shape
  - not smooth
  - maximum height is maintained all the time exept while lifting up and seting down the foot
+ Spline trajectory (cubic)
  - moderate calculation cost
  - smooth transfer between liftoff/set down and linear movement 
  - duration of maximum height is dependent on the slope of the splines
  

# Bibliography
1. Craig, J. J., 2005. Introduction to robotics : mechanics and control / John J. Craig. 3. ed. Pearson Prentice Hall. 
2. Paskarbeit, J., 2017. Consider the robot - Abstraction of bioinspired leg coordination and its application to a hexapod robot under consideration of technical constraints, Bielefeld: Universität Bielefeld.
