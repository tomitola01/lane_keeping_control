import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class Car:

    def __init__(self, length=2.3, velocity=5, x=0, y=0, theta=0):
        self.length = length
        self.velocity = velocity
        self.x = x
        self.y = y
        self.theta = theta

    def move(self, steering, dt):
        def dynamics(t, z):
            theta = z[2]
            return [self.velocity * np.cos(theta),
                    self.velocity * np.sin(theta),
                    self.velocity * np.tan(steering) / self.length]

        z_initial = [self.x, self.y, self.theta]
        sol = solve_ivp(dynamics,
                        [0, dt],
                        z_initial)
        self.x = sol.y[0][-1]
        self.y = sol.y[1][-1]
        self.theta = sol.y[2][-1]

class PIDController:
    def __init__(self, kp=0.1, ki=0, kd=0, ts=0.01):
        self.kp = kp
        self.ki= ki*ts
        self.kd = kd/ts
        self.ts = ts
        self.prev_error = None
        self.sum_errors = 0

    def control(self, y, y_sp=0):
        error = y_sp - y
        u = self.kp*error

        if self.prev_error is not None:
            u += self.kd*(error-self.prev_error)

        u += self.ki*self.sum_errors

        self.prev_error = error
        self.sum_errors += error
        return u

sampling_time = 0.025
murphy = Car(y=0.5)
controller = PIDController(kp =0.2, ki = 0.1, kd = 0.3, ts = sampling_time)
num_points =2000
u_disturbance = 1*np.pi/180
y_black_box = np.array([murphy.y])
x_black_box = np.array([murphy.x])
theta_black_box = np.array([murphy.theta])
u_black_box = np.array((controller.control(y=murphy.y))+u_disturbance)
for t in range(num_points):
    steering = controller.control(y=murphy.y)
    u = steering+u_disturbance
    murphy.move(steering+u_disturbance, sampling_time)
    y_black_box = np.append(y_black_box, murphy.y)
    x_black_box = np.append(x_black_box, murphy.x)
    theta_black_box = np.append(theta_black_box, murphy.theta)
    u_black_box = np.append(u_black_box, u)

t_span = sampling_time*np.arange(num_points+1)
# plt.plot(t_span, y_black_box, label='y Trajectory')
# plt.plot(t_span, x_black_box, label='x Trajectory')
# plt.plot(x_black_box, y_black_box)
plt.plot(t_span, u_black_box)
plt.xlabel("Time (s)")
plt.ylabel("Steering Angle (rad)")
plt.grid()
plt.show()
