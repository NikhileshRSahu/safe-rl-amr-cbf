import math
import numpy as np
import matplotlib.pyplot as plt

class RealisticUnicycleModel:
    """
    First-Order Non-Holonomic Differential-Drive Unicycle Kinematic Model
    Features: Exact integration, Motor Lag, Inverse/Forward Kinematics, Wheel Slip.
    """
    def __init__(self, dt: float = 0.1, tau: float = 0.15, seed: int = 42):
        self.dt = dt
        self.tau = tau  # Motor inertia time constant
        self.rng = np.random.default_rng(seed)
        
        # Domain Randomization (Sim-to-Real gap bridging)
        self.wheel_radius = self.rng.uniform(0.09, 0.11)  # Nominal ~0.1m
        self.wheel_base = self.rng.uniform(0.45, 0.55)    # Nominal ~0.5m
        
        # State: [x, y, theta]
        self.pose = np.zeros(3, dtype=np.float64)
        
        # Physical actuator states
        self.v_actual = 0.0
        self.omega_actual = 0.0

    def step(self, v_cmd: float, omega_cmd: float):
        """Advances the robot dynamics by one timestep dt."""
        
        # Motor Inertia Lag (1st-Order Low-Pass Filter)
        alpha_lag = self.dt / (self.tau + self.dt)
        v_target = self.v_actual + alpha_lag * (v_cmd - self.v_actual)
        omega_target = self.omega_actual + alpha_lag * (omega_cmd - self.omega_actual)

        # Inverse Kinematics (Body velocities -> Wheel angular velocities)
        w_right_target = (v_target + (omega_target * self.wheel_base / 2.0)) / self.wheel_radius
        w_left_target  = (v_target - (omega_target * self.wheel_base / 2.0)) / self.wheel_radius

        # Non-Idealities: Gaussian Encoder Noise & Wheel Slip
        noise_scale = 0.05
        w_right_actual = (w_right_target + self.rng.normal(0, noise_scale)) * self.rng.uniform(0.92, 1.0)
        w_left_actual  = (w_left_target + self.rng.normal(0, noise_scale)) * self.rng.uniform(0.92, 1.0)

        # Forward Kinematics (Wheel velocities -> Body velocities)
        self.v_actual = (self.wheel_radius / 2.0) * (w_right_actual + w_left_actual)
        self.omega_actual = (self.wheel_radius / self.wheel_base) * (w_right_actual - w_left_actual)

        # Exact Circular-Arc Integration
        x, y, theta = self.pose
        
        if abs(self.omega_actual) < 1e-8: # Linear motion limit
            x_new = x + self.v_actual * math.cos(theta) * self.dt
            y_new = y + self.v_actual * math.sin(theta) * self.dt
            theta_new = theta
        else: # Turning motion
            theta_new = theta + self.omega_actual * self.dt
            x_new = x + (self.v_actual / self.omega_actual) * (math.sin(theta_new) - math.sin(theta))
            y_new = y - (self.v_actual / self.omega_actual) * (math.cos(theta_new) - math.cos(theta))

        # Normalize angle to [-pi, pi]
        theta_new = (theta_new + math.pi) % (2 * math.pi) - math.pi
        
        self.pose = np.array([x_new, y_new, theta_new])
        return self.pose, self.v_actual, self.omega_actual

def simulate_and_plot():
    robot = RealisticUnicycleModel()
    
    steps = 50
    dt = robot.dt
    
    # Data tracking for plotting
    history_x, history_y = [], []
    history_v_cmd, history_v_act = [], []
    time_steps = []
    
    # Run the simulation for 50 steps (5 seconds)
    for i in range(steps):
        # Command 1.0 m/s forward and a constant left turn
        v_command = 1.0
        w_command = 0.5
        
        pose, v_act, w_act = robot.step(v_cmd=v_command, omega_cmd=w_command)
        
        # Record data
        time_steps.append(i * dt)
        history_x.append(pose[0])
        history_y.append(pose[1])
        history_v_cmd.append(v_command)
        history_v_act.append(v_act)

    # --- Plotting the Results ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graph 1: X-Y Trajectory
    ax1.plot(history_x, history_y, marker='o', markersize=4, linestyle='-', color='b')
    ax1.set_title("Robot X-Y Trajectory (Circular Arc Integration)")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.grid(True)
    ax1.axis('equal') # Keeps the aspect ratio accurate
    
    # Graph 2: Motor Lag Profile
    ax2.plot(time_steps, history_v_cmd, 'r--', label="Commanded Velocity (1.0 m/s)")
    ax2.plot(time_steps, history_v_act, 'g-', linewidth=2, label="Actual Velocity (with Inertia)")
    ax2.set_title("Velocity Response showing Motor Inertia Lag")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Linear Velocity (m/s)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_and_plot()