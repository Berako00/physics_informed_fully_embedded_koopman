import torch

def generate_data(x1range, x2range, numICs, mu, lam, T, dt, seed):
   # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Generate random initial conditions for x1 and x2
    x1 = (x1range[1] - x1range[0]) * torch.rand(numICs) + x1range[0]
    x2 = (x2range[1] - x2range[0]) * torch.rand(numICs) + x2range[0]
    u = torch.rand(numICs, T) - 0.5

    dt_lam = dt * lam

    # Preallocate xu with shape [numICs, lenT, 3]
    xuk = torch.zeros(numICs, T, 3, dtype=torch.float32)

    xuk[:, :, 2] = u

    for t in range(T):

        xuk[:, t, 0] = x1
        xuk[:, t, 1] = x2

        dx1 = dt * mu * x1 + dt*u[:, t-1]
        dx2 = dt_lam * (x2 - x1**2)

        x1 += dx1
        x2 += dx2

    return xuk

def generate_data_unforced(x1range, x2range, numICs, mu, lam, T_step, dt, seed):
   # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Generate random initial conditions for x1 and x2
    x1 = (x1range[1] - x1range[0]) * torch.rand(numICs) + x1range[0]
    x2 = (x2range[1] - x2range[0]) * torch.rand(numICs) + x2range[0]
    u = torch.rand(numICs, T_step)*0 # Sets input to be 0

    dt_lam = dt * lam

    # Preallocate xu with shape [numICs, lenT, 3]
    xuk = torch.zeros(numICs, T_step, 3, dtype=torch.float32)

    xuk[:, :, 2] = u

    for t in range(T_step):

        xuk[:, t, 0] = x1
        xuk[:, t, 1] = x2

        dx1 = dt * mu * x1 + dt*u[:, t-1]
        dx2 = dt_lam * (x2 - x1**2)

        x1 += dx1
        x2 += dx2

    return xuk

def DataGenerator(x1range, x2range, numICs, mu, lam, T, dt):

    # Create test, validation, and training tensors with different percentages of numICs
    seed = 1
    test_tensor = generate_data(x1range, x2range, round(0.1 * numICs), mu, lam, T, dt, seed)

    seed = 2
    val_tensor = generate_data(x1range, x2range, round(0.2 * numICs), mu, lam, T, dt, seed)

    seed = 3
    train_tensor = generate_data(x1range, x2range, round(0.7 * numICs), mu, lam, T, dt, seed)

    return train_tensor, test_tensor, val_tensor


def generate_two_link_data(q1_range, q2_range, dq1_range, dq2_range, numICs, T, dt, seed,
                           tau_max=1.0,
                           m1=1.0, m2=1.0,
                           l1=1.0, l2=1.0,
                           g=9.81):
    """
    Generate simulation data for a two-link planar manipulator.

    Parameters:
        q1_range, q2_range : tuple (min, max)
            Ranges for initial joint angles (in radians).
        dq1_range, dq2_range : tuple (min, max)
            Ranges for initial joint angular velocities.
        tau_max : float
            Maximum absolute torque applied at each joint.
        m1, m2 : float
            Masses of link 1 and link 2.
        l1, l2 : float
            Lengths of link 1 and link 2.
        g : float
            Acceleration due to gravity.

    Returns:
        data : torch.Tensor of shape [numICs, T, 6]
            For each trajectory and each time step, the first four entries are
            [q1, q2, dq1, dq2] and the last two are the applied torques [tau1, tau2].
    """
    torch.manual_seed(seed)

    # Compute the moments of inertia dynamically
    lc1, lc2 = l1 / 2, l2 / 2  # Assuming center of mass at middle of each link
    I1 = m1 * lc1**2  # Moment of inertia of link 1
    I2 = m2 * lc2**2  # Moment of inertia of link 2

    # Print computed values for debugging
    print(f"Computed Inertia: I1 = {I1:.4f}, I2 = {I2:.4f}")

    # Generate initial conditions
    q1 = (q1_range[1] - q1_range[0]) * torch.rand(numICs) + q1_range[0]
    q2 = (q2_range[1] - q2_range[0]) * torch.rand(numICs) + q2_range[0]
    dq1 = (dq1_range[1] - dq1_range[0]) * torch.rand(numICs) + dq1_range[0]
    dq2 = (dq2_range[1] - dq2_range[0]) * torch.rand(numICs) + dq2_range[0]

    # Generate random control torques for each time step
    tau = (torch.rand(numICs, T, 2) - 0.5) * 2 * tau_max

    # Preallocate data tensor
    data = torch.zeros(numICs, T, 6, dtype=torch.float32)
    data[:, :, 4:] = tau

    for t in range(T):
        # Save current state
        data[:, t, 0] = q1
        data[:, t, 1] = q2
        data[:, t, 2] = dq1
        data[:, t, 3] = dq2

        # Precompute trigonometric functions
        cos_q2 = torch.cos(q2)
        sin_q2 = torch.sin(q2)
        cos_q1q2 = torch.cos(q1 + q2)

        # Compute elements of the inertia matrix M(q)
        M11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos_q2)
        M12 = I2 + m2 * (lc2**2 + l1 * lc2 * cos_q2)
        M21 = M12
        M22 = I2 + m2 * lc2**2

        # Compute the inverse of the inertia matrix
        detM = M11 * M22 - M12 * M21
        invM11 = M22 / detM
        invM12 = -M12 / detM
        invM21 = -M21 / detM
        invM22 = M11 / detM

        # Compute Coriolis terms
        h = -m2 * l1 * lc2 * sin_q2
        C1 = h * dq2 * (2 * dq1 + dq2)
        C2 = h * dq1**2

        # Compute gravity terms
        G1 = (m1 * lc1 + m2 * l1) * g * torch.cos(q1) + m2 * lc2 * g * cos_q1q2
        G2 = m2 * lc2 * g * cos_q1q2

        # Get current torques
        tau_t = tau[:, t, :]

        # Compute joint accelerations
        rhs1 = tau_t[:, 0] - C1 - G1
        rhs2 = tau_t[:, 1] - C2 - G2

        ddq1 = invM11 * rhs1 + invM12 * rhs2
        ddq2 = invM21 * rhs1 + invM22 * rhs2

        # Euler integration
        dq1 = dq1 + ddq1 * dt
        dq2 = dq2 + ddq2 * dt
        q1 = q1 + dq1 * dt
        q2 = q2 + dq2 * dt

    return data

def generate_two_link_lab_data(q1_range, q2_range, dq1_range, dq2_range, numICs, T, dt, seed,
                           tau_max=7.5,
                           mL1=0.196, mL2=0.14,
                           L1=0.3, L2=0.29,
                           g=9.81):
    """
    Generate simulation data for a two-link planar manipulator.

    Parameters:
        q1_range, q2_range : tuple (min, max)
            Ranges for initial joint angles (in radians).
        dq1_range, dq2_range : tuple (min, max)
            Ranges for initial joint angular velocities.
        tau_max : float
            Maximum absolute torque applied at each joint.
        m1, m2 : float
            Masses of link 1 and link 2.
        l1, l2 : float
            Lengths of link 1 and link 2.
        g : float
            Acceleration due to gravity.

    Returns:
        data : torch.Tensor of shape [numICs, T, 6]
            For each trajectory and each time step, the first four entries are
            [q1, q2, dq1, dq2] and the last two are the applied torques [tau1, tau2].
    """
    torch.manual_seed(seed)
    NG = 172  # Gear ratio from the document
    Imot = 15.2 * (10**-7)  # motor inertia
    IG   = 7.92 * (10**-7)  # gearbox inertia
    b1 = 0.054  # Width link 1
    b2 = 0.044  # Width link 2

    # Compute the moments of inertia dynamically
    lc1, lc2 = 0.188, 0.178  # center of mass
    I1 = (1.0 / 12.0) * mL1 * (b1**2 + L1**2) + NG**2 * (Imot + IG)
    I2 = (1.0 / 12.0) * mL2 * (b2**2 + L2**2) + NG**2 * (Imot + IG)

    m_cm1 = 1.008
    m_cm2 = 0.29
    lc1 = 0.188
    lc2 = 0.178

    B1 = 1.45
    B2 = 0.493
    # Print computed values for debugging
    print(f"Computed Inertia: I1 = {I1:.4f}, I2 = {I2:.4f}")

    # Generate initial conditions
    q1 = (q1_range[1] - q1_range[0]) * torch.rand(numICs) + q1_range[0]
    q2 = (q2_range[1] - q2_range[0]) * torch.rand(numICs) + q2_range[0]
    dq1 = (dq1_range[1] - dq1_range[0]) * torch.rand(numICs) + dq1_range[0]
    dq2 = (dq2_range[1] - dq2_range[0]) * torch.rand(numICs) + dq2_range[0]

    # --- NEW PART: choose 20% of trajectories to be “all-zero” ---
    n_zero = int(0.3 * numICs)                 # how many to zero out
    perm   = torch.randperm(numICs)            # random shuffle of [0..numICs-1]
    zero_idx = perm[:n_zero]                   # pick first n_zero of them
    traj_zero_mask = torch.zeros(numICs, dtype=torch.bool)
    traj_zero_mask[zero_idx] = True            # mark those trajectories

    # Generate all torques randomly…
    tau = (torch.rand(numICs, T, 2) - 0.5) * 2 * tau_max

    # …but then zero out *entire* trajectories:
    # for each i where traj_zero_mask[i] is True, set tau[i, :, :] = 0
    tau[traj_zero_mask, :, :] = 0
    # -----------------------------------------------------------------
                              
    # Preallocate data tensor
    data = torch.zeros(numICs, T, 6, dtype=torch.float32)
    data[:, :, 4:] = tau

    for t in range(T):
        # Save current state
        data[:, t, 0] = q1
        data[:, t, 1] = q2
        data[:, t, 2] = dq1
        data[:, t, 3] = dq2

        # Precompute trigonometric functions
        cos_q1 = torch.cos(q1)
        sin_q1 = torch.sin(q1)
        cos_q2 = torch.cos(q2)
        sin_q2 = torch.sin(q2)
        cos_q1q2 = torch.cos(q1 + q2)

        # Compute H values using precomputed trigonometry
        H1 = (2 * m_cm2 * L1 * cos_q2 + (L1**2 + lc2**2) * m_cm2 + m_cm1 * lc1**2 + I1 + I2)

        H2 = m_cm2 * L1 * cos_q2

        H3 = m_cm2 * lc2**2 + I2

        H4 = m_cm2 * L1 * sin_q2

        HG1 = ((m_cm2 * L1 * cos_q1) + (lc2 * cos_q1q2) + (m_cm1 * lc1 * cos_q1)) * g

        H5 = m_cm2 * L1 * cos_q2 + (m_cm2 * lc2**2 + I2)

        H6 = m_cm2 * L1 * lc2 * sin_q2

        HG2 = m_cm2 * lc2 * cos_q1q2 * g

        # Get current torques
        tau_t = tau[:, t, :]
        tau1 = tau_t[:, 0]         # shape: [numICs]
        tau2 = tau_t[:, 1]         # shape: [numICs]

        # Compute joint accelerations (ddq1, ddq2)
        ddq1 = (H4 * dq1 * dq2 - H2 * dq2 + H6 * (dq1 + dq2) + HG2 + H4 * dq2**2 - HG1) / (H1 - H5) + ((tau1 - B1 * dq1) - (tau2 - B2 * dq2)) / (H1 - H5)

        ddq2 = ((H4 * H1 - 2 * H4 * H5) * dq1 * dq2 - H4 * H5 * dq2**2 - H6 * H1 * dq1 + (H2 * H5 - H6 * H1) * dq2) / (H1 * H3 - H3 * H5) - (H1 * HG2 - H5 * HG1 - H1 * (tau2 - B2 * dq2) + H5 * (tau1 - B1 * dq1)) / (H1 * H3 - H3 * H5)

        # Euler integration
        dq1 = dq1 + ddq1 * dt
        dq2 = dq2 + ddq2 * dt
        q1 = q1 + dq1 * dt
        q2 = q2 + dq2 * dt

    return data


def generate_two_link_lab_data_unforced(q1_range, q2_range, dq1_range, dq2_range, numICs, T, dt, seed,
                           tau_max=7.5,
                           mL1=0.196, mL2=0.14,
                           L1=0.3, L2=0.29,
                           g=9.81):
    """
    Generate simulation data for a two-link planar manipulator.

    Parameters:
        q1_range, q2_range : tuple (min, max)
            Ranges for initial joint angles (in radians).
        dq1_range, dq2_range : tuple (min, max)
            Ranges for initial joint angular velocities.
        tau_max : float
            Maximum absolute torque applied at each joint.
        m1, m2 : float
            Masses of link 1 and link 2.
        l1, l2 : float
            Lengths of link 1 and link 2.
        g : float
            Acceleration due to gravity.

    Returns:
        data : torch.Tensor of shape [numICs, T, 6]
            For each trajectory and each time step, the first four entries are
            [q1, q2, dq1, dq2] and the last two are the applied torques [tau1, tau2].
    """
    torch.manual_seed(seed)
    NG = 172  # Gear ratio from the document
    Imot = 15.2 * (10**-7)  # motor inertia
    IG   = 7.92 * (10**-7)  # gearbox inertia
    b1 = 0.054  # Width link 1
    b2 = 0.044  # Width link 2

    # Compute the moments of inertia dynamically
    lc1, lc2 = 0.188, 0.178  # center of mass
    I1 = (1.0 / 12.0) * mL1 * (b1**2 + L1**2) + NG**2 * (Imot + IG)
    I2 = (1.0 / 12.0) * mL2 * (b2**2 + L2**2) + NG**2 * (Imot + IG)

    m_cm1 = 1.008
    m_cm2 = 0.29
    lc1 = 0.188
    lc2 = 0.178

    B1 = 1.45
    B2 = 0.493
    # Print computed values for debugging
    print(f"Computed Inertia: I1 = {I1:.4f}, I2 = {I2:.4f}")

    # Generate initial conditions
    q1 = (q1_range[1] - q1_range[0]) * torch.rand(numICs) + q1_range[0]
    q2 = (q2_range[1] - q2_range[0]) * torch.rand(numICs) + q2_range[0]
    dq1 = (dq1_range[1] - dq1_range[0]) * torch.rand(numICs) + dq1_range[0]
    dq2 = (dq2_range[1] - dq2_range[0]) * torch.rand(numICs) + dq2_range[0]

    # Generate random control torques for each time step
    tau = (torch.rand(numICs, T, 2) - 0.5) * 2 * tau_max * 0

    # Preallocate data tensor
    data = torch.zeros(numICs, T, 6, dtype=torch.float32)
    data[:, :, 4:] = tau

    for t in range(T):
        # Save current state
        data[:, t, 0] = q1
        data[:, t, 1] = q2
        data[:, t, 2] = dq1
        data[:, t, 3] = dq2

        # Precompute trigonometric functions
        cos_q1 = torch.cos(q1)
        sin_q1 = torch.sin(q1)
        cos_q2 = torch.cos(q2)
        sin_q2 = torch.sin(q2)
        cos_q1q2 = torch.cos(q1 + q2)

        # Compute H values using precomputed trigonometry
        H1 = (2 * m_cm2 * L1 * cos_q2 + (L1**2 + lc2**2) * m_cm2 + m_cm1 * lc1**2 + I1 + I2)

        H2 = m_cm2 * L1 * cos_q2

        H3 = m_cm2 * lc2**2 + I2

        H4 = m_cm2 * L1 * sin_q2

        HG1 = ((m_cm2 * L1 * cos_q1) + (lc2 * cos_q1q2) + (m_cm1 * lc1 * cos_q1)) * g

        H5 = m_cm2 * L1 * cos_q2 + (m_cm2 * lc2**2 + I2)

        H6 = m_cm2 * L1 * lc2 * sin_q2

        HG2 = m_cm2 * lc2 * cos_q1q2 * g

        # Get current torques
        tau_t = tau[:, t, :]
        tau1 = tau_t[:, 0]         # shape: [numICs]
        tau2 = tau_t[:, 1]         # shape: [numICs]

        # Compute joint accelerations (ddq1, ddq2)
        ddq1 = (H4 * dq1 * dq2 - H2 * dq2 + H6 * (dq1 + dq2) + HG2 + H4 * dq2**2 - HG1) / (H1 - H5) + ((tau1 - B1 * dq1) - (tau2 - B2 * dq2)) / (H1 - H5)

        ddq2 = ((H4 * H1 - 2 * H4 * H5) * dq1 * dq2 - H4 * H5 * dq2**2 - H6 * H1 * dq1 + (H2 * H5 - H6 * H1) * dq2) / (H1 * H3 - H3 * H5) - (H1 * HG2 - H5 * HG1 - H1 * (tau2 - B2 * dq2) + H5 * (tau1 - B1 * dq1)) / (H1 * H3 - H3 * H5)

        # Euler integration
        dq1 = dq1 + ddq1 * dt
        dq2 = dq2 + ddq2 * dt
        q1 = q1 + dq1 * dt
        q2 = q2 + dq2 * dt

    return data

def TwoLinkRobotDataGenerator(q1_range, q2_range, dq1_range, dq2_range, numICs, T, dt, tau_max = 7.5):

    # Create test, validation, and training tensors with different percentages of numICs
    seed = 1
    test_tensor = generate_two_link_lab_data(q1_range, q2_range, dq1_range, dq2_range, round(0.2 * numICs), T, dt, seed, tau_max)

    seed = 2
    val_tensor = generate_two_link_lab_data(q1_range, q2_range, dq1_range, dq2_range, round(0.1 * numICs), T, dt, seed, tau_max)

    seed = 3
    train_tensor = generate_two_link_lab_data(q1_range, q2_range, dq1_range, dq2_range, round(0.7 * numICs), T, dt, seed, tau_max)

    return train_tensor, test_tensor, val_tensor


def TwoLinkRobotDataGenerator_mixed(q1_range, q2_range, dq1_range, dq2_range, numICs, T, dt, tau_max):

    # Create test, validation, and training tensors with different percentages of numICs
    seed = 1
    test_tensor = generate_two_link_lab_data(q1_range, q2_range, dq1_range, dq2_range, round(0.05 * numICs), T, dt, seed, tau_max)

    seed = 2
    test_tensor_unforced = generate_two_link_lab_data_unforced(q1_range, q2_range, dq1_range, dq2_range, round(0.05 * numICs), T, dt, seed, tau_max)

    seed = 3
    val_tensor = generate_two_link_lab_data(q1_range, q2_range, dq1_range, dq2_range, round(0.2 * numICs), T, dt, seed, tau_max)

    seed = 4
    train_tensor = generate_two_link_lab_data(q1_range, q2_range, dq1_range, dq2_range, round(0.35 * numICs), T, dt, seed, tau_max)

    seed = 5
    train_tensor_unforced = generate_two_link_lab_data_unforced(q1_range, q2_range, dq1_range, dq2_range, round(0.35 * numICs), T, dt, seed, tau_max)

    return train_tensor_unforced, train_tensor, test_tensor_unforced, test_tensor, val_tensor

def DataGenerator_mixed(x1range, x2range, numICs, mu, lam, T, dt):

    # Create test, validation, and training tensors with different percentages of numICs
    seed = 1
    test_tensor_unforced = generate_data_unforced(x1range, x2range, round(0.05 * numICs), mu, lam, T, dt, seed)

    seed = 2
    test_tensor_forced = generate_data(x1range, x2range, round(0.05 * numICs), mu, lam, T, dt, seed)

    seed = 3
    val_tensor = generate_data(x1range, x2range, round(0.1 * numICs), mu, lam, T, dt, seed)

    seed = 4
    train_tensor_unforced = generate_data_unforced(x1range, x2range, round(0.4 * numICs), mu, lam, T, dt, seed)

    seed = 5
    train_tensor_forced = generate_data(x1range, x2range, round(0.4 * numICs), mu, lam, T, dt, seed)

    return train_tensor_unforced, train_tensor_forced, test_tensor_unforced, test_tensor_forced, val_tensor
