from get_stiffness import get_stiffness
from get_classical_damping import get_classical_damping
from get_continuous_state_space import get_continuous_state_space
from get_response_state_space import get_response_state_space


def compute_response_SPs(parameter_SP, M_global, damping, DOF, B, output_type, step, a, t):
    K_global = get_stiffness(parameter_SP, DOF)
    C_global, _, _, _ = get_classical_damping(K_global, M_global, damping, "no")
    Ac, Bc, Cc, Dc = get_continuous_state_space(K_global, M_global, C_global, B, output_type)
    response, _, _, _, _, _ = get_response_state_space(Ac, Bc, Cc, Dc, a[0: step], t)
    return response