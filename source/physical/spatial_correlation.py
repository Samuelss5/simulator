import numpy as np

from scipy.integrate import dblquad # type: ignore
from scipy.linalg import toeplitz # type: ignore

import numba # type: ignore

class GaussianCorrelation:

    @staticmethod
    @numba.jit
    def real_vertical(d_phi: float, d_theta: float, asd_phi: float, asd_theta: float, d: int):

        phase = np.pi * d * np.cos(d_theta)

        pdf_phi = np.exp( (-d_phi * d_phi) / ( np.sqrt(2 * np.pi) * asd_phi) )
        pdf_theta = np.exp( (-d_theta * d_theta) / (np.sqrt(2 * np.pi) * asd_theta) )

        intg = np.exp(1j * phase) * pdf_phi * pdf_theta

        return intg.real

    @staticmethod
    def imag_vertical(d_phi: float, d_theta: float, asd_phi: float, asd_theta: float, d: int):

        phase = np.pi * d * np.cos(d_theta)

        pdf_phi = np.exp( (-d_phi * d_phi) / ( np.sqrt(2 * np.pi) * asd_phi) )
        pdf_theta = np.exp( (-d_theta * d_theta) / (np.sqrt(2 * np.pi) * asd_theta) )

        intg = np.exp(1j * phase) * pdf_phi * pdf_theta

        return intg.imag


    @staticmethod
    def real_horizontal(d_phi, d_theta, asd_phi: float, asd_theta: float, d: int):

        phase = np.pi * d * np.sin(d_phi) * np.cos(d_theta)

        pdf_phi = np.exp( (-d_phi * d_phi) / ( np.sqrt(2 * np.pi) * asd_phi) )
        pdf_theta = np.exp( (-d_theta * d_theta) / (np.sqrt(2 * np.pi) * asd_theta) )

        intg = np.exp(1j * phase) * pdf_phi * pdf_theta

        return intg.real


    @staticmethod
    def imag_horizontal(d_phi, d_theta, asd_phi: float, asd_theta: float, d: int):

        phase = np.pi * d * np.sin(d_phi) * np.cos(d_theta)

        pdf_phi = np.exp( (-d_phi * d_phi) / ( np.sqrt(2 * np.pi) * asd_phi) )
        pdf_theta = np.exp( (-d_theta * d_theta) / (np.sqrt(2 * np.pi) * asd_theta) )

        intg = np.exp(1j * phase) * pdf_phi * pdf_theta

        return intg.imag

    def compute(relative_AoAs_h: np.ndarray, relative_AoAs_v: np.ndarray, N_h: int, N_v: int):

        """ Compute the spatial correlation matrixes from the receivers perspective """

        # Number of receivers and transmitters
        num_rx, num_tx, num_rx_panels, num_tx_panels = relative_AoAs_h.shape

        # Reserving some space to store the correlation matrixes
        R_matrixes = np.zeros((num_rx, num_tx, num_rx_panels, num_tx_panels), dtype=object)

        # Angular spread
        asd_h = 15 # degrees
        asd_v = 15 # degrees

        asd_h = asd_h * np.pi / 180
        asd_v = asd_v * np.pi / 180

        N_hor_idxs = np.arange(0, N_h)
        N_ver_idxs = np.arange(0, N_v)

        for rx_k in range(num_rx):
            for tx_l in range(num_tx):
                for rx_k_panel in range(num_rx_panels):
                    for tx_j_panel in range(num_tx_panels):

                        d_phi_inf = relative_AoAs_h[rx_k, tx_l, rx_k_panel, tx_j_panel] - asd_h
                        d_phi_sup = relative_AoAs_h[rx_k, tx_l, rx_k_panel, tx_j_panel] + asd_h

                        d_theta_inf = relative_AoAs_v[rx_k, tx_l, rx_k_panel, tx_j_panel] - asd_v
                        d_theta_sup = relative_AoAs_v[rx_k, tx_l, rx_k_panel, tx_j_panel] + asd_v

                        row_h = np.zeros(N_h, dtype=np.complex128)
                        row_v = np.zeros(N_v, dtype=np.complex128)

                        for n_h in range(N_h):
                            arguments = (asd_h, asd_v, n_h)

                            Rxx = dblquad(Gaussian_correlation.real_horizontal, d_phi_inf, d_phi_sup, d_theta_inf, d_theta_sup, args = arguments)
                            Rxy = dblquad(Gaussian_correlation.imag_horizontal, d_phi_inf, d_phi_sup, d_theta_inf, d_theta_sup, args = arguments)

                            row_h[n_h] = Rxx[0] + 1j * Rxy[0]

                        for n_v in range(N_v):
                            arguments = (asd_h, asd_v, n_v)

                            Rxx = dblquad(Gaussian_correlation.real_vertical, d_phi_inf, d_phi_sup, d_theta_inf, d_theta_sup, args = arguments)
                            Rxy = dblquad(Gaussian_correlation.imag_vertical, d_phi_inf, d_phi_sup, d_theta_inf, d_theta_sup, args = arguments)

                            row_v[n_v] = Rxx[0] + 1j * Rxy[0]

                        R_h = toeplitz(row_h)
                        R_v = toeplitz(row_v)

                        R_h = R_h * (N_h / np.trace(R_h) )
                        R_v = R_v * (N_v / np.trace(R_v) )


                        R_matrixes[rx_k, tx_l, rx_k_panel, tx_j_panel] = np.kron(R_h, R_v)


        return R_matrixes

    @classmethod
    def compute_fast_integrals(cls, r_aoas_h: np.ndarray, r_aoas_v: np.ndarray, N_h: int, N_v: int):

        # Angular spread
        asd_h = 15 # degrees
        asd_v = 15 # degrees

        asd_h = asd_h * np.pi / 180
        asd_v = asd_v * np.pi / 180

        aoas_h_inf = r_aoas_h - 20 * asd_h
        aoas_h_sup = r_aoas_h + 20 * asd_h

        aoas_v_inf = r_aoas_v - 20 * asd_v
        aoas_v_sup = r_aoas_v + 20 * asd_v

        N_h_idxs = np.arange(0, N_h)
        N_v_idxs = np.arange(0, N_v)

        num_rx, num_tx, num_rx_panels, num_tx_panels = r_aoas_h.shape

        R_matrices = np.zeros((num_rx, num_tx, num_rx_panels, num_tx_panels), dtype=np.ndarray)

        for rx in range(num_rx):
            for tx in range(num_tx):
                for rx_p in range(num_rx_panels):
                    for tx_p in range(num_tx_panels):

                        h_samples = np.linspace(aoas_h_inf[rx, tx, rx_p, tx_p], aoas_h_sup[rx, tx, rx_p, tx_p], 300)
                        v_samples = np.linspace(aoas_v_inf[rx, tx, rx_p, tx_p], aoas_v_sup[rx, tx, rx_p, tx_p], 300)

                        H, V = np.meshgrid(h_samples, v_samples)

                        pdf =  np.exp(-H**2 / (2*asd_h**2)) * np.exp(-V**2 / (2*asd_v**2))

                        res_h = []
                        for n_h in N_h_idxs:

                            phase_h = np.exp(1j * np.pi * n_h * np.sin(H) * np.cos(V))

                            integrand_grid = phase_h * pdf

                            res = np.trapezoid(np.trapezoid(integrand_grid, v_samples, axis=0), h_samples)
                            res_h.append(res)

                        res_v = []
                        for n_v in N_v_idxs:

                            phase_v = np.exp(1j * np.pi * n_v * np.cos(V))

                            integrand_grid = phase_v * pdf

                            res = np.trapezoid(np.trapezoid(integrand_grid, v_samples, axis=0), h_samples)
                            res_v.append(res)


                        R_h = toeplitz(np.array(res_h))
                        R_v = toeplitz(np.array(res_v))

                        R_h = R_h * (N_h / np.trace(R_h) )
                        R_v = R_v * (N_v / np.trace(R_v) )

                        R_matrices[rx, tx, rx_p, tx_p] = np.kron(R_h, R_v)

        return R_matrices


    @classmethod
    def compute_fast_integrals_for_queue(cls, queue, r_aoas_h: np.ndarray, r_aoas_v: np.ndarray, N_h: int, N_v: int):

        # Angular spread
        asd_h = 15 # degrees
        asd_v = 15 # degrees

        asd_h = asd_h * np.pi / 180
        asd_v = asd_v * np.pi / 180

        aoas_h_inf = r_aoas_h - 20 * asd_h
        aoas_h_sup = r_aoas_h + 20 * asd_h

        aoas_v_inf = r_aoas_v - 20 * asd_v
        aoas_v_sup = r_aoas_v + 20 * asd_v

        N_h_idxs = np.arange(0, N_h)
        N_v_idxs = np.arange(0, N_v)

        num_rx, num_tx, num_rx_panels, num_tx_panels = r_aoas_h.shape

        R_matrices = np.zeros((num_rx, num_tx, num_rx_panels, num_tx_panels), dtype=np.ndarray)

        for rx in range(num_rx):
            for tx in range(num_tx):
                for rx_p in range(num_rx_panels):
                    for tx_p in range(num_tx_panels):

                        h_samples = np.linspace(aoas_h_inf[rx, tx, rx_p, tx_p], aoas_h_sup[rx, tx, rx_p, tx_p], 300)
                        v_samples = np.linspace(aoas_v_inf[rx, tx, rx_p, tx_p], aoas_v_sup[rx, tx, rx_p, tx_p], 300)

                        H, V = np.meshgrid(h_samples, v_samples)

                        pdf =  np.exp(-H**2 / (2*asd_h**2)) * np.exp(-V**2 / (2*asd_v**2))

                        res_h = []
                        for n_h in N_h_idxs:

                            phase_h = np.exp(1j * np.pi * n_h * np.sin(H) * np.cos(V))

                            integrand_grid = phase_h * pdf

                            res = np.trapezoid(np.trapezoid(integrand_grid, v_samples, axis=0), h_samples)
                            res_h.append(res)

                        res_v = []
                        for n_v in N_v_idxs:

                            phase_v = np.exp(1j * np.pi * n_v * np.cos(V))

                            integrand_grid = phase_v * pdf

                            res = np.trapezoid(np.trapezoid(integrand_grid, v_samples, axis=0), h_samples)
                            res_v.append(res)


                        R_h = toeplitz(np.array(res_h))
                        R_v = toeplitz(np.array(res_v))

                        R_h = R_h * (N_h / np.trace(R_h) )
                        R_v = R_v * (N_v / np.trace(R_v) )

                        R_matrices[rx, tx, rx_p, tx_p] = np.kron(R_h, R_v)

        queue.put(R_matrices)












### === Gaussian Channel Spatial Correlation === ###

def gauss_func_real(delta, epsilon, asd_phi: float, asd_theta: float, d: int):

    phase = np.exp(1j * np.pi * d * np.sin(delta) * np.cos(epsilon))

    pdf_phi = np.exp( (-delta * delta) / ( np.sqrt(2 * np.pi) * asd_phi) )

    pdf_theta = np.exp( (-epsilon * epsilon) / (np.sqrt(2 * np.pi) * asd_theta) )

    intg = phase * pdf_phi * pdf_theta

    return intg.real



def gauss_func_imag(delta, epsilon, asd_phi: float, asd_theta: float, d: int):
    """  """

    phase = np.exp(1j * np.pi * d * np.sin(delta) * np.cos(epsilon))

    pdf_phi = np.exp( (-delta * delta) / ( np.sqrt(2 * np.pi) * asd_phi) )

    pdf_theta = np.exp( (-epsilon * epsilon) / (np.sqrt(2 * np.pi) * asd_theta) )

    intg = phase * pdf_phi * pdf_theta

    return intg.imag




def compute_gaussian_spatial_correlation(AoAs_horizontal_rx: np.ndarray, AoAs_vertical_rx: np.ndarray, N_rx: int):

    """ Compute the spatial correlation matrixes from the receivers perspective """

    # Number of receivers and transmitters
    num_rx, num_tx = AoAs_horizontal_rx.shape

    # Reserving some space to store the correlation matrixes
    Correlation_matrixes = np.zeros((num_rx, num_tx, N_rx, N_rx), dtype = np.complex128)

    # Angular spread
    asd_h = 15 # degrees
    asd_v = 15 # degrees

    asd_h = asd_h * np.pi / 180
    asd_v = asd_v * np.pi / 180

    for rx_k in range(AoAs_horizontal_rx.shape[0]):
        for tx_l in range(AoAs_horizontal_rx.shape[1]):

            delta_inf = AoAs_horizontal_rx[rx_k, tx_l] - asd_h
            delta_sup = AoAs_horizontal_rx[rx_k, tx_l] + asd_h

            epsilon_inf = AoAs_vertical_rx[rx_k, tx_l] - asd_v
            epsilon_sup = AoAs_vertical_rx[rx_k, tx_l] + asd_v

            row = np.zeros(N_rx, dtype = complex)

            for n in range(N_rx):

                arguments = (asd_h, asd_v, n)

                Rxx = dblquad(gauss_func_real, delta_inf, delta_sup, epsilon_inf, epsilon_sup, args = arguments)

                Rxy = dblquad(gauss_func_imag, delta_inf, delta_sup, epsilon_inf, epsilon_sup, args = arguments)

                row[n] = Rxx[0] + 1j * Rxy[0]

            R = toeplitz(row)

            Correlation_matrixes[rx_k, tx_l] = R

    return Correlation_matrixes







