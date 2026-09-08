import numpy as np

class RicianChannel:

    def generate_queued_multiple_channels(queue, Gain_Coefficients, K_coefficients, R_rx, R_tx, AoAs_h_rx, AoAs_v_rx, AoAs_h_tx, AoAs_v_tx,
                                    N_h_rx, N_v_rx, N_h_tx, N_v_tx, rng: object):

    
        num_rx, num_tx, num_rx_panels, num_tx_panels = Gain_Coefficients.shape

        N_h_rx_idxs = np.arange(0, N_h_rx)
        N_v_rx_idxs = np.arange(0, N_v_rx)

        N_h_tx_idxs = np.arange(0, N_h_tx)
        N_v_tx_idxs = np.arange(0, N_v_tx)


        channel_Coefficients = np.zeros(Gain_Coefficients.shape, dtype = object)

        for rx in range(num_rx):
            for tx in range(num_tx):
                for rx_p in range(num_rx_panels):
                    for tx_p in range(num_tx_panels):

                        aoa_h_rx = AoAs_h_rx[rx, tx, rx_p, tx_p]
                        aoa_v_rx = AoAs_v_rx[rx, tx, rx_p, tx_p]

                        aoa_h_tx = AoAs_h_tx[tx, rx, tx_p, rx_p]
                        aoa_v_tx = AoAs_v_tx[tx, rx, tx_p, rx_p]

                        # Receiver array factors (horizontal, vertical )
                        Af_h_rx_k = np.exp(1j * np.pi * N_h_rx_idxs * np.sin(aoa_h_rx) * np.cos(aoa_v_rx))[:, np.newaxis]
                        Af_v_rx_k = np.exp(1j * np.pi * N_v_rx_idxs * np.cos(aoa_v_rx))[:, np.newaxis]
                        AF_rx = np.kron(Af_h_rx_k, Af_v_rx_k)

                        Af_h_tx_j = np.exp(1j * np.pi * N_h_tx_idxs * np.sin(aoa_h_tx) * np.cos(aoa_v_tx))[:, np.newaxis]
                        Af_v_tx_j = np.exp(1j * np.pi * N_v_tx_idxs * np.cos(aoa_v_tx))[:, np.newaxis] 
                        AF_tx = np.kron(Af_h_tx_j, Af_v_tx_j)

                        # Los component
                        AF_matrix = AF_rx @ AF_tx.T

                        A = rng.normal(0,1, size=AF_matrix.shape) + 1j * rng.normal(0,1, size=AF_matrix.shape)

                        H_nlos = np.sqrt(R_rx[rx, tx, rx_p, tx_p]) @ A @ np.sqrt(R_tx[tx, rx, tx_p, rx_p])

                        K = K_coefficients[rx, tx]

                        los_coefficient  = np.sqrt(K / (K + 1))
                        nlos_coefficient = np.sqrt(1 / (K + 1))

                        channel_Coefficients[rx, tx, rx_p, tx_p] = np.sqrt(Gain_Coefficients[rx, tx, rx_p, tx_p] / 2) * (los_coefficient * AF_matrix + nlos_coefficient * H_nlos)


        queue.put(channel_Coefficients)


    def generate_multiple_channels(gain_Coefficients, K_Coefficients, rx_R_Matrixes, tx_R_Matrixes, 
                                   rx_relative_AoAs_h, rx_relative_AoAs_v, 
                                   tx_relative_AoAs_h, tx_relative_AoAs_v,
                                    N_h_rx, N_v_rx, N_h_tx, N_v_tx, rng: object):

        num_rx, num_tx, num_rx_panels, num_tx_panels = gain_Coefficients.shape

    
        rx_N_h_idxs = np.arange(0, N_h_rx)
        rx_N_v_idxs = np.arange(0, N_v_rx)

        tx_N_h_idxs = np.arange(0, N_h_tx)
        tx_N_v_idxs = np.arange(0, N_v_tx)

        N_rx = N_h_rx * N_v_rx
        N_tx = N_h_tx * N_v_tx

        channel_Coefficients = np.zeros((num_rx, num_tx, num_rx_panels, num_tx_panels, N_rx, N_tx), dtype=np.complex128)

        for rx in range(num_rx):
            for tx in range(num_tx):
                for rx_panel in range(num_rx_panels):
                    for tx_panel in range(num_tx_panels):

                        rx_aoa_h = rx_relative_AoAs_h[rx, tx, rx_panel, tx_panel]
                        rx_aoa_v = rx_relative_AoAs_v[rx, tx, rx_panel, tx_panel]

                        tx_aoa_h = tx_relative_AoAs_h[tx, rx, tx_panel, rx_panel]
                        tx_aoa_v = tx_relative_AoAs_h[tx, rx, tx_panel, rx_panel]

                        # Receiver array factors (horizontal, vertical)
                        rx_Af_h = np.exp(1j * np.pi * rx_N_h_idxs * np.sin(rx_aoa_h) * np.cos(rx_aoa_v))[:, np.newaxis]
                        rx_Af_v = np.exp(1j * np.pi * rx_N_v_idxs * np.cos(rx_aoa_v))[:, np.newaxis]
                        AF_rx = np.kron(rx_Af_h, rx_Af_v)

                        # Transmitter array factors (horizontal, vertical)
                        tx_Af_h = np.exp(1j * np.pi * tx_N_h_idxs * np.sin(tx_aoa_h) * np.cos(tx_aoa_v))[:, np.newaxis]
                        tx_Af_v = np.exp(1j * np.pi * tx_N_v_idxs * np.cos(tx_aoa_v))[:, np.newaxis] 
                        AF_tx = np.kron(tx_Af_h, tx_Af_v)

                        # Los component
                        steeringMatrix = AF_rx @ AF_tx.T

                        A = rng.normal(0,1, size=steeringMatrix.shape) + 1j * rng.normal(0,1, size=steeringMatrix.shape)

                        H_nlos = np.sqrt(rx_R_Matrixes[rx, tx, rx_panel, tx_panel]) @ A @ np.sqrt(tx_R_Matrixes[tx, rx, tx_panel, rx_panel])

                        K = K_Coefficients[rx, tx]

                        los_coefficient  = np.sqrt(K / (K + 1))
                        nlos_coefficient = np.sqrt(1 / (K + 1))

                        channel_Coefficients[rx, tx, rx_panel, tx_panel] = np.sqrt(gain_Coefficients[rx, tx, rx_panel, tx_panel] / 2) * (los_coefficient * steeringMatrix 
                                                                                                                     + nlos_coefficient * H_nlos)

        return channel_Coefficients



