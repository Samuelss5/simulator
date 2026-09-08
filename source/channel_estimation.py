import numpy as np

class Centralized_MMSE_estimation:

    @classmethod
    def compute(cls, channel_Coefficients, R_matrixes, panels_idxs, scheduled_ues_vec,
                    tau_p: int, ul_max_power: float, noise_var: float):

        

        # K = number of user equipments (UEs)
        # S = number of access points (APs)
        # _ = number of panels/arrays per UE
        # A = number of panels/arrays per AP
        K, S, _, A, Nt, Ns = channel_Coefficients.shape


        # Nt = number of antennas at each UE panel/array
        # Ns = number of antennas at each AP panel/array
        #Nt, Ns = channel_Coefficients[0,0,0,0].shape

        # L   = number of APs arrays
        # LNs = total number of antennas
        L = S * A
        LNs = L * Ns

        H_dense = np.array(channel_Coefficients.tolist())

        # The matrix dimension is (K,S,A,Nt,Ns)
        H_sel = H_dense[np.arange(K), :, panels_idxs, ...]

        # The matrix dimension must be (S,A,K,Ns,Nt)
        H_trans = H_sel.transpose(1,2,0,4,3)

        H_ul = H_trans.reshape(L, K, Ns, Nt)


        Rs_dense = np.array(R_matrixes.tolist())

        # (K, L, A, Ns, Ns)
        Rs_sel = Rs_dense[:, np.arange(K), :, panels_idxs].swapaxes(0,1)

        Rs_trans = Rs_sel.swapaxes(1,2)

        Rs_ul = Rs_trans.reshape(L, K, Ns,Ns)




        # OBS: It is important to note that we consider that each UE terminal is equipped with a single antenna

        # Power allocation -> the non scheduled UEs will have their powers set to zero, while the scheduled ones will transmit with max power

        transmit_powers_vec = np.zeros(K, dtype=float)
        transmit_powers_vec[scheduled_ues_vec] = ul_max_power


        # Pre-compute all concatenated channels
        H_concatenated = np.zeros((K, LNs, Nt), dtype=np.complex128)

        for ue_j in range(K):
            if transmit_powers_vec[ue_j] > 10**-6:  # Only compute for UEs with non-negligible power

                H_concatenated[ue_j] = np.concatenate(
                    H_ul[:, ue_j, :, :], axis=0
                )
            else:
                pass



        # Identify
        pilot_allocation_vec = np.full(K, None, dtype = object)

        num_scheduled = len(scheduled_ues_vec)

        if num_scheduled > 0:
            pilot_allocation_vec[scheduled_ues_vec] = np.arange(num_scheduled) % tau_p



        H_estimated = np.zeros(H_ul.shape, dtype = np.complex128)

        C_error_matrixes = np.zeros((K, L, Ns, Ns), dtype = np.complex128)

        for p in range(tau_p):
            
            # UEs sharing the pilot sequence p
            pilot_p_ues_vec = np.where(pilot_allocation_vec == p)[0]

            
            for l in range(L):

                # Pilot signal received by the l-th Station
                Y_l = np.sqrt(ul_max_power * tau_p) * np.sum(H_ul[l, pilot_p_ues_vec], axis = 0)

                Noise_l = np.random.normal(size=Y_l.shape) + 1j * np.random.normal(size=Y_l.shape)

                Y_l += Noise_l * np.sqrt(0.5) * noise_var
                
                Psi_matrix = np.sum(Rs_ul[l, pilot_p_ues_vec], axis = 0) * tau_p * ul_max_power + np.eye(Ns) * noise_var

                for k in pilot_p_ues_vec:

                    R_kl = Rs_ul[l, k]

                    R_kl_Psi = R_kl * np.linalg.inv(Psi_matrix)


                    C_kl = R_kl - ul_max_power * tau_p * (R_kl_Psi @ R_kl)

                    C_error_matrixes[k,l] = C_kl

                    # Estimated channel
                    H_kl = np.sqrt(ul_max_power * tau_p) * (R_kl_Psi @ Y_l)

                    H_estimated[l,k] = H_kl 

        return H_estimated, C_error_matrixes

    


                    


    
            
        



