import numpy as np

from source.geometry.coordinates import calculate_2d_distances


class IndividualEstimated_INR:
    
    # @classmethod
    # def perform(cls, Gains, term_sel_panels_vec, max_ul_power, noise_variance):


    @classmethod
    def perform(cls, context):

        terminal_max_power = context.terminal_max_power

        ls_gains = context.inter_network_lsg

        terminal_panels = context.terminal_panels
 
        pn_noise_variance = context.pn_noise_variance


        # OBS: We consider that there is a single receiver that is being affected by the secondary terminals interference
        # OBS: We consider that this unique receiver is equipped with a single panel

        # OBS: We consider that this scheduling is performed after panel selection

        _, K, _, _ = ls_gains.shape

        estimated_INRs_Matrix = 10 * np.log10(terminal_max_power * ls_gains[:, np.arange(K), 0, terminal_panels] / pn_noise_variance)

        #print('estimated_INRs_matrix: ', estimated_INRs_Matrix)

        scheduled_terminals= np.where(estimated_INRs_Matrix[0] <= context.scheduling_threshold)[0]

        return scheduled_terminals


class CumulativeIndividualEstimated_INR:

    @classmethod
    def perform(cls, context):

        terminal_max_power = context.terminal_max_power
        ls_gains = context.inter_network_lsg
        terminal_panels = context.terminal_panels
        pn_noise_variance = context.pn_noise_variance


        _, K, _, _ = ls_gains.shape

        estimated_INRs = (terminal_max_power * ls_gains[:, np.arange(K), 0, terminal_panels] / pn_noise_variance)[0]

        ordered_estimated_INRs = np.sort(estimated_INRs)
        ordered_indexes = np.argsort(estimated_INRs)

        # Summing the ordered INRs
        cumulative_estimated_INRs = np.cumsum(ordered_estimated_INRs)

        selected_indexes = np.where(10*np.log10(cumulative_estimated_INRs) <= context.scheduling_threshold)[0]
        scheduled_terminals = ordered_indexes[selected_indexes]

        return np.sort(scheduled_terminals)



# ____________________________________________
# APs scheduling techniques
# ____________________________________________

class StationsIndivEstINR:

    @classmethod
    def perform(cls, context):

        lsg_coeffs = context.inter_network_lsg

        L = lsg_coeffs.shape[1] * lsg_coeffs.shape[3] 
        gain_vec = lsg_coeffs[0].reshape(1, L)

        station_max_power = context.station_max_power

        pn_noise_variance = context.pn_noise_variance


        estimated_inrs = 10*np.log10( station_max_power * gain_vec / pn_noise_variance)

        scheduled_stations = np.where(estimated_inrs[0] <= context.scheduling_threshold)[0]

        return scheduled_stations
