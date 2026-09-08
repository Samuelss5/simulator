import numpy as np



class UserEquipmentSectorization:

    @classmethod
    def compute(cls, num_elements, num_arrays, rng):

        # Randomic rotation of the user equipments
        Random_rotation = rng.uniform(0, 360, num_elements).reshape(-1,1)

        angular_spacing = 360 / num_arrays

        H_bs_possible = np.arange(0, 360, angular_spacing).reshape(1,-1)
        V_bs = (90)

        H_boresights = np.tile(H_bs_possible, (num_elements, 1)) + np.tile(Random_rotation, (1, num_arrays))
        H_boresights = H_boresights % 360

        V_boresights = np.ones((num_elements, num_arrays)) * V_bs

        return H_boresights, V_boresights




class UniformSectorization:

    @classmethod
    def compute(cls, num_elements, num_arrays, vert_tilt):

        
        angular_spacing = 360 / num_arrays

        H_bs_possible = np.arange(0, 360, angular_spacing).reshape(1,-1)
        V_bs = (90 + vert_tilt)

        H_boresights = np.tile(H_bs_possible, (num_elements, 1)) 
        V_boresights = np.ones((num_elements, num_arrays)) * V_bs

        return H_boresights, V_boresights


class PerfectAlignment:

    @classmethod
    def compute(cls, 
        AoAs_h,
        AoAs_v,
        v5
    ):
        
        num_rx, _, _, _ = AoAs_h.shape

        



class Sectorization:

    @classmethod
    def compute(cls, num_elements, num_arrays_per_element, vert_tilt):

        boresights_spacing = 360 / num_arrays_per_element
        
        Available_horz_boresights = np.arange(0, 360, boresights_spacing).reshape(-1,1)
        Available_horz_boresights = np.pi * Available_horz_boresights / 180

        vert = (90 + vert_tilt) * np.pi / 180

        Boresights_horz = np.tile(Available_horz_boresights, (num_elements, 1))

        Boresights_vert = np.ones((num_arrays_per_element * num_elements, 1)) * vert 
        
        return Boresights_horz, Boresights_vert

class Station_sectorization_3gpp:

    def compute(num_arrays_per_station, num_stations, downtilt):


        boresights_spacing = 360 / num_arrays_per_station

        Available_horz_boresights = np.arange(0, 360, boresights_spacing).reshape(-1,1)
        Available_horz_boresights = np.pi * Available_horz_boresights / 180

        vert = (90 - downtilt) * np.pi / 180

        Boresights_horz = np.tile(Available_horz_boresights, (num_stations, 1))

        Boresights_vert = np.ones((num_arrays_per_station * num_stations, 1)) * vert

        return Boresights_horz, Boresights_vert

