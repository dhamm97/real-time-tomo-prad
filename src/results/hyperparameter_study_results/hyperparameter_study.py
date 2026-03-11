import numpy as np
import os
import sys
import scipy.sparse as sp
import time
import copy
import src.routines.tomo_fusion.functionals_definition as fct_def
import src.routines.tomo_fusion.bayesian_computations as bcomp
from pyxu_diffops.operator import AnisDiffusionOp


def run_hyperparameter_study(phantom_indices, saving_dir):
    # reconstruction and phantom shape
    Nz, Nr = 120, 41
    tcv_zmin, tcv_zmax = -0.75, 0.75
    tcv_rmin, tcv_rmax = 0.624, 1.1376
    dz, dr = (tcv_zmax - tcv_zmin) / Nz, (tcv_rmax - tcv_rmin) / Nr
    tcv_Zs = np.linspace(tcv_zmin, tcv_zmax, Nz + 1, endpoint=True)[:-1] + dz / 2
    tcv_Rs = np.linspace(tcv_rmin, tcv_rmax, Nr + 1, endpoint=True)[:-1] + dr / 2
    sampling = (dr+dz)/2
    # load geometry matrix and tcv mask
    geometry_matrix = np.load("../../tcv_geometry/geometry_matrix_NINO.npy")
    geometry_matrix = geometry_matrix.reshape(geometry_matrix.shape[0], -1)
    geometry_matrix = sp.csr_matrix(geometry_matrix)
    # compute geometry matrix scaling factor amd rescale it
    geometry_matrix_rescaling_factor = 1 / np.max(geometry_matrix)
    geometry_matrix_rescaled = copy.deepcopy(geometry_matrix) * geometry_matrix_rescaling_factor
    tcv_mask = np.load("../../tcv_geometry/tcv_mask_1_subpixels_NINO.npy")

    # load phantoms to be analyzed
    phantoms = np.load('phantoms/phantoms.npy')
    phantoms_psis = np.load('phantoms/phantoms_psis.npy')
    phantoms_trim_values = np.load('phantoms/phantoms_trim_values.npy')

    # compute tomographic datas and define error levels
    tomo_datas = geometry_matrix.reshape(geometry_matrix.shape[0], -1) @ (phantoms.reshape(phantoms.shape[0], -1).T)
    sigma0_level = 0.05 * np.mean(tomo_datas)
    sigma_signaldependent_level = 0.05

    # retain only phantoms to be analzyed
    phantoms = phantoms[phantom_indices, :, :]
    phantoms_psis = phantoms_psis[phantom_indices, :, :]
    phantoms_trim_values = phantoms_trim_values[phantom_indices, :]

    # hyperparameter grid search
    for i, idx in enumerate(phantom_indices):
        st = time.time()
        print("phantom ", idx)

        # fix seed
        np.random.seed(i)

        ground_truth = phantoms[i, :, :]
        psi = phantoms_psis[i, :, :]

        # determine x-point lcoation
        xpoint_z_idx_base_psi = 90
        trimming_vals = np.array(
            [phantoms_trim_values[i, 0], int(120 - phantoms_trim_values[i, 1]),
             phantoms_trim_values[i, 2], int(Nr - phantoms_trim_values[i, 3])]
        )
        z_xpoint_idx = int(Nz * (xpoint_z_idx_base_psi - trimming_vals[0]) / (
                trimming_vals[1] - trimming_vals[0]))
        # compute core mask
        rho_eff = copy.copy(psi)
        rescaling_factor = np.abs(np.min(rho_eff))
        rho_eff += rescaling_factor
        rho_eff /= rescaling_factor
        # core mask
        rho_eff_sqrt = np.sqrt(rho_eff)
        idxs_neg_psi = np.where(rho_eff_sqrt < 0.95)
        idxs_core = (idxs_neg_psi[0][idxs_neg_psi[0] < z_xpoint_idx],
                     idxs_neg_psi[1][idxs_neg_psi[0] < z_xpoint_idx])
        mask_core = np.zeros((Nz, Nr), dtype=bool)
        mask_core[idxs_core] = True

        # generate tomographic data
        tomo_data = geometry_matrix @ (ground_truth.flatten())
        noisy_tomo_data = tomo_data + sigma0_level * np.random.randn(120) + sigma_signaldependent_level * np.random.randn(120) * tomo_data
        noisy_tomo_data_rescaled = noisy_tomo_data * geometry_matrix_rescaling_factor

        # hyperparameter tuning
        reg_params = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
        anis_params = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
        sigma_errs = np.array([1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1])

        # normalize data in pre-processing phase
        max_scaling_factor = 1
        scaling_coeff = np.max(noisy_tomo_data_rescaled) / max_scaling_factor
        noisy_tomo_data_normalized = noisy_tomo_data_rescaled / scaling_coeff

        # number of hyperparameter combinations
        nb_grid_values = int(sigma_errs.size * anis_params.size * reg_params.size)

        # hyperparameters
        hyperparameters = np.zeros((nb_grid_values, 3))
        # maps
        prads_tot = np.zeros(nb_grid_values)
        prads_core = np.zeros(nb_grid_values)

        # coeffs for prad_tot, prad_core computation
        weight_vec_2piRs_tot = np.array(
            [2 * np.pi * tcv_Rs] * Nz).flatten() * dr * dz * tcv_mask.flatten()
        weight_vec_2piRs_core = np.array(
            [2 * np.pi * tcv_Rs] * Nz).flatten() * dr * dz * mask_core.flatten()

        # true prad_tot, prad_core
        prad_tot_true = np.dot(weight_vec_2piRs_tot, ground_truth.flatten())
        prad_core_true = np.dot(weight_vec_2piRs_core, ground_truth.flatten())

        # counter
        counter = 0

        for sigma_err in sigma_errs:
            # define data-fidelity functional
            f = fct_def._DataFidelityFunctional(dim_shape=(1, 120, 41), noisy_tomo_data=noisy_tomo_data_normalized,
                                                     sigma_err=sigma_err, geometry_matrix=sp.csr_matrix(geometry_matrix_rescaled))
            for reg_param in reg_params:
                for anis_param in anis_params:
                    hyperparameters[counter, :] = np.array([sigma_err, reg_param, anis_param])
                    # define regularization functional
                    g = AnisDiffusionOp(dim_shape=(1, 120, 41),
                                              alpha=anis_param,
                                              diff_method_struct_tens="fd",
                                              freezing_arr=psi,
                                              sampling=sampling,
                                              matrix_based_impl=True)
                    # compute the MAP
                    im_map = bcomp.compute_MAP(f, g, reg_param, with_pos_constraint=False,
                                               clipping_mask=tcv_mask, show_progress=False)
                    im_map *= scaling_coeff

                    prads_tot[counter] = np.dot(weight_vec_2piRs_tot, im_map.flatten())
                    prads_core[counter] = np.dot(weight_vec_2piRs_core, im_map.flatten())

                    counter += 1

        # save the computed quantities: noisy_tomo_data, hyperparameters used, core mask, MAP, prad tot and core
        data = {}
        data['hyperparameters'] = hyperparameters
        data['noisy_tomo_data'] = noisy_tomo_data
        #data["im_map"] = im_maps
        data["prads_tot"] = prads_tot
        data["prad_tot_true"] = prad_tot_true
        data['prads_core'] = prads_core
        data["prad_core_true"] = prad_core_true
        data["mask_core"] = mask_core
        data["time"] = (time.time() - st)
        # save to file the computed information
        np.save(saving_dir + "/phantom_" + str(idx) + ".npy", data)
    return


if __name__ == '__main__':

    argv = sys.argv
    if len(argv) == 1:
        phantom_indices = np.arange(0, 1000)
    elif len(argv) == 3:
        phantom_indices = np.arange(int(argv[1]), int(argv[2]))
        print("Running pipeline on phantoms {}-{}".format(int(argv[1]), int(argv[2])))
    else:
        raise ValueError("Number of passed arguments must be either 1 or 3")

    # define saving directory
    saving_dir = 'phantom_analysis_results'
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)

    # number of pixels in vertical and radial direction
    Nz, Nr = 120, 41

    run_hyperparameter_study(phantom_indices=phantom_indices, saving_dir=saving_dir)
