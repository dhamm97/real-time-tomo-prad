import numpy as np
import os
import sys
import time
from src.routines.rt_roi_prad import RtPradROI

def run_campaign_study(shot_preprocessing_information,  chunk, saving_dir, reverse_field_shots, usn_shots):
    # extract indices of shots to be processed
    chunk_size = int(5)
    if (chunk + 1) * chunk_size < shot_preprocessing_information["shots_to_be_analyzed"].size:
        shots_to_be_processed = shot_preprocessing_information["shots_to_be_analyzed"][chunk * chunk_size:(chunk + 1) * chunk_size].astype(int)
    else:
        shots_to_be_processed = shot_preprocessing_information["shots_to_be_analyzed"][chunk * chunk_size:].astype(int)
    print("Processing shots {}-{}".format(shots_to_be_processed[0], shots_to_be_processed[-1]))
    # iterate over the shots to be processed
    for shot_idx, shot in enumerate(shots_to_be_processed):
        print("Processing shot ", int(shot))
        # determine if shot is USN or not
        if shot in usn_shots:
            is_usn = True
        else:
            is_usn = False
        # initialize rt_prad module
        shot_idx_global = int(chunk * chunk_size + shot_idx)
        good_channels = shot_preprocessing_information["shot_good_channels"][shot_idx_global]
        st = time.time()
        rt_prad = RtPradROI(shot_nb=int(shot), bolo_good_channels=good_channels, downsample=False, is_usn=is_usn)
        if shot in reverse_field_shots:
            rt_prad.fbte_eqs = - rt_prad.fbte_eqs
            rt_prad.liuqe_eqs = - rt_prad.liuqe_eqs
        # compute the FBT-based masks for total and core radiated power estimation
        mask_types = ["total", "core", "divertor", "main_chamber"]
        rt_prad.compute_roi_fbte_based_masks(time_range=None, mask_types=mask_types)
        # compute the FBT-based coefficients
        _, _ = rt_prad.compute_fbte_based_coefficients(sigma_err=2.5e-2, reg_param=1e1,
                                                                        anis_param=1e-2, mask_type="fbte",
                                                                        bc_type="noflux")
        coeff_def_time = time.time() - st
        # estimate radiated power
        prads_roi, stds_prad_roi = rt_prad.estimate_prad_roi()
        # compute the tomographic baseline
        (inversion_times, _,
         prads_roi_inversions, _, _) = rt_prad.compute_baseline_tomographic_inversion(
            decimation_factor=1,
            sigma_err=2.5e-2, reg_param=1e1,
            anis_param=1e-2, with_positivity_constraint=False,
            mask_type="liuqe")
        total_time = time.time() - st

        # if extra bad channels compared to previous shot, try estimation using the previous shot good channels
        if shot_preprocessing_information["shot_prev_bad_channels_differ"][shot_idx_global]:
            good_channels_prev = shot_preprocessing_information["shot_prev_good_channels"][shot_idx_global]
            rt_prad_prev = RtPradROI(shot_nb=int(shot), bolo_good_channels=good_channels_prev, downsample=False, is_usn=is_usn)
            if shot in reverse_field_shots:
                rt_prad_prev.fbte_eqs = - rt_prad_prev.fbte_eqs
                rt_prad_prev.liuqe_eqs = - rt_prad_prev.liuqe_eqs
            # compute the FBT-based masks for total and core radiated power estimation
            rt_prad_prev.compute_roi_fbte_based_masks(time_range=None, mask_types=mask_types)
            # compute the FBT-based coefficients
            _, _ = rt_prad_prev.compute_fbte_based_coefficients(sigma_err=2.5e-2, reg_param=1e1,
                                                                            anis_param=1e-2, mask_type="fbte",
                                                                            bc_type="noflux")
            # estimate radiated power
            prads_roi_prev, stds_prad_roi_prev = rt_prad_prev.estimate_prad_roi()

        # repeat estimates always excluding, on top of the bad channels from previous shot, also all the channels accounting for most single-channel failure
        good_channels_prev_stable = shot_preprocessing_information["shot_prev_stable_good_channels"][shot_idx_global]
        rt_prad_prev_stable = RtPradROI(shot_nb=int(shot), bolo_good_channels=good_channels_prev_stable, downsample=False, is_usn=is_usn)
        if shot in reverse_field_shots:
            rt_prad_prev_stable.fbte_eqs = - rt_prad_prev_stable.fbte_eqs
            rt_prad_prev_stable.liuqe_eqs = - rt_prad_prev_stable.liuqe_eqs
        # compute the FBT-based masks for total and core radiated power estimation
        rt_prad_prev_stable.compute_roi_fbte_based_masks(time_range=None, mask_types=mask_types)
        # compute the FBT-based coefficients
        _, _ = rt_prad_prev_stable.compute_fbte_based_coefficients(sigma_err=2.5e-2, reg_param=1e1,
                                                            anis_param=1e-2, mask_type="fbte",
                                                            bc_type="noflux")
        # estimate radiated power
        prads_roi_prev_stable, stds_prad_roi_prev_stable = rt_prad_prev_stable.estimate_prad_roi()


        # basic appraoch: estimate radiated power excluding only known bad channels, no history considered [70,102,106,118]
        good_channels_basic = np.arange(0,120)+1
        bad_channels_basic = np.array([70, 102, 106, 118], dtype=int)
        for bad_ch_ in bad_channels_basic:
            good_channels_basic = np.delete(good_channels_basic, np.where(good_channels_basic == bad_ch_)[0])
        rt_prad_basic = RtPradROI(shot_nb=int(shot), bolo_good_channels=good_channels_basic, downsample=False, is_usn=is_usn)
        if shot in reverse_field_shots:
            rt_prad_basic.fbte_eqs = - rt_prad_basic.fbte_eqs
            rt_prad_basic.liuqe_eqs = - rt_prad_basic.liuqe_eqs
        # compute the FBT-based masks for total and core radiated power estimation
        rt_prad_basic.compute_roi_fbte_based_masks(time_range=None, mask_types=mask_types)
        # compute the FBT-based coefficients
        _, _ = rt_prad_basic.compute_fbte_based_coefficients(sigma_err=2.5e-2, reg_param=1e1,
                                                            anis_param=1e-2, mask_type="fbte",
                                                            bc_type="noflux")
        # estimate radiated power
        prads_roi_basic, stds_prad_roi_basic = rt_prad_basic.estimate_prad_roi()


        # save quantities
        data = {}
        data["shot_idx_global"] = shot_idx_global
        data["is_usn"] = is_usn
        data["coeff_def_time"] = coeff_def_time
        data["nb_fbt_equilibria"] = rt_prad.fbte_masks.shape[1]
        data["total_time"] = total_time
        data["prads_roi"] = prads_roi
        data["stds_prad_roi"] = stds_prad_roi
        data["inversion_times"] = inversion_times
        data["prads_roi_inversions"] = prads_roi_inversions
        if shot_preprocessing_information["shot_prev_bad_channels_differ"][shot_idx_global]:
            data["shot_prev_bad_channels_differ"] = True
            data["prads_roi_prev"] = prads_roi_prev
            data["stds_prad_roi_prev"] = stds_prad_roi_prev
        else:
            data["shot_prev_bad_channels_differ"] = False
        data["prads_roi_prev_stable"] = prads_roi_prev_stable
        data["stds_roi_prev_stable"] = stds_prad_roi_prev_stable
        data["good_channels_prev_stable"] = good_channels_prev_stable
        data["prads_roi_basic"] = prads_roi_basic
        data["stds_roi_basic"] = stds_prad_roi_basic
        data["good_channels_basic"] = good_channels_basic

        # save to file the computed information
        np.save(saving_dir + "/shot_" + str(shot) + ".npy", data)

    return


if __name__ == '__main__':
    # load shot pre-processing information
    shot_information = np.load('shot_infos.npy', allow_pickle=True).item()

    # LSN configuration
    lsn_idxs = np.array([85184, 85185, 85269, 85270, 86554, 86572, 86605, 86735, 86850, 87081])
    # USN configuration
    usn_idxs = np.array([85432, 85434, 86210, 86462, 86585, 86591, 86838, 86839, 87063, 87076])
    # NT configuration
    nt_idxs = np.array([84762, 84764, 85138, 85997, 86063, 86088, 86089, 86289, 86436, 86744])
    # XPT configuration
    xpt_idxs = np.array([85820, 85199, 85350, 85353, 85357, 85815, 85816, 85818, 85819, 87080])
    # LL configuration
    ll_idxs = np.array([85166, 85174, 85192, 85194, 85439, 85487, 85489, 86015, 86117, 86985])
    # all shot numbers
    all_shot_idxs = np.hstack((lsn_idxs, usn_idxs, nt_idxs, xpt_idxs, ll_idxs))

    # LSN configuration
    lsn_idxs = np.array([86554, 86572, 86735])
    # NT configuration
    nt_idxs = np.array([85138, 85997])
    # XPT configuration
    xpt_idxs = np.array([85199, 85350, 85353, 85357, 85818, 85819])
    # LL configuration
    ll_idxs = np.array([85166, 85192, 85194, 85439, 85487, 85489, 86117])
    # all shot numbers
    reverse_field_shot_idxs = np.hstack((lsn_idxs, nt_idxs, xpt_idxs, ll_idxs))

    argv = sys.argv
    if len(argv) == 2:
        chunk_to_be_analyzed = int(argv[1])
    else:
        raise ValueError("Number of passed arguments must be either 2, with second argument specifying which chunk of data to analyse.")

    # define saving directory
    results_saving_dir = 'campaign_study_results'
    if not os.path.isdir(results_saving_dir):
        os.mkdir(results_saving_dir)

    run_campaign_study(shot_preprocessing_information=shot_information, chunk=chunk_to_be_analyzed,
                       saving_dir=results_saving_dir, reverse_field_shots=reverse_field_shot_idxs, usn_shots=usn_idxs)
