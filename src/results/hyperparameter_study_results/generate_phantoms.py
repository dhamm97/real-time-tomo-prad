import numpy as np
import os
import skimage.transform as skimt
import skimage.filters as skimf


def define_composite_phantom(coeffs, psi, trim_params=None, dim_shape=(120,41)):
    # coeffs: np.array([coeff_in, coeff_out, coeff_ring_core_solps, coeff_xpt_rad, coeff_core])
    xpoint_z_idx_base_psi=90 #, 14]
    xpoint_r_idx_base_psi=14
    if trim_params is None:
        trim_params = np.zeros(4)
    upper_trim = int(trim_params[0])
    lower_trim = int(120 - trim_params[1])
    hfs_trim = int(trim_params[2])
    lfs_trim = int(40 - trim_params[3])
    trimming_vals = np.array([upper_trim, lower_trim, hfs_trim, lfs_trim])
    z_xpoint_idx = int(dim_shape[0] * (xpoint_z_idx_base_psi - trimming_vals[0]) / (
                     trimming_vals[1] - trimming_vals[0]))
    psi_resh = skimt.resize(psi[upper_trim:lower_trim, hfs_trim:lfs_trim], dim_shape, anti_aliasing=True, mode='edge')
    idxs_neg_psi = np.where(psi_resh<0)
    idxs_core = (idxs_neg_psi[0][idxs_neg_psi[0]<z_xpoint_idx],
             idxs_neg_psi[1][idxs_neg_psi[0]<z_xpoint_idx])
    mask_core = np.zeros(dim_shape, dtype=bool)
    mask_core[idxs_core] = True

    inner_leg = skimt.resize(solps_ph_inner_leg[upper_trim:lower_trim, hfs_trim:lfs_trim], dim_shape, anti_aliasing=True, mode='edge')
    inner_leg /= np.max(inner_leg)
    outer_leg = skimt.resize(solps_ph_outer_leg[upper_trim:lower_trim, hfs_trim:lfs_trim], dim_shape, anti_aliasing=True, mode='edge')
    outer_leg /= np.max(outer_leg)
    ring_and_core = skimt.resize(solps_ph_ring_and_core[upper_trim:lower_trim, hfs_trim:lfs_trim], dim_shape, anti_aliasing=True, mode='edge')
    ring_and_core /= np.max(ring_and_core)

    xpoint_rad = skimt.resize(xpt_rad[upper_trim:lower_trim, hfs_trim:lfs_trim], dim_shape, anti_aliasing=True, mode='edge')

    core = np.exp(-((psi_resh-np.min(psi_resh))/(np.min(psi_resh)))**2 / (2*0.5**2))
    core *= mask_core
    core = skimf.gaussian(core, sigma=1.5)

    phantom = coeffs[0]*inner_leg + coeffs[1]*outer_leg + coeffs[2]*ring_and_core +\
              coeffs[3]*xpoint_rad + coeffs[4]*core
    return phantom, psi_resh, trimming_vals


if __name__ == '__main__':
    # define directory where phantoms are stored
    phantom_dir = 'phantoms'
    if not os.path.isdir(phantom_dir):
        os.mkdir(phantom_dir)

    # number of pixels in vertical and radial direction
    Nz, Nr = 120, 41
    # load the components for generation of the phantoms
    solps_ph_inner_leg = np.load("phantoms/solps_phantom_inner_leg.npy")
    solps_ph_outer_leg = np.load('phantoms/solps_phantom_outer_leg.npy')
    solps_ph_ring_and_core = np.load('phantoms/solps_phantom_ring_and_core.npy')
    xpt_rad = np.load('phantoms/xpt_rad.npy')
    # load base magnetic equilibrium
    mag_eq = np.load('phantoms/magnetic_equilibrium.npy')
    # load tcv mask
    tcv_mask = np.load("../../tcv_geometry/tcv_mask_1_subpixels_NINO.npy")

    # number of phantoms to be generated
    nb_phantoms = 1000
    # bounds for parameters
    trim_bounds = np.array([30, 20, 4, 3])
    coeff_bounds = np.array([1, 1, 1, 0.5, 0.5])
    scaling_bounds = np.array([1e5, 1e6])
    # initialize quantities to be saved
    phantoms = np.zeros((nb_phantoms, Nz, Nr))
    phantoms_psis = np.zeros((nb_phantoms, Nz, Nr))
    trim_values = np.zeros((nb_phantoms, 4))
    # seed
    np.random.seed(0)
    # generate phantoms
    for i in range(nb_phantoms):
        # generate trimming parameters for magnetic equilibrium variability
        trim_params = trim_bounds * np.random.rand(4)
        # generate coefficients of the different phantom components
        coeffs = np.random.rand(5) * coeff_bounds
        coeffs[2] = np.random.rand(1)[0] * coeffs[
            0] * 2 / 3  # ring coefficient should not be more than 2/3 the inner leg coefficient
        # generate the phantom
        phantoms[i, :, :], phantoms_psis[i, :, :], _ = define_composite_phantom(coeffs, mag_eq, trim_params=trim_params,
                                                                                dim_shape=(Nz, Nr))
        phantoms[i, :, :] *= np.random.rand(1)[0] * (scaling_bounds[1] - scaling_bounds[0]) + scaling_bounds[0]
        phantoms[i, :, :] *= tcv_mask
        trim_values[i, :] = trim_params

    np.save(phantom_dir+'/phantoms.npy', phantoms)
    np.save(phantom_dir + '/phantoms_psis.npy', phantoms_psis)
    np.save(phantom_dir + '/phantoms_trim_values.npy', trim_values)