from scipy import constants
import numpy as np


def electron_wavelength(ht):
    """
    Returns electron wavelenght in nm.
    Parameters
    ----------
    ht : float
        High tension in kV.
    """
    m_e, q_e, c, h = constants.m_e, constants.elementary_charge, constants.c, constants.h
    momentum = 2 * m_e * q_e * ht * 1000 * (1 + q_e * ht * 1000 / (2 * m_e * c ** 2))
    wavelength = h / np.sqrt(momentum) * 1e9  # in nm
    return wavelength


def aberration_function(w, aber_dict, v_acc):
    # TODO: Taken from Florian! Use dictionary!
    w_cc = np.conjugate(w)
    chi_i = {'C1': aber_dict['C1'] * w * w_cc / 2}
    chi_sum = np.zeros_like(w)
    for key in aber_dict.keys():
        chi_sum += chi_i[key]
    return (2 * np.pi / electron_wavelength(v_acc)) * np.real(chi_sum)


def apply_aberrations(phasemap, aber_dict, v_acc):

    # Define complex scattering angle w
    f_freq_v = np.fft.fftfreq(phasemap.dim_uv[0], phasemap.a)
    f_freq_u = np.fft.fftfreq(phasemap.dim_uv[1], phasemap.a)
    f_freq_mesh = np.meshgrid(f_freq_u, f_freq_v)

    w = f_freq_mesh[0] + 1j * f_freq_mesh[1]
    w *= electron_wavelength(v_acc)

    chi = aberration_function(w, aber_dict, v_acc)

    wave = np.exp(1j * phasemap.phase)
    wave_fft = np.fft.fftn(wave) / np.prod(phasemap.dim_uv)
    # TODO: This shows the phase plate! Toggle/verbose?
    # import matplotlib.pyplot as plt
    # fig, axis = plt.subplots(1, 1)
    # axis.plot(np.fft.fftshift(w[0, :]*1E3),
    #           np.fft.fftshift(np.angle(np.exp(-1j * chi))[0, :]))
    # plt.matshow(np.fft.fftshift(np.angle(np.exp(-1j * chi))))

    wave_fft *= np.exp(-1j * chi)
    wave = np.fft.ifftn(wave_fft) * np.prod(phasemap.dim_uv)

    return wave, chi
