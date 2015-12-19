from __future__ import division

import argparse

import numpy as np
import scipy.optimize


def build_parameters_parser():
    parser = argparse.ArgumentParser(
        description='Parameters for optimizing and simulating Wave.'
    )

    parser.add_argument(
        '--array_radius',
        type=float,
        default=1.0,
        help='array radius in meters',
    )

    parser.add_argument(
        '--speed_of_sound',
        type=float,
        default=3300.0,
        help='speed of sound in meters per second',
    )

    parser.add_argument(
        '--lowest_tone',
        type=float,
        default=10e3,
        help='lowest tone in Hertz',
   )

    parser.add_argument(
        '--highest_tone',
        type=float,
        default=15e3,
        help='highest tone in Hertz',
    )

    parser.add_argument(
        '--num_elements',
        type=int,
        default=10,
        help='number of transducer elements'
    )

    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='number of samples in each dimension'
    )

    parser.add_argument(
        '--num_tones',
        type=int,
        default=30,
        help='number of tones used in each transducer'
    )

    return parser


def simulate(weights, params):

    array_radius = params.array_radius

    num_elements = params.num_elements
    num_samples = params.num_samples
    num_tones = params.num_tones

    angles = np.linspace(0, 2*np.pi, num_elements, endpoint=False)
    elements = array_radius * np.exp(1j * angles)

    f_low, f_high = params.lowest_tone, params.highest_tone
    frequencies = np.linspace(f_low, f_high, params.num_tones, endpoint=True)
    wavelengths = params.speed_of_sound / frequencies

    limit = 1 / np.sqrt(2)
    lhs = np.linspace(-limit, limit, num_samples)
    rhs = 1j * lhs
    samples = array_radius * (lhs + rhs.reshape(-1, 1)).flatten()

    distance = np.abs(samples - elements.reshape(-1, 1))
    wavelengths = wavelengths.reshape(-1, 1, 1)

    intensity = np.exp(1j * 2 * np.pi * distance / wavelengths) / distance
    intensity *= weights.reshape(num_tones, num_elements, 1)
    intensity = intensity.sum(axis=1)
    intensity = (np.abs(intensity) ** 2).sum(axis=0)
    intensity = intensity.reshape(num_samples, num_samples)
    return intensity


def calculate_weights(image, params):

    num_tones = params.num_tones
    num_elements = params.num_elements
    num_samples = params.num_samples

    angles = np.linspace(0, 2*np.pi, num_elements, endpoint=False)
    elements = params.array_radius * np.exp(1j * angles)

    limit = 1 / np.sqrt(2)
    lhs = np.linspace(-limit, limit, num_samples)
    rhs = 1j * lhs
    samples = params.array_radius * (lhs + rhs.reshape(-1, 1)).flatten()

    f_low, f_high = params.lowest_tone, params.highest_tone
    frequencies = np.linspace(f_low, f_high, params.num_tones, endpoint=True)
    wavelengths = params.speed_of_sound / frequencies
    wavelengths = wavelengths.reshape(-1, 1)

    transfer = []
    for index in range(samples.size):
        distance = np.abs(samples[index] - elements)
        data = np.exp(1j * 2*np.pi * distance / wavelengths) / distance
        transfer.append(data)

    def evaluate_response(x):
        x = x.reshape(2, num_tones, num_elements)
        x = x[0] + 1j * x[1]
        response = []
        for index in range(samples.size):
            Q = transfer[index]
            power = np.abs((Q * x).sum(axis=1) ** 2).sum()
            response.append(power)
        response = np.array(response)
        return response

    desired = image.flatten()
    def objective(x):
        error = desired - evaluate_response(x)
        return (error * error).sum()

    def real2complex(x):
        x = x.reshape(2, num_tones, num_elements)
        x = x[0] + 1j * x[1]
        return x

    def complex2real(x):
        x = x.flatten()
        x = np.stack((x.real, x.imag)).flatten()
        return x

    def fprime(x):
        x = real2complex(x)

        cumsum = np.zeros(x.shape, dtype=complex)
        for index in range(samples.size):
            A = desired[index]
            Q = transfer[index]
            power = np.abs((Q * x).sum(axis=1) ** 2).sum()
            cumsum += 2 * (power - A) * (Q.conj().T * (Q * x).sum(axis=1)).T

        cumsum = cumsum.flatten()
        cumsum = np.stack((cumsum.real, cumsum.imag)).flatten()
        return cumsum

    def fhess_p(x, p):
        x = real2complex(x)
        p = real2complex(p)

        cumsum = np.zeros(x.shape, dtype=complex)
        for index in range(samples.size):
            A = desired[index]
            Q = transfer[index]

            term_p = (Q.conj().T * (Q * p).sum(axis=1)).T

            product = (Q * x).sum(axis=1)
            term_x = (Q.conj().T * product).T
            power = (np.abs(product) ** 2).sum()

            cumsum += 4 * (x.conj() * term_p).sum() * term_x
            cumsum += 2 * power * term_p
            cumsum -= 2 * A * term_p

        cumsum = complex2real(cumsum)
        return cumsum

    f = objective
    x0 = np.random.normal(size=num_elements*num_tones*2)
    minimizer = scipy.optimize.fmin_ncg(f, x0, fprime=fprime, fhess_p=fhess_p)

    minimizer = minimizer.reshape(2, -1)
    minimizer = minimizer[0] + 1j * minimizer[1]
    return minimizer
