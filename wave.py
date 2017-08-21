from __future__ import division

import argparse
from collections import namedtuple

import numpy as np
import scipy.misc
import scipy.ndimage as ndimage

import matplotlib.pyplot as plt


def expi(x):
    return np.exp(1j * x)


def build_elements(system):
    angles = np.linspace(0, 2*np.pi, system.num_e, endpoint=False)
    elements = system.r_a * np.exp(1j * angles)
    return elements


def build_samples(system):
    limit = 1 / np.sqrt(2)
    lhs = np.linspace(-limit, limit, system.num_s)
    rhs = 1j * lhs
    samples = system.r_s * (lhs + rhs.reshape(-1, 1)).flatten()
    return samples


def build_transfer(system):
    samples = build_samples(system)
    elements = build_elements(system).reshape(1, -1)

    start, stop, num = system.f_low, system.f_high, system.num_f
    frequencies = np.linspace(start, stop, num, endpoint=True)
    wavelengths = system.c / frequencies
    wavelengths = wavelengths.reshape(-1, 1)

    transfer = []

    for index in range(system.num_s * system.num_s):
        distance = np.abs(samples[index] - elements)
        data = expi(2*np.pi * distance / wavelengths) / distance
        transfer.append(data)

    return np.dstack(transfer)


def calculate_weights(image, system):
    transfer = build_transfer(system)

    def real2complex(x):
        x = x.reshape(2, system.num_f, system.num_e)
        x = x[0] + 1j * x[1]
        return x

    def complex2real(x):
        x = x.flatten()
        x = np.stack((x.real, x.imag)).flatten()
        return x

    def evaluate_response(x):
        x = real2complex(x)
        x = x.reshape(system.num_f, system.num_e, 1)

        y = (transfer * x).sum(axis=1)
        p = (y.real*y.real + y.imag*y.imag).sum(axis=0)

        return p

    desired = image.flatten().reshape(1, 1, -1)

    def objective(x):
        error = desired - evaluate_response(x)
        return (error * error).sum()

    def fprime(x):
        x = real2complex(x)[:, :, np.newaxis]

        A = desired
        Q = transfer
        y = (Q * x).sum(axis=1, keepdims=True)
        p = (y.real*y.real + y.imag*y.imag).sum(axis=0, keepdims=True)

        sum_ = (2 * (p - A) * (Q.conj() * y)).sum(axis=2)

        sum_ = sum_.flatten()
        sum_ = np.stack((sum_.real, sum_.imag)).flatten()
        return sum_

    def fhess_p(x, p):
        x = real2complex(x)[:, :, np.newaxis]
        p = real2complex(p)[:, :, np.newaxis]

        A = desired
        Q = transfer

        y0 = (Q * x).sum(axis=1, keepdims=True)
        t0 = (Q.conj() * y0)

        y1 = (Q * p).sum(axis=1, keepdims=True)
        t1 = (Q.conj() * y1)

        p_ = (y0.real*y0.real + y0.imag*y0.imag).sum(axis=0, keepdims=True)

        lhs = 4 * (x.conj()*t1).sum(axis=(0, 1), keepdims=True) * t0
        rhs = 2 * t1 * (p_ - A)
        sum_ = (lhs + rhs).sum(axis=2)

        return complex2real(sum_)

    f = objective
    x0 = np.random.normal(size=2*system.num_e*system.num_f)

    minimizer = scipy.optimize.fmin_ncg(
        f, x0, fprime=fprime, fhess_p=fhess_p,
        maxiter=75, avextol=1e-4
    )
    minimizer = real2complex(minimizer)
    return minimizer


def simulate(weights, system):
    transfer = build_transfer(system)
    weights = weights.reshape(system.num_f, system.num_e, 1)
    y = (transfer * weights).sum(axis=1)
    p = (y.real*y.real + y.imag*y.imag).sum(axis=0)
    return p.reshape(system.num_s, system.num_s)


def build_argparser():
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
        '--signal_radius',
        type=float,
        default=0.2,
        help='signal radius in meters',
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
        default=500,
        help='lowest tone in Hertz',
    )

    parser.add_argument(
        '--highest_tone',
        type=float,
        default=25e3,
        help='highest tone in Hertz',
    )

    parser.add_argument(
        '--num_elements',
        type=int,
        default=20,
        help='number of transducer elements'
    )

    parser.add_argument(
        '--num_samples',
        type=int,
        default=20,
        help='number of samples in each dimension'
    )

    parser.add_argument(
        '--num_tones',
        type=int,
        default=20,
        help='number of tones used in each transducer'
    )

    return parser


System = namedtuple(
    'System',
    ['num_e', 'num_f', 'num_s', 'c', 'r_a', 'r_s', 'f_high', 'f_low']
)


def build_system(args):
    return System(
        num_e=args.num_elements,
        num_f=args.num_tones,
        num_s=args.num_samples,
        c=args.speed_of_sound,
        r_a=args.array_radius,
        r_s=args.signal_radius,
        f_high=args.highest_tone,
        f_low=args.lowest_tone
    )


def load_image(filepath, system):
    image = ndimage.imread(filepath, mode='L')

    grid_size = (system.num_s, system.num_s)
    image = scipy.misc.imresize(image, size=grid_size)

    image = image.astype(float)
    image /= image.max()
    image = 1 - image

    return image


def main():
    parser = build_argparser()
    args = parser.parse_args()

    system = build_system(args)

    desired = load_image('multiply.jpg', system)
    weights = calculate_weights(desired, system)
    actual = simulate(weights, system)

    plt.subplot(121)
    plt.imshow(actual, cmap='Greys')

    plt.subplot(122)
    plt.imshow(desired, cmap='Greys')

    plt.show()


if __name__ == '__main__':
    main()
