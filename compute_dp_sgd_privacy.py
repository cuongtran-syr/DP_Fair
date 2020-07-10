# clone from https://github.com/ebagdasa/differential-privacy-vs-fairness/blob/master/compute_dp_sgd_privacy.py

r"""Command-line script for computing privacy of a model trained with DP-SGD.
The script applies the RDP accountant to estimate privacy budget of an iterated
Sampled Gaussian Mechanism. The mechanism's parameters are controlled by flags.
Example:
  compute_dp_sgd_privacy
    --N=60000 \
    --batch_size=256 \
    --noise_multiplier=1.12 \
    --epochs=60 \
    --delta=1e-5
The output states that DP-SGD with these parameters satisfies (2.92, 1e-5)-DP.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from absl import flags

from tfcode.rdp_accountant import compute_rdp
from tfcode.rdp_accountant import get_privacy_spent

FLAGS = flags.FLAGS

flags.DEFINE_integer('N', None, 'Total number of examples')
flags.DEFINE_integer('batch_size', None, 'Batch size')
flags.DEFINE_float('noise_multiplier', None, 'Noise multiplier for DP-SGD')
flags.DEFINE_float('epochs', None, 'Number of epochs (may be fractional)')
flags.DEFINE_float('delta', 1e-6, 'Target delta')

flags.mark_flag_as_required('N')
flags.mark_flag_as_required('batch_size')
flags.mark_flag_as_required('noise_multiplier')
flags.mark_flag_as_required('epochs')


def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
  """Compute and print results of DP-SGD analysis."""

  rdp = compute_rdp(q, sigma, steps, orders)

  eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)

  # print('DP-SGD with sampling rate = {:.3g}% and noise_multiplier = {} iterated'
  #       ' over {} steps satisfies'.format(100 * q, sigma, steps), end=' ')
  # print('differential privacy with eps = {:.3g} and delta = {}.'.format(
  #     eps, delta))
  # print('The optimal RDP order is {}.'.format(opt_order))
  #
  # if opt_order == max(orders) or opt_order == min(orders):
  #   print('The privacy estimate is likely to be improved by expanding '
  #         'the set of orders.')

  return eps

def main(argv):
  del argv  # argv is not used.

  q = FLAGS.batch_size / FLAGS.N  # q - the sampling ratio.

  if q > 1:
    raise app.UsageError('N must be larger than the batch size.')

  orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
            list(range(5, 64)) + [128, 256, 512])

  steps = int(math.ceil(FLAGS.epochs * FLAGS.N / FLAGS.batch_size))

  apply_dp_sgd_analysis(q, FLAGS.noise_multiplier, steps, orders, FLAGS.delta)


if __name__ == '__main__':
  app.run(main)