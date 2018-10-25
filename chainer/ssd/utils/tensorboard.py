import os
from datetime import datetime
from logging import getLogger

from chainer.training import extension
from chainer.training.extensions import log_report as log_report_module

from tb_chainer import SummaryWriter

logger = getLogger('callback')


class Tensorboard(extension.Extension):

    """Trainer extension to tensorboard the accumulated results to ABEJA Platform.
    This extension uses the log accumulated by a :class:`LogReport` extension
    to print specified entries of the log in a human-readable format.
    Args:
        entries (list of str): List of keys of observations to print.
        log_report (str or LogReport): Log report to accumulate the
            observations. This is either the name of a LogReport extensions
            registered to the trainer, or a LogReport instance to use
            internally.
    """

    def __init__(self, entries, out_dir='logs', log_report='LogReport'):
        self._entries = entries
        self._log_report = log_report
        self._log_len = 0  # number of observations already printed
        self.writer = SummaryWriter(out_dir)

    def __call__(self, trainer):
        log_report = self._log_report
        if isinstance(log_report, str):
            log_report = trainer.get_extension(log_report)
        elif isinstance(log_report, log_report_module.LogReport):
            log_report(trainer)  # update the log report
        else:
            raise TypeError('log report has a wrong type %s' %
                            type(log_report))

        log = log_report.log
        log_len = self._log_len
        while len(log) > log_len:
            self._print(log[log_len])
            log_len += 1
        self._log_len = log_len

    def serialize(self, serializer):
        log_report = self._log_report
        if isinstance(log_report, log_report_module.LogReport):
            log_report.serialize(serializer['_log_report'])

    def _print(self, observation):
        epoch = observation['epoch']
        for key, value in observation.items():
            self.writer.add_scalar(key, value, epoch)