from logging import getLogger

from abeja.train.client import Client
from abeja.train.statistics import Statistics as ABEJAStatistics

from chainer.training import extension
from chainer.training.extensions import log_report as log_report_module

logger = getLogger('callback')


class Statistics(extension.Extension):

    """Trainer extension to report the accumulated results to ABEJA Platform.
    This extension uses the log accumulated by a :class:`LogReport` extension
    to print specified entries of the log in a human-readable format.
    Args:
        entries (list of str): List of keys of observations to print.
        log_report (str or LogReport): Log report to accumulate the
            observations. This is either the name of a LogReport extensions
            registered to the trainer, or a LogReport instance to use
            internally.
    """

    def __init__(self, entries, total_epochs, obs_key='epoch', log_report='LogReport'):
        self._entries = entries
        self._log_report = log_report
        self._total_epochs = total_epochs
        self._obs_key = obs_key
        self.client = Client()

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
        if len(log) > 0:
            self._print(log[-1])

    def serialize(self, serializer):
        log_report = self._log_report
        if isinstance(log_report, log_report_module.LogReport):
            log_report.serialize(serializer['_log_report'])

    def _print(self, observation):
        train_loss = None
        train_acc = None
        val_loss = None
        val_acc = None

        train_list = {}
        val_list = {}

        epoch = observation[self._obs_key]
        statistics = ABEJAStatistics(num_epochs=self._total_epochs, epoch=epoch)

        for key, value in observation.items():
            keys = key.split('/')
            if len(keys) > 1 and keys[0] == 'main':
                name = '/'.join(keys[1:])
                if name == 'loss':
                    train_loss = value
                elif name == 'accuracy':
                    train_acc = value
                else:
                    train_list[name] = value
            elif len(keys) > 2 and keys[0] == 'validation' and keys[1] == 'main':
                name = '/'.join(keys[2:])
                if name == 'loss':
                    val_loss = value
                elif name == 'accuracy':
                    val_acc = value
                else:
                    val_list[name] = value

        statistics.add_stage(ABEJAStatistics.STAGE_TRAIN,
                             train_acc, train_loss, **train_list)
        statistics.add_stage(ABEJAStatistics.STAGE_VALIDATION,
                             val_acc, val_loss, **val_list)

        try:
            self.client.update_statistics(statistics)
        except Exception:
            logger.warning('failed to update statistics.')