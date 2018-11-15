from logging import getLogger

from abeja.train.client import Client
from abeja.train.statistics import Statistics as ABEJAStatistics

logger = getLogger('callback')


class Statistics(object):

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

    def __init__(self, total_epochs):
        self._total_epochs = total_epochs
        self.client = Client()

    def __call__(self, epoch, train_loss, train_acc, val_loss, val_acc):
        statistics = ABEJAStatistics(num_epochs=self._total_epochs, epoch=epoch)

        statistics.add_stage(ABEJAStatistics.STAGE_TRAIN,
                             train_acc, train_loss)
        statistics.add_stage(ABEJAStatistics.STAGE_VALIDATION,
                             val_acc, val_loss)

        try:
            self.client.update_statistics(statistics)
        except Exception:
            logger.warning('failed to update statistics.')
