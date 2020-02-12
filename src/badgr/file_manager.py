from loguru import logger
import os
import shutil
import subprocess
import sys


class FileManager(object):

    badgr_dir = os.path.abspath(__file__)[:os.path.abspath(__file__).find('src/badgr')]
    data_dir = os.path.join(badgr_dir, 'data')
    configs_dir = os.path.join(badgr_dir, 'configs')

    def __init__(self, exp_name, is_continue=False, log_fname=None, config_fname=None, add_logger=True):
        self._exp_name = exp_name
        self._exp_dir = os.path.join(self.data_dir, self._exp_name)

        if is_continue:
            assert os.path.exists(self._exp_dir),\
                'Experiment folder "{0}" does not exists, but continue = True'.format(self._exp_name)
        else:
            assert not os.path.exists(self._exp_dir),\
                'Experiment folder "{0}" exists, but continue = False'.format(self._exp_name)

        if not os.path.exists(self.git_commit_fname):
            subprocess.call('cd {0}; git log -1 > {1}'.format(FileManager.badgr_dir, self.git_commit_fname),
                            shell=True)
        if not os.path.exists(self.git_diff_fname):
            subprocess.call('cd {0}; git diff > {1}'.format(FileManager.badgr_dir, self.git_diff_fname),
                            shell=True)

        if config_fname is not None:
            shutil.copy(config_fname, os.path.join(self.exp_dir, 'config.py'))

        if add_logger:
            logger.remove()
            if log_fname:
                logger.add(os.path.join(self.exp_dir, log_fname),
                           format=self._exp_name + " {time} {level} {message}",
                           level="DEBUG")
            logger.add(sys.stdout,
                       colorize=True,
                       format="<yellow>" + self._exp_name + "</yellow> | "
                              "<green>{time:HH:mm:ss}</green> | "
                              "<blue>{level: <8}</blue> | "
                              "<magenta>{name}:{function}:{line: <5}</magenta> | "
                              "<white>{message}</white>",
                       level="DEBUG",
                       filter=lambda record: record["level"].name == "DEBUG")
            logger.add(sys.stdout,
                       colorize=True,
                       format="<yellow>" + self._exp_name + "</yellow> | "
                               "<green>{time:HH:mm:ss}</green> | "
                               "<blue>{level: <8}</blue> | "
                               "<white>{message}</white>",
                       level="INFO")

    @property
    def exp_dir(self):
        os.makedirs(self._exp_dir, exist_ok=True)
        return self._exp_dir

    ###########
    ### Git ###
    ###########

    @property
    def git_dir(self):
        git_dir = os.path.join(self.exp_dir, 'git')
        os.makedirs(git_dir, exist_ok=True)
        return git_dir

    @property
    def git_commit_fname(self):
        return os.path.join(self.git_dir, 'commit.txt')

    @property
    def git_diff_fname(self):
        return os.path.join(self.git_dir, 'diff.txt')

    ##############
    ### Models ###
    ##############

    @property
    def ckpts_dir(self):
        ckpts_dir = os.path.join(self.exp_dir, 'ckpts')
        os.makedirs(ckpts_dir, exist_ok=True)
        return ckpts_dir

    @property
    def ckpt_prefix(self):
        return os.path.join(self.ckpts_dir, 'ckpt')
