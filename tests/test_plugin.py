import pytest
import snakemake.common.tests

from snakemake_executor_plugin_local_gpu import ExecutorSettings


class TestWorkflows(snakemake.common.tests.TestWorkflowsLocalStorageBase):
    __test__ = True

    @pytest.fixture(autouse=True)
    def isolated_lock_dir(self, tmp_path):
        self.lock_dir = tmp_path / "locks"

    def get_executor(self):
        return "local-gpu"

    def get_executor_settings(self):
        return ExecutorSettings(lock_dir=self.lock_dir)
