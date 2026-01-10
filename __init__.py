import inspect
import os

from core.compiler import Sheaf as CoreSheaf
from core.error_handler import install_exception_handler

# Install error handler automatically when Sheaf is imported
install_exception_handler()


class Sheaf(CoreSheaf):
    def __init__(self, path=None):
        super().__init__()
        if path:
            self.load_from_path(path)

    def _get_external_caller_dir(self):
        """
        Finds the directory of the first script calling Sheaf
        that is not part of the sheaf package.
        """
        stack = inspect.stack()
        # Get the directory where this specific file (__init__.py) is located
        current_package_dir = os.path.dirname(os.path.abspath(__file__))

        for frame_info in stack:
            file_path = os.path.abspath(frame_info.filename)
            # If the file is not inside the sheaf package directory, it's our user!
            if not file_path.startswith(current_package_dir):
                return os.path.dirname(file_path)

        return os.getcwd()

    def load_from_path(self, path):
        if not os.path.isabs(path):
            caller_dir = self._get_external_caller_dir()
            path = os.path.join(caller_dir, path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Sheaf file not found: {path}")

        with open(path, "r") as f:
            self.load(f.read())

    def __getattr__(self, name):
        lisp_name = name.replace("_", "-")
        if lisp_name in self.registry:
            return self.registry[lisp_name]
        raise AttributeError(f"Sheaf: function '{lisp_name}' not found.")


__all__ = ["Sheaf"]
