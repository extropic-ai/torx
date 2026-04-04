# type: ignore
from abc import abstractmethod

import equinox as eqx
import ihoop


class _StrictEqxMeta(ihoop.strict._StrictMeta, eqx._module._module._ModuleMeta):
    def __new__(mcs, name, bases, namespace, **kwargs):
        if name == "AbstractStrictModule":

            @abstractmethod
            def _strict_base_(self):
                raise NotImplementedError

            namespace["_strict_base_"] = _strict_base_
        elif not name.startswith("Abstract") and not name.startswith("_Abstract"):
            if "_strict_base_" not in namespace:

                def _strict_base_(self):
                    pass

                namespace["_strict_base_"] = _strict_base_
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class AbstractStrictModule(eqx.Module, ihoop.Strict, metaclass=_StrictEqxMeta):
    pass
