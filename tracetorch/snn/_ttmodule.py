import torch
from torch import nn
from typing import Any, Set


class TTModule(nn.Module):
    """
    Base class that makes it trivial to call lifecycle methods (zero_states, detach_states, ...)
    across an entire model tree (registered submodules *and* common non-module containers).
    Inherit your models from this to get model.zero_states() / model.detach_states() behavior.
    """

    def get_param_count(self) -> int:
        total_params = sum(p.numel() for p in self.parameters())
        return total_params

    def zero_states(self) -> None:
        """Public API — zero any stateful child that implements zero_states()."""
        self._call_recursive("zero_states")

    def detach_states(self) -> None:
        """Public API — detach (stop-gradient) any stateful child that implements detach_states()."""
        self._call_recursive("detach_states")

    # --- internal recursive walker ---
    def _call_recursive(self, method_name: str) -> None:
        """
        Walk the object graph rooted at self and call `method_name()` on any object
        that exposes it. Uses a visited set to avoid duplicates / infinite loops.
        """
        visited: Set[int] = set()

        def recurse(obj: Any) -> None:
            oid = id(obj)
            if oid in visited:
                return
            visited.add(oid)

            # 1) If object defines the requested method, call it.
            #    If object is an instance of TTModule, we call the override (if any).
            #    If object is a plain python object with a callable method_name, call that too.
            fn = getattr(obj, method_name, None)
            if callable(fn):
                # Avoid calling TTModule.zero_states itself in a naive way that would loop forever:
                # - If obj is a TTModule instance, check whether its class actually overrides the method
                #   (i.e., the implementation is not the same as TTModule.method_name).
                if isinstance(obj, TTModule):
                    # get the unbound function defined on the class (if any)
                    cls_fn = getattr(obj.__class__, method_name, None)
                    base_fn = getattr(TTModule, method_name, None)
                    if cls_fn is not None and cls_fn is not base_fn:
                        # the class overrides the method -> call the override
                        try:
                            fn()
                        except TypeError:
                            # defensive: some implementations may accept args; ignore them here
                            fn()
                # else: class doesn't override, so don't call TTModule.zero_states (that would just re-enter recursion)
                else:
                    # Non-TTModule object that happens to have a method_name: call it
                    try:
                        fn()
                    except TypeError:
                        fn()

            # 2) Recurse into registered submodules (this covers ModuleList, Sequential, etc.)
            if isinstance(obj, nn.Module):
                for child in obj._modules.values():
                    if child is None:
                        continue
                    recurse(child)

            # 3) Recurse into container attributes that might hold modules or other objects with methods.
            #    This covers plain python lists/tuples/dicts/sets assigned as attributes on modules.
            #    We purposely ignore common atomic types (tensors, numbers, strings).
            try:
                attrs = getattr(obj, "__dict__", {})
            except Exception:
                attrs = {}

            for attr in attrs.values():
                if attr is None:
                    continue
                # common containers
                if isinstance(attr, (list, tuple, set)):
                    for el in attr:
                        recurse(el)
                elif isinstance(attr, dict):
                    for el in attr.values():
                        recurse(el)
                else:
                    # If attribute is a module or object, recurse (visited prevents duplicates)
                    if isinstance(attr, (nn.Module, object)):
                        recurse(attr)

        # start recursion from self
        recurse(self)
