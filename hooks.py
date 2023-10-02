import logging
from utils import parent_module

LOG = logging.getLogger(__name__)


def hook_model(model, pnames, detach=True, once=False):
    def linear_backward_hook(mod, grad_in, grad_out):
        if once:
            if mod in linear_backward_hook.visited:
                LOG.debug(f"visited {mod} {id(mod)} bwd hook")
                return
            linear_backward_hook.visited.add(mod)

        LOG.debug(f"{mod} {id(mod)} bwd hook")
        if not hasattr(mod, "weight"):
            LOG.error(f"{mod} {id(mod)} has no weight!")
            raise RuntimeError(f"{mod} {id(mod)} has no weight!")

        if hasattr(mod.weight, "__x__"):
            assert len(grad_out) == 1
            mod.weight.__delta__ = grad_out[0]
            if detach:
                mod.weight.__delta__ = mod.weight.__delta__.detach()
        else:
            LOG.error(f"{mod} {id(mod)} has no __x__")
            raise RuntimeError(f"{mod} {id(mod)} has no __x__")

    def linear_forward_hook(mod, activations, output):
        if once:
            if mod in linear_forward_hook.visited:
                LOG.debug(f"visited {mod} {id(mod)} fwd hook")
                return
            linear_forward_hook.visited.add(mod)

        LOG.debug(f"{mod} {id(mod)} fwd hook")
        assert len(activations) == 1
        mod.weight.__x__ = activations[0]
        if detach:
            mod.weight.__x__ = mod.weight.__x__.detach()

    if once:
        linear_backward_hook.visited = set()
        linear_forward_hook.visited = set()

    handles = []
    for m in [parent_module(model, pname) for pname in pnames]:
        handles.append(m.register_full_backward_hook(linear_backward_hook))
        handles.append(m.register_forward_hook(linear_forward_hook))

    model.handles = handles
