import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq
from xfuser.logger import init_logger
logger = init_logger(__name__)

def same_meta(node1: torch.fx.Node, node2: torch.fx.Node):
    """True if two nodes have the same metadata"""
    try:
        val1 = node1.meta.get("val")
        val2 = node2.meta.get("val")
        return (
            val1 is not None
            and val2 is not None
            and type(val1) == type(val2)
            and statically_known_true(sym_eq(val1.size(), val2.size()))
            and val1.layout == val2.layout
            and val1.dtype == val2.dtype
            and val1.device == val2.device
            and (
                val1.layout != torch.strided
                or statically_known_true(sym_eq(val1.stride(), val2.stride()))
            )
        )
    except Exception as e:
        return False

import torch._inductor.fx_passes.post_grad as post_grad

def apply_torch_compile_dynamic_shape_monkey_patch():
    # Save original function for future reference if needed
    global original_same_meta
    original_same_meta = post_grad.same_meta
    # Apply the monkey patch
    logger.info("Applying monkey patch for same_meta")
    post_grad.same_meta = same_meta

# Add a function to restore original if needed
def restore_torch_compile_dynamic_shape_monkey_patch():
    global original_same_meta
    if original_same_meta:
        post_grad.same_meta = original_same_meta