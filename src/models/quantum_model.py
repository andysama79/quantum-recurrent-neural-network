from typing import Tuple, List, Dict, Optional, Union, NewType

import torch
from torch import tensor
from math import pi, sqrt

import functools

# quantum states, batches, operators

Ket = NewType("Ket", torch.Tensor)
KetBatch = NewType("KetBatch", torch.Tensor)
Operator = NewType("Operator", torch.Tensor)

def ket0(num_qubits):
    psi = torch.zeros(2**num_qubits)
    psi[0] = 1
    return psi.view(shape=[2]*num_qubits)

def ket1(num_qubits):
    psi = torch.zeros(2**num_qubits)
    psi[-1] = 1
    return psi.view(shape=[2]*num_qubits)

def ket(descr):
    out = None
    for s in descr:
        if s == "0":
            psi = tensor([1.0, 0.0])
        elif s == "1":
            psi = tensor([0.0, 1.0])
        elif s == "+":
            # psi = normalize(tensor([1.0, 1.0]))
            psi = tensor([1.0, 1.0])
            psi = psi / psi.norm(p=2)
        elif s == "-":
            # psi = normalize(tensor([1.0, -1.0]))
            psi = tensor([1.0, -1.0])
            psi = psi / psi.norm(p=2)
        else:
            raise ValueError(f"Unknown state {s}")
        
        if out is None:
            out = psi
        else:
            out = torch.ger(out, psi).view(-1)
    
    return out.view(shape=[2]*len(descr))

# batches

KetOrBatch = NewType("KetOrBatch", Union[Ket, KetBatch])

def mark_batch(batch: KetBatch):
    batch._is_batch = True
    return batch

def ket_to_batch(psi, copies, share_memory=True):
    batch = (
        psi.expand(copies, *psi.shape)
        if share_memory
        else psi.repeat(copies, *[1 for _ in psi.shape])
    )

    return mark_batch(batch)

def is_batch(psi):
    return isinstance(psi, torch.Tensor) and hasattr(psi, "_is_batch")

def batch_size(batch):
    return batch.shape[0]

def mark_batch_like(model, to_mark):
    if is_batch(model):
        return mark_batch(to_mark)
    else:
        return to_mark
    
#todo: einsum indices
_EINSUM_BATCH_CHAR = "a"
_EINSUM_ALPHA = "bcdefghijklmnopqrstuvwxyz"

def squish_idcs_up(idcs):
    sorted_idcs = sorted(idcs)

    return "".join(_EINSUM_ALPHA[-i-1] for i in [len(idcs)-1-sorted_idcs.index(j) for j in idcs])

@functools.lru_cache(maxsize=10**6)
def einsum_indices_op(m, n, target_lanes, batched):
    idcs_op = squish_idcs_up("".join(_EINSUM_ALPHA[-l-1] for l in target_lanes)) + "".join(_EINSUM_ALPHA[i] for i in target_lanes)

    idcs_target = _EINSUM_ALPHA[:n]

    idcs_result = ""
    idcs_op_lut = dict(zip(idcs_op[m:], idcs_op[:m]))

    for idc in idcs_target:
        if idc in idcs_op_lut:
            idcs_result += idcs_op_lut[idc]
        else:
            idcs_result += idc

    idx_batch = _EINSUM_BATCH_CHAR if batched else ""

    return (idcs_op, idx_batch + idcs_target, idx_batch + idcs_result)

def einsum_indices_entrywise(m, n, target_lanes, batched):
    idcs_op = "".join(_EINSUM_ALPHA[i] for i in target_lanes)
    idcs_target = _EINSUM_ALPHA[:n]
    idcs_result = idcs_target

    idx_batch = _EINSUM_BATCH_CHAR if batched else ""
    return (idcs_op, idx_batch + idcs_target, idx_batch + idcs_result)
    
# quantum operations
def normalize(kob):
    if is_batch(kob):
        norm = kob.norm(p=2, dim=list(range(1, kob.dim())))
        return mark_batch((kob.transpose(0, -1) / norm).transpose(0, -1))
    else:
        return kob / kob.norm(p=2)
    
def num_state_qubits(kob):
    if is_batch(kob):
        return len(kob.shape) - 1
    else:
        return len(kob.shape)
    
def num_op_qubits(op):
    return len(op.shape) // 2

def probs(kob, measured_lanes=None, verbose=False):
    n = num_state_qubits(kob)

    if measured_lanes is None:
        measured_lanes = range(n)

    idx_batch = _EINSUM_BATCH_CHAR if is_batch(kob) else ""
    idcs_kept = "".join(_EINSUM_ALPHA[i] for i in measured_lanes)
    idcs_einsum = f"{idx_batch}{ _EINSUM_ALPHA[:n] }, {idx_batch}{ _EINSUM_ALPHA[:n] } -> {idx_batch}{idcs_kept}"

    if verbose:
        print(idcs_einsum)

    pvec = torch.einsum(idcs_einsum, kob, kob)
    
    if is_batch(kob):
        return mark_batch(pvec.reshape(batch_size(kob), -1))
    else:
        pvec.reshape(-1)

def apply(op, kob, target_lanes, verbose):
    n = num_state_qubits(kob)
    m = num_op_qubits(op)

    idcs_op, idcs_target, idcs_result = einsum_indices_op(
        m, n, tuple(target_lanes), batched=is_batch(kob)
    )

    idcs_einsum = f"{idcs_op}, {idcs_target} -> {idcs_result}"

    if verbose:
        print(idcs_einsum)

    return mark_batch_like(kob, torch.einsum(idcs_einsum, op, kob))

def apply_entrywise(state_op, kob, target_lanes, verbose):
    pass

def dot(a, b):
    idx_batch = _EINSUM_BATCH_CHAR if is_batch(a) or is_batch(b) else ""
    idcs_einsum = f"{idx_batch}{ _EINSUM_ALPHA[:len(a.shape)] }, {idx_batch}{ _EINSUM_ALPHA[:len(b.shape)] } -> {idx_batch}"

    return mark_batch_like(a, torch.einsum(idcs_einsum, a, b))

def ctrlMat(op, num_ctrl_lanes):
    if num_ctrl_lanes == 0:
        return op
    
    n = num_op_qubits(op)
    A = torch.eye(2**n)
    AB = torch.zeros(2**n, 2**n)
    BA = torch.zeros(2**n, 2**n)

    return ctrlMat(torch.cat([torch.cat([A, AB], dim=0), torch.cat([BA, op.view(2**n, -1)], dim=0)], dim=1).reshape(*[2]*(2*(n+1))), num_ctrl_lanes-1)