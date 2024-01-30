from . import BaseModel, register_model  
import numpy as np
from torch import Tensor
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from torch.nn import Parameter
import torch.nn as nn
np.set_printoptions(precision=4)
from textwrap import indent
from typing import Any, Dict, List, Optional, Tuple, Union,Any
import numpy as np
import scipy.sparse
from torch_sparse.storage import SparseStorage, get_layout







def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)

def scatter_mul(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.scatter_mul(src, index, dim, out, dim_size)

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out

def scatter_min(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_min(src, index, dim, out, dim_size)

def scatter_max(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)

def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
            out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor:

    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mul':
        return scatter_mul(src, index, dim, out, dim_size)
    elif reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    elif reduce == 'min':
        return scatter_min(src, index, dim, out, dim_size)[0]
    elif reduce == 'max':
        return scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise ValueError

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def segment_sum_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_sum_csr(src, indptr, out)

def segment_add_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_sum_csr(src, indptr, out)

def segment_mean_csr(src: torch.Tensor, indptr: torch.Tensor,
                     out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_mean_csr(src, indptr, out)

def segment_min_csr(
        src: torch.Tensor, indptr: torch.Tensor,
        out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.segment_min_csr(src, indptr, out)

def segment_max_csr(
        src: torch.Tensor, indptr: torch.Tensor,
        out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.segment_max_csr(src, indptr, out)

def segment_csr(src: torch.Tensor, indptr: torch.Tensor,
                out: Optional[torch.Tensor] = None,
                reduce: str = "sum") -> torch.Tensor:

    if reduce == 'sum' or reduce == 'add':
        return segment_sum_csr(src, indptr, out)
    elif reduce == 'mean':
        return segment_mean_csr(src, indptr, out)
    elif reduce == 'min':
        return segment_min_csr(src, indptr, out)[0]
    elif reduce == 'max':
        return segment_max_csr(src, indptr, out)[0]
    else:
        raise ValueError




def is_torch_sparse_tensor(src: Any) -> bool:
    if isinstance(src, Tensor):
        if src.layout == torch.sparse_coo:
            return True
        if src.layout == torch.sparse_csr:
            return True
        if src.layout == torch.sparse_csc:
            return True
    return False



@torch.jit.script
class SparseTensor(object):
    storage: SparseStorage

    def __init__(
        self,
        row: Optional[torch.Tensor] = None,
        rowptr: Optional[torch.Tensor] = None,
        col: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
        is_sorted: bool = False,
        trust_data: bool = False,
    ):
        self.storage = SparseStorage(
            row=row,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=sparse_sizes,
            rowcount=None,
            colptr=None,
            colcount=None,
            csr2csc=None,
            csc2csr=None,
            is_sorted=is_sorted,
            trust_data=trust_data,
        )

    @classmethod
    def from_storage(self, storage: SparseStorage):
        out = SparseTensor(
            row=storage._row,
            rowptr=storage._rowptr,
            col=storage._col,
            value=storage._value,
            sparse_sizes=storage._sparse_sizes,
            is_sorted=True,
            trust_data=True,
        )
        out.storage._rowcount = storage._rowcount
        out.storage._colptr = storage._colptr
        out.storage._colcount = storage._colcount
        out.storage._csr2csc = storage._csr2csc
        out.storage._csc2csr = storage._csc2csr
        return out

    @classmethod
    def from_edge_index(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
        is_sorted: bool = False,
        trust_data: bool = False,
    ):
        return SparseTensor(row=edge_index[0], rowptr=None, col=edge_index[1],
                            value=edge_attr, sparse_sizes=sparse_sizes,
                            is_sorted=is_sorted, trust_data=trust_data)

    @classmethod
    def from_dense(self, mat: torch.Tensor, has_value: bool = True):
        if mat.dim() > 2:
            index = mat.abs().sum([i for i in range(2, mat.dim())]).nonzero()
        else:
            index = mat.nonzero()
        index = index.t()

        row = index[0]
        col = index[1]

        value: Optional[torch.Tensor] = None
        if has_value:
            value = mat[row, col]

        return SparseTensor(row=row, rowptr=None, col=col, value=value,
                            sparse_sizes=(mat.size(0), mat.size(1)),
                            is_sorted=True, trust_data=True)

    @classmethod
    def from_torch_sparse_coo_tensor(self, mat: torch.Tensor,
                                     has_value: bool = True):
        mat = mat.coalesce()
        index = mat._indices()
        row, col = index[0], index[1]

        value: Optional[torch.Tensor] = None
        if has_value:
            value = mat.values()

        return SparseTensor(row=row, rowptr=None, col=col, value=value,
                            sparse_sizes=(mat.size(0), mat.size(1)),
                            is_sorted=True, trust_data=True)

    @classmethod
    def from_torch_sparse_csr_tensor(self, mat: torch.Tensor,
                                     has_value: bool = True):
        rowptr = mat.crow_indices()
        col = mat.col_indices()

        value: Optional[torch.Tensor] = None
        if has_value:
            value = mat.values()

        return SparseTensor(row=None, rowptr=rowptr, col=col, value=value,
                            sparse_sizes=(mat.size(0), mat.size(1)),
                            is_sorted=True, trust_data=True)

    @classmethod
    def eye(self, M: int, N: Optional[int] = None, has_value: bool = True,
            dtype: Optional[int] = None, device: Optional[torch.device] = None,
            fill_cache: bool = False):

        N = M if N is None else N

        row = torch.arange(min(M, N), device=device)
        col = row

        rowptr = torch.arange(M + 1, device=row.device)
        if M > N:
            rowptr[N + 1:] = N

        value: Optional[torch.Tensor] = None
        if has_value:
            value = torch.ones(row.numel(), dtype=dtype, device=row.device)

        rowcount: Optional[torch.Tensor] = None
        colptr: Optional[torch.Tensor] = None
        colcount: Optional[torch.Tensor] = None
        csr2csc: Optional[torch.Tensor] = None
        csc2csr: Optional[torch.Tensor] = None

        if fill_cache:
            rowcount = torch.ones(M, dtype=torch.long, device=row.device)
            if M > N:
                rowcount[N:] = 0

            colptr = torch.arange(N + 1, dtype=torch.long, device=row.device)
            colcount = torch.ones(N, dtype=torch.long, device=row.device)
            if N > M:
                colptr[M + 1:] = M
                colcount[M:] = 0
            csr2csc = csc2csr = row

        out = SparseTensor(
            row=row,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=(M, N),
            is_sorted=True,
            trust_data=True,
        )
        out.storage._rowcount = rowcount
        out.storage._colptr = colptr
        out.storage._colcount = colcount
        out.storage._csr2csc = csr2csc
        out.storage._csc2csr = csc2csr
        return out

    def copy(self):
        return self.from_storage(self.storage)

    def clone(self):
        return self.from_storage(self.storage.clone())

    def type(self, dtype: torch.dtype, non_blocking: bool = False):
        value = self.storage.value()
        if value is None or dtype == value.dtype:
            return self
        return self.from_storage(
            self.storage.type(dtype=dtype, non_blocking=non_blocking))

    def type_as(self, tensor: torch.Tensor, non_blocking: bool = False):
        return self.type(dtype=tensor.dtype, non_blocking=non_blocking)

    def to_device(self, device: torch.device, non_blocking: bool = False):
        if device == self.device():
            return self
        return self.from_storage(
            self.storage.to_device(device=device, non_blocking=non_blocking))

    def device_as(self, tensor: torch.Tensor, non_blocking: bool = False):
        return self.to_device(device=tensor.device, non_blocking=non_blocking)

    # Formats #################################################################

    def coo(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.storage.row(), self.storage.col(), self.storage.value()

    def csr(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.storage.rowptr(), self.storage.col(), self.storage.value()

    def csc(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        perm = self.storage.csr2csc()
        value = self.storage.value()
        if value is not None:
            value = value[perm]
        return self.storage.colptr(), self.storage.row()[perm], value

    # Storage inheritance #####################################################

    def has_value(self) -> bool:
        return self.storage.has_value()

    def set_value_(self, value: Optional[torch.Tensor],
                   layout: Optional[str] = None):
        self.storage.set_value_(value, layout)
        return self

    def set_value(self, value: Optional[torch.Tensor],
                  layout: Optional[str] = None):
        return self.from_storage(self.storage.set_value(value, layout))

    def sparse_sizes(self) -> Tuple[int, int]:
        return self.storage.sparse_sizes()

    def sparse_size(self, dim: int) -> int:
        return self.storage.sparse_sizes()[dim]

    def sparse_resize(self, sparse_sizes: Tuple[int, int]):
        return self.from_storage(self.storage.sparse_resize(sparse_sizes))

    def sparse_reshape(self, num_rows: int, num_cols: int):
        return self.from_storage(
            self.storage.sparse_reshape(num_rows, num_cols))

    def is_coalesced(self) -> bool:
        return self.storage.is_coalesced()

    def coalesce(self, reduce: str = "sum"):
        return self.from_storage(self.storage.coalesce(reduce))

    def fill_cache_(self):
        self.storage.fill_cache_()
        return self

    def clear_cache_(self):
        self.storage.clear_cache_()
        return self

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False

        if self.sizes() != other.sizes():
            return False

        rowptrA, colA, valueA = self.csr()
        rowptrB, colB, valueB = other.csr()

        if valueA is None and valueB is not None:
            return False
        if valueA is not None and valueB is None:
            return False
        if not torch.equal(rowptrA, rowptrB):
            return False
        if not torch.equal(colA, colB):
            return False
        if valueA is None and valueB is None:
            return True
        return torch.equal(valueA, valueB)

    # Utility functions #######################################################

    def fill_value_(self, fill_value: float, dtype: Optional[int] = None):
        value = torch.full((self.nnz(), ), fill_value, dtype=dtype,
                           device=self.device())
        return self.set_value_(value, layout='coo')

    def fill_value(self, fill_value: float, dtype: Optional[int] = None):
        value = torch.full((self.nnz(), ), fill_value, dtype=dtype,
                           device=self.device())
        return self.set_value(value, layout='coo')

    def sizes(self) -> List[int]:
        sparse_sizes = self.sparse_sizes()
        value = self.storage.value()
        if value is not None:
            return list(sparse_sizes) + list(value.size())[1:]
        else:
            return list(sparse_sizes)

    def size(self, dim: int) -> int:
        return self.sizes()[dim]

    def dim(self) -> int:
        return len(self.sizes())

    def nnz(self) -> int:
        return self.storage.col().numel()

    def numel(self) -> int:
        value = self.storage.value()
        if value is not None:
            return value.numel()
        else:
            return self.nnz()

    def density(self) -> float:
        return self.nnz() / (self.sparse_size(0) * self.sparse_size(1))

    def sparsity(self) -> float:
        return 1 - self.density()

    def avg_row_length(self) -> float:
        return self.nnz() / self.sparse_size(0)

    def avg_col_length(self) -> float:
        return self.nnz() / self.sparse_size(1)

    def bandwidth(self) -> int:
        row, col, _ = self.coo()
        return int((row - col).abs_().max())

    def avg_bandwidth(self) -> float:
        row, col, _ = self.coo()
        return float((row - col).abs_().to(torch.float).mean())

    def bandwidth_proportion(self, bandwidth: int) -> float:
        row, col, _ = self.coo()
        tmp = (row - col).abs_()
        return int((tmp <= bandwidth).sum()) / self.nnz()

    def is_quadratic(self) -> bool:
        return self.sparse_size(0) == self.sparse_size(1)

    def is_symmetric(self) -> bool:
        if not self.is_quadratic():
            return False

        rowptr, col, value1 = self.csr()
        colptr, row, value2 = self.csc()

        if (rowptr != colptr).any() or (col != row).any():
            return False

        if value1 is None or value2 is None:
            return True
        else:
            return bool((value1 == value2).all())

    def to_symmetric(self, reduce: str = "sum"):
        N = max(self.size(0), self.size(1))

        row, col, value = self.coo()
        idx = col.new_full((2 * col.numel() + 1, ), -1)
        idx[1:row.numel() + 1] = row
        idx[row.numel() + 1:] = col
        idx[1:] *= N
        idx[1:row.numel() + 1] += col
        idx[row.numel() + 1:] += row

        idx, perm = idx.sort()
        mask = idx[1:] > idx[:-1]
        perm = perm[1:].sub_(1)
        idx = perm[mask]

        if value is not None:
            ptr = mask.nonzero().flatten()
            ptr = torch.cat([ptr, ptr.new_full((1, ), perm.size(0))])
            value = torch.cat([value, value])[perm]
            value = segment_csr(value, ptr, reduce=reduce)

        new_row = torch.cat([row, col], dim=0, out=perm)[idx]
        new_col = torch.cat([col, row], dim=0, out=perm)[idx]

        out = SparseTensor(
            row=new_row,
            rowptr=None,
            col=new_col,
            value=value,
            sparse_sizes=(N, N),
            is_sorted=True,
            trust_data=True,
        )
        return out

    def detach_(self):
        value = self.storage.value()
        if value is not None:
            value.detach_()
        return self

    def detach(self):
        value = self.storage.value()
        if value is not None:
            value = value.detach()
        return self.set_value(value, layout='coo')

    def requires_grad(self) -> bool:
        value = self.storage.value()
        if value is not None:
            return value.requires_grad
        else:
            return False

    def requires_grad_(self, requires_grad: bool = True,
                       dtype: Optional[int] = None):
        if requires_grad and not self.has_value():
            self.fill_value_(1., dtype)

        value = self.storage.value()
        if value is not None:
            value.requires_grad_(requires_grad)
        return self

    def pin_memory(self):
        return self.from_storage(self.storage.pin_memory())

    def is_pinned(self) -> bool:
        return self.storage.is_pinned()

    def device(self):
        return self.storage.col().device

    def cpu(self):
        return self.to_device(device=torch.device('cpu'), non_blocking=False)

    def cuda(self):
        return self.from_storage(self.storage.cuda())

    def is_cuda(self) -> bool:
        return self.storage.col().is_cuda

    def dtype(self):
        value = self.storage.value()
        return value.dtype if value is not None else torch.float

    def is_floating_point(self) -> bool:
        value = self.storage.value()
        return torch.is_floating_point(value) if value is not None else True

    def bfloat16(self):
        return self.type(dtype=torch.bfloat16, non_blocking=False)

    def bool(self):
        return self.type(dtype=torch.bool, non_blocking=False)

    def byte(self):
        return self.type(dtype=torch.uint8, non_blocking=False)

    def char(self):
        return self.type(dtype=torch.int8, non_blocking=False)

    def half(self):
        return self.type(dtype=torch.half, non_blocking=False)

    def float(self):
        return self.type(dtype=torch.float, non_blocking=False)

    def double(self):
        return self.type(dtype=torch.double, non_blocking=False)

    def short(self):
        return self.type(dtype=torch.short, non_blocking=False)

    def int(self):
        return self.type(dtype=torch.int, non_blocking=False)

    def long(self):
        return self.type(dtype=torch.long, non_blocking=False)

    # Conversions #############################################################

    def to_dense(self, dtype: Optional[int] = None) -> torch.Tensor:
        row, col, value = self.coo()

        if value is not None:
            mat = torch.zeros(self.sizes(), dtype=value.dtype,
                              device=self.device())
        else:
            mat = torch.zeros(self.sizes(), dtype=dtype, device=self.device())

        if value is not None:
            mat[row, col] = value
        else:
            mat[row, col] = torch.ones(self.nnz(), dtype=mat.dtype,
                                       device=mat.device)

        return mat

    def to_torch_sparse_coo_tensor(
            self, dtype: Optional[int] = None) -> torch.Tensor:
        row, col, value = self.coo()
        index = torch.stack([row, col], dim=0)

        if value is None:
            value = torch.ones(self.nnz(), dtype=dtype, device=self.device())

        return torch.sparse_coo_tensor(index, value, self.sizes())

    def to_torch_sparse_csr_tensor(
            self, dtype: Optional[int] = None) -> torch.Tensor:
        rowptr, col, value = self.csr()

        if value is None:
            value = torch.ones(self.nnz(), dtype=dtype, device=self.device())

        return torch.sparse_csr_tensor(rowptr, col, value, self.sizes())

    def to_torch_sparse_csc_tensor(
            self, dtype: Optional[int] = None) -> torch.Tensor:
        colptr, row, value = self.csc()

        if value is None:
            value = torch.ones(self.nnz(), dtype=dtype, device=self.device())

        return torch.sparse_csc_tensor(colptr, row, value, self.sizes())

# Python Bindings #############################################################

def share_memory_(self: SparseTensor) -> SparseTensor:
    self.storage.share_memory_()
    return self


def is_shared(self: SparseTensor) -> bool:
    return self.storage.is_shared()


def to(self, *args: Optional[List[Any]],
       **kwargs: Optional[Dict[str, Any]]) -> SparseTensor:
    device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)[:3]

    if dtype is not None:
        self = self.type(dtype=dtype, non_blocking=non_blocking)
    if device is not None:
        self = self.to_device(device=device, non_blocking=non_blocking)

    return self


def cpu(self) -> SparseTensor:
    return self.device_as(torch.tensor(0., device='cpu'))


def cuda(self, device: Optional[Union[int, str]] = None,
         non_blocking: bool = False):
    return self.device_as(torch.tensor(0., device=device or 'cuda'))


def __getitem__(self: SparseTensor, index: Any) -> SparseTensor:
    index = list(index) if isinstance(index, tuple) else [index]
    # More than one `Ellipsis` is not allowed...
    if len([
            i for i in index
            if not isinstance(i, (torch.Tensor, np.ndarray)) and i == ...
    ]) > 1:
        raise SyntaxError

    dim = 0
    out = self
    while len(index) > 0:
        item = index.pop(0)
        if isinstance(item, (list, tuple)):
            item = torch.tensor(item, device=self.device())
        if isinstance(item, np.ndarray):
            item = torch.from_numpy(item).to(self.device())

        if isinstance(item, int):
            out = out.select(dim, item)
            dim += 1
        elif isinstance(item, slice):
            if item.step is not None:
                raise ValueError('Step parameter not yet supported.')

            start = 0 if item.start is None else item.start
            start = self.size(dim) + start if start < 0 else start

            stop = self.size(dim) if item.stop is None else item.stop
            stop = self.size(dim) + stop if stop < 0 else stop

            out = out.narrow(dim, start, max(stop - start, 0))
            dim += 1
        elif torch.is_tensor(item):
            if item.dtype == torch.bool:
                out = out.masked_select(dim, item)
                dim += 1
            elif item.dtype == torch.long:
                out = out.index_select(dim, item)
                dim += 1
        elif item == Ellipsis:
            if self.dim() - len(index) < dim:
                raise SyntaxError
            dim = self.dim() - len(index)
        else:
            raise SyntaxError

    return out


def __repr__(self: SparseTensor) -> str:
    i = ' ' * 6
    row, col, value = self.coo()
    infos = []
    infos += [f'row={indent(row.__repr__(), i)[len(i):]}']
    infos += [f'col={indent(col.__repr__(), i)[len(i):]}']

    if value is not None:
        infos += [f'val={indent(value.__repr__(), i)[len(i):]}']

    infos += [
        f'size={tuple(self.sizes())}, nnz={self.nnz()}, '
        f'density={100 * self.density():.02f}%'
    ]

    infos = ',\n'.join(infos)

    i = ' ' * (len(self.__class__.__name__) + 1)
    return f'{self.__class__.__name__}({indent(infos, i)[len(i):]})'


SparseTensor.share_memory_ = share_memory_
SparseTensor.is_shared = is_shared
SparseTensor.to = to
SparseTensor.cpu = cpu
SparseTensor.cuda = cuda
SparseTensor.__getitem__ = __getitem__
SparseTensor.__repr__ = __repr__

# Scipy Conversions ###########################################################

ScipySparseMatrix = Union[scipy.sparse.coo_matrix, scipy.sparse.csr_matrix,
                          scipy.sparse.csc_matrix]


@torch.jit.ignore
def from_scipy(mat: ScipySparseMatrix, has_value: bool = True) -> SparseTensor:
    colptr = None
    if isinstance(mat, scipy.sparse.csc_matrix):
        colptr = torch.from_numpy(mat.indptr).to(torch.long)

    mat = mat.tocsr()
    rowptr = torch.from_numpy(mat.indptr).to(torch.long)
    mat = mat.tocoo()
    row = torch.from_numpy(mat.row).to(torch.long)
    col = torch.from_numpy(mat.col).to(torch.long)
    value = None
    if has_value:
        value = torch.from_numpy(mat.data)
    sparse_sizes = mat.shape[:2]

    storage = SparseStorage(row=row, rowptr=rowptr, col=col, value=value,
                            sparse_sizes=sparse_sizes, rowcount=None,
                            colptr=colptr, colcount=None, csr2csc=None,
                            csc2csr=None, is_sorted=True)

    return SparseTensor.from_storage(storage)


@torch.jit.ignore
def to_scipy(self: SparseTensor, layout: Optional[str] = None,
             dtype: Optional[torch.dtype] = None) -> ScipySparseMatrix:
    assert self.dim() == 2
    layout = get_layout(layout)

    if not self.has_value():
        ones = torch.ones(self.nnz(), dtype=dtype).numpy()

    if layout == 'coo':
        row, col, value = self.coo()
        row = row.detach().cpu().numpy()
        col = col.detach().cpu().numpy()
        value = value.detach().cpu().numpy() if self.has_value() else ones
        return scipy.sparse.coo_matrix((value, (row, col)), self.sizes())
    elif layout == 'csr':
        rowptr, col, value = self.csr()
        rowptr = rowptr.detach().cpu().numpy()
        col = col.detach().cpu().numpy()
        value = value.detach().cpu().numpy() if self.has_value() else ones
        return scipy.sparse.csr_matrix((value, col, rowptr), self.sizes())
    elif layout == 'csc':
        colptr, row, value = self.csc()
        colptr = colptr.detach().cpu().numpy()
        row = row.detach().cpu().numpy()
        value = value.detach().cpu().numpy() if self.has_value() else ones
        return scipy.sparse.csc_matrix((value, row, colptr), self.sizes())

SparseTensor.from_scipy = from_scipy
SparseTensor.to_scipy = to_scipy



def softmax(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0,
) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`

    Examples:

        >>> src = torch.tensor([1., 1., 1., 1.])
        >>> index = torch.tensor([0, 0, 1, 2])
        >>> ptr = torch.tensor([0, 2, 3, 4])
        >>> softmax(src, index)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> softmax(src, None, ptr)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> src = torch.randn(4, 4)
        >>> ptr = torch.tensor([0, 4])
        >>> softmax(src, index, dim=-1)
        tensor([[0.7404, 0.2596, 1.0000, 1.0000],
                [0.1702, 0.8298, 1.0000, 1.0000],
                [0.7607, 0.2393, 1.0000, 1.0000],
                [0.8062, 0.1938, 1.0000, 1.0000]])
    """
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        count = ptr[1:] - ptr[:-1]
        ptr = ptr.view(size)
        src_max = segment_csr(src.detach(), ptr, reduce='max')
        src_max = src_max.repeat_interleave(count, dim=dim)
        out = (src - src_max).exp()
        out_sum = segment_csr(out, ptr, reduce='sum') + 1e-16
        out_sum = out_sum.repeat_interleave(count, dim=dim)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
        out = src - src_max.index_select(dim, index)
        out = out.exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + 1e-16
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / out_sum


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    
    pass


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    
    pass


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        if is_torch_sparse_tensor(edge_index):
            return max(edge_index.size(0), edge_index.size(1))
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))



class DisenLayer(nn.Module):
    def __init__(self, edge_index, edge_type, in_channels, out_channels, num_rels, 
                 act=lambda x: x, params=None, head_num=1):
        #super(self.__class__, self).__init__(aggr='add', flow='target_to_source', node_dim=0)
        ########################################
        super(DisenLayer, self).__init__()
        self.node_dim = 0
        ###################################
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.device = None
        self.head_num = head_num
        self.num_rels = num_rels

        # params for init
#######################
        self.drop = torch.nn.Dropout(self.p.gcn_drop)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn = torch.nn.BatchNorm1d(self.p.num_factors * out_channels)
        if self.p.bias:
            self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

        
        num_edges = self.edge_index.size(1) // 2
        if self.device is None:
            self.device = self.edge_index.device
        self.in_index, self.out_index = self.edge_index[:, :num_edges], self.edge_index[:, num_edges:]
        self.in_type, self.out_type = self.edge_type[:num_edges], self.edge_type[num_edges:]
        self.loop_index = torch.stack([torch.arange(self.p.num_ent), torch.arange(self.p.num_ent)]).to(self.device)
        self.loop_type = torch.full((self.p.num_ent,), 2 * self.num_rels, dtype=torch.long).to(self.device)
        num_ent = self.p.num_ent

        self.leakyrelu = nn.LeakyReLU(0.2)
        if self.p.att_mode == 'cat_emb' or self.p.att_mode == 'cat_weight':
            self.att_weight = get_param((1, self.p.num_factors, 2 * out_channels))
        else:
            self.att_weight = get_param((1, self.p.num_factors, out_channels))
        self.rel_weight = get_param((2 * self.num_rels + 1, self.p.num_factors, out_channels))
        self.loop_rel = get_param((1, out_channels))
        self.w_rel = get_param((out_channels, out_channels))

    def forward(self, x, rel_embed, mode):
#       message  和 aggregate，update
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        edge_index = torch.cat([self.edge_index, self.loop_index], dim=1)
        edge_type = torch.cat([self.edge_type, self.loop_type])

        # x.shape == [14541,3,200],  edge_index.shape == [2,558771],     rel_embed.shape == [475,200]
        # rel_weight.shape == [475,3,200]

        # 原代码   out真实形状为[14541,3,200]
        #out = self.propagate(edge_index, size=None, x=x, edge_type=edge_type,rel_embed=rel_embed, rel_weight=self.rel_weight)
        
#############################修改后代码###########################################
        # flow 是目标导源，这里j表示源节点，但是用到的却是edge_index[1]（真实目标节点）
        edge_index_i= edge_index[0]
        edge_index_j= edge_index[1]
        x_i = torch.index_select(x, dim=0, index=edge_index_i)
        x_j = torch.index_select(x, dim=0, index=edge_index_j)


        message_res = self.message(edge_index_i=edge_index_i,edge_index_j=edge_index_j,x_i=x_i,x_j=x_j,
                                   edge_type=edge_type,rel_embed=rel_embed,rel_weight=self.rel_weight)
        # message_res.shape  ==  [558771,3,200]
        aggr_res = self.aggregate(input=message_res,edge_index_i=edge_index_i)
        out =  self.update(aggr_res)  # out.shape  真正的形状应该是[14541,3,200]
#######################################################################        
        if self.p.bias:
            out = out + self.bias
        out = self.bn(out.view(-1, self.p.num_factors * self.p.gcn_dim)).view(-1, self.p.num_factors, self.p.gcn_dim)
        # out.shape == [14541,3,200]
        entity1 = out if self.p.no_act else self.act(out)
        return entity1, torch.matmul(rel_embed, self.w_rel)[:-1]

    def message(self, edge_index_i, edge_index_j, x_i, x_j, edge_type, rel_embed, rel_weight):
        '''
        edge_index_i : [E]
        x_i: [E, F]
        x_j: [E, F]
        '''
        rel_embed = torch.index_select(rel_embed, 0, edge_type)
        rel_weight = torch.index_select(rel_weight, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_embed, rel_weight)
        # start to compute the attention
        alpha = self._get_attention(edge_index_i, edge_index_j, x_i, x_j, rel_embed, rel_weight, xj_rel)
        alpha = self.drop(alpha)

        # xj_rel  == [558771,3,200]  alpha == [558771,3,1]  , 相乘之后的形状是[558771,3,200]
        return xj_rel * alpha  # 每条边上，加权后的每条边的源节点特征

    def aggregate(self,input,edge_index_i): # input是每条边上源节点的特征，edge_index_i是每条边上目标节点的id
        return scatter_sum(input,edge_index_i,dim=0)

    def update(self, aggr_out):     #   aggr_out == [14541,3,200]
        return aggr_out

    def _get_attention(self, edge_index_i, edge_index_j, x_i, x_j, rel_embed, rel_weight, mes_xj):
        if self.p.att_mode == 'learn':
            alpha = self.leakyrelu(torch.einsum('ekf, xkf->ek', [mes_xj, self.att_weight])) # [E K]
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)

        elif self.p.att_mode == 'dot_weight':
            sub_rel_emb = x_i * rel_weight
            obj_rel_emb = x_j * rel_weight

            alpha = self.leakyrelu(torch.einsum('ekf,ekf->ek', [sub_rel_emb, obj_rel_emb]))
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)

        elif self.p.att_mode == 'dot_emb':
            sub_rel_emb = x_i * rel_embed.unsqueeze(1)
            obj_rel_emb = x_j * rel_embed.unsqueeze(1)

            alpha = self.leakyrelu(torch.einsum('ekf,ekf->ek', [sub_rel_emb, obj_rel_emb]))
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)

        elif self.p.att_mode == 'cat_weight':
            sub_rel_emb = x_i * rel_weight
            obj_rel_emb = x_j * rel_weight

            alpha = self.leakyrelu(torch.einsum('ekf,xkf->ek', torch.cat([sub_rel_emb, obj_rel_emb], dim=2), self.att_weight))
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)

        elif self.p.att_mode == 'cat_emb':
            sub_rel_emb = x_i * rel_embed.unsqueeze(1)
            obj_rel_emb = x_j * rel_embed.unsqueeze(1)

            alpha = self.leakyrelu(torch.einsum('ekf,xkf->ek', torch.cat([sub_rel_emb, obj_rel_emb], dim=2), self.att_weight))
            alpha = softmax(alpha, edge_index_i, num_nodes=self.p.num_ent)
        else:
            raise NotImplementedError

        return alpha.unsqueeze(2)


    def rel_transform(self, ent_embed, rel_embed, rel_weight, opn=None):
        if opn is None:
            opn = self.p.opn
        if opn == 'corr':
            trans_embed = ccorr(ent_embed * rel_weight, rel_embed.unsqueeze(1))
        elif opn == 'corr_ra':
            trans_embed = ccorr(ent_embed * rel_weight, rel_embed)
        elif opn == 'sub':
            trans_embed = ent_embed * rel_weight - rel_embed.unsqueeze(1)
        elif opn == 'es':
            trans_embed = ent_embed
        elif opn == 'sub_ra':
            trans_embed = ent_embed * rel_weight - rel_embed.unsqueeze(1)
        elif opn == 'mult':
            trans_embed = (ent_embed * rel_embed.unsqueeze(1)) * rel_weight
        elif opn == 'mult_ra':
            trans_embed = (ent_embed * rel_embed) * rel_weight
        elif opn == 'cross':
            trans_embed = ent_embed * rel_embed.unsqueeze(1) * rel_weight + ent_embed * rel_weight
        elif opn == 'cross_wo_rel':
            trans_embed = ent_embed * rel_weight
        elif opn == 'cross_simplfy':
            trans_embed = ent_embed * rel_embed + ent_embed
        elif opn == 'concat':
            trans_embed = torch.cat([ent_embed, rel_embed], dim=1)
        elif opn == 'concat_ra':
            trans_embed = torch.cat([ent_embed, rel_embed], dim=1) * rel_weight
        elif opn == 'ent_ra':
            trans_embed = ent_embed * rel_weight + rel_embed
        else:
            raise NotImplementedError

        return trans_embed

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)

class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
#         print(x_samples.size())
#         print(y_samples.size())
        mu, logvar = self.get_mu_logvar(x_samples)

        return (-(mu - y_samples)**2 /2./logvar.exp()).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
        return torch.mm(x, self.weight) + self.bias

class CapsuleBase(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(CapsuleBase, self).__init__(params)
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.device = self.edge_index.device
        self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        self.init_rel = get_param((num_rel * 2, self.p.gcn_dim))
        self.pca = SparseInputLinear(self.p.init_dim, self.p.num_factors * self.p.gcn_dim)
        conv_ls = []
        for i in range(self.p.gcn_layer):
            conv = DisenLayer(self.edge_index, self.edge_type, self.p.init_dim, self.p.gcn_dim, num_rel,
                               act=self.act, params=self.p, head_num=self.p.head_num)
            self.add_module('conv_{}'.format(i), conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls
        if self.p.mi_train:
            if self.p.mi_method == 'club_b':
                num_dis = int((self.p.num_factors) * (self.p.num_factors - 1) / 2)
                # print(num_dis)
                self.mi_Discs = nn.ModuleList([CLUBSample(self.p.gcn_dim, self.p.gcn_dim, self.p.gcn_dim) for fac in range(num_dis)])
            elif self.p.mi_method == 'club_s':
                self.mi_Discs = nn.ModuleList([CLUBSample((fac + 1 ) * self.p.gcn_dim, self.p.gcn_dim, (fac + 1 ) * self.p.gcn_dim) for fac in range(self.p.num_factors - 1)])

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.rel_drop = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def lld_bst(self, sub, rel, drop1, mode='train'):
        x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim) # [N K F]
        r = self.init_rel
        for conv in self.conv_ls:
            x, r = conv(x, r, mode) # N K F
            if self.p.mi_drop:
                x = drop1(x)
            else:
                continue

        sub_emb = torch.index_select(x, 0, sub)
        lld_loss = 0.
        sub_emb = sub_emb.view(-1, self.p.gcn_dim * self.p.num_factors)
        if self.p.mi_method == 'club_s':
            for i in range(self.p.num_factors - 1):
                bnd = i + 1
                lld_loss += self.mi_Discs[i].learning_loss(sub_emb[:, :bnd * self.p.gcn_dim], sub_emb[:, bnd * self.p.gcn_dim : (bnd + 1) * self.p.gcn_dim])
        
        elif self.p.mi_method == 'club_b':
            cnt = 0
            for i in range(self.p.num_factors):
                for j in range( i + 1, self.p.num_factors):
                    lld_loss += self.mi_Discs[cnt].learning_loss(sub_emb[:, i * self.p.gcn_dim: (i + 1) * self.p.gcn_dim], sub_emb[:, j * self.p.gcn_dim: (j + 1) * self.p.gcn_dim])
                    cnt += 1
        return lld_loss

    def mi_cal(self, sub_emb):
        def loss_dependence_hisc(zdata_trn, ncaps, nhidden):
            loss_dep = torch.zeros(1).cuda()
            hH = (-1/nhidden)*torch.ones(nhidden, nhidden).cuda() + torch.eye(nhidden).cuda()
            kfactor = torch.zeros(ncaps, nhidden, nhidden).cuda()

            for mm in range(ncaps):
                data_temp = zdata_trn[:, mm * nhidden:(mm + 1) * nhidden]
                kfactor[mm, :, :] = torch.mm(data_temp.t(), data_temp)

            for mm in range(ncaps):
                for mn in range(mm + 1, ncaps):
                    mat1 = torch.mm(hH, kfactor[mm, :, :])
                    mat2 = torch.mm(hH, kfactor[mn, :, :])
                    mat3 = torch.mm(mat1, mat2)
                    teststat = torch.trace(mat3)

                    loss_dep = loss_dep + teststat
            return loss_dep

        def loss_dependence_club_s(sub_emb):
            mi_loss = 0.
            for i in range(self.p.num_factors - 1):
                bnd = i + 1
                mi_loss += self.mi_Discs[i](sub_emb[:, :bnd * self.p.gcn_dim], sub_emb[:, bnd * self.p.gcn_dim : (bnd + 1) * self.p.gcn_dim])
            return mi_loss

        def loss_dependence_club_b(sub_emb):
            mi_loss = 0.
            cnt = 0
            for i in range(self.p.num_factors):
                for j in range( i + 1, self.p.num_factors):
                    mi_loss += self.mi_Discs[cnt](sub_emb[:, i * self.p.gcn_dim: (i + 1) * self.p.gcn_dim], sub_emb[:, j * self.p.gcn_dim: (j + 1) * self.p.gcn_dim])
                    cnt += 1
            return mi_loss
        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                   torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        if self.p.mi_method == 'club_s':
            mi_loss = loss_dependence_club_s(sub_emb)
        elif self.p.mi_method == 'club_b':
            mi_loss = loss_dependence_club_b(sub_emb)
        elif self.p.mi_method == 'hisc':
            mi_loss = loss_dependence_hisc(sub_emb, self.p.num_factors, self.p.gcn_dim)
        elif self.p.mi_method == "dist":
            cor = 0.
            for i in range(self.p.num_factors):
                for j in range(i + 1, self.p.num_factors):
                    cor += DistanceCorrelation(sub_emb[:, i * self.p.gcn_dim: (i + 1) * self.p.gcn_dim], sub_emb[:, j * self.p.gcn_dim: (j + 1) * self.p.gcn_dim])
            return cor
        else:
            raise NotImplementedError

        return mi_loss

    def forward_base(self, sub, rel, drop1, drop2, mode):
        if not self.p.no_enc:
            x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim) # [N K F]
            r = self.init_rel
            for conv in self.conv_ls:
                x, r = conv(x, r, mode) # N K F
                x = drop1(x)
        else:
            x = self.init_embed
            r = self.init_rel
            x = drop1(x)
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(self.init_rel, 0, rel).repeat(1, self.p.num_factors)
        mi_loss = 0.
        sub_emb = sub_emb.view(-1, self.p.gcn_dim * self.p.num_factors)
        mi_loss = self.mi_cal(sub_emb)

        return sub_emb, rel_emb, x, mi_loss

    def test_base(self, sub, rel, drop1, drop2, mode):
        if not self.p.no_enc:
            x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim) # [N K F]
            r = self.init_rel
            for conv in self.conv_ls:
                x, r = conv(x, r, mode) # N K F
                x = drop1(x)
        else:
            x = self.init_embed.view(-1, self.p.num_factors, self.p.gcn_dim)
            r = self.init_rel
            x = drop1(x)
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(self.init_rel, 0, rel).repeat(1, self.p.num_factors)

        return sub_emb, rel_emb, x, 0.




class DisenKGAT_TransE(CapsuleBase):
    
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)
        self.rel_weight = self.conv_ls[-1].rel_weight  # DisenLayers
        gamma_init = torch.FloatTensor([self.p.init_gamma])
        if not self.p.fix_gamma:
            self.register_parameter('gamma', Parameter(gamma_init))

    def lld_best(self, sub, rel):
        return self.lld_bst(sub, rel, self.drop)

    def forward(self, sub, rel, neg_ents=None, mode='train'):
        if mode == 'train' and self.p.mi_train:
            sub_emb, rel_emb, all_ent, corr = self.forward_base(sub, rel, self.drop, self.drop, mode)  # all_ent is about memory
            sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        else:
            sub_emb, rel_emb, all_ent, corr = self.test_base(sub, rel, self.drop, self.drop, mode)

        rel_weight = torch.index_select(self.rel_weight, 0, rel) 
        rel_emb = rel_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        if self.p.score_method == 'dot_rel':
            sub_rel_emb = sub_emb * rel_weight
            rel_emb = rel_emb
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb])) # B K
        elif self.p.score_method == 'dot_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb])) # B K
        elif self.p.score_method == 'cat_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze()) # B K
        elif self.p.score_method == 'cat_sub':
            sub_rel_emb = sub_emb 
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze()) # B K
        elif self.p.score_method == 'learn':
            att_rel = torch.index_select(self.fc_att, 0, rel)
            attention = self.leakyrelu(att_rel) # [B K]
        attention = nn.Softmax(dim=-1)(attention)
        # calculate the score 
        obj_emb = sub_emb + rel_emb
        if self.p.gamma_method == 'ada':
            x = self.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=3).transpose(1, 2)
        elif self.p.gamma_method == 'norm':
            x2 = torch.sum(obj_emb * obj_emb, dim=-1)  #    x2.shape == [2048,3]
            y2 = torch.sum(all_ent * all_ent, dim=-1)   #   y2真正应该是[14541,3]
            xy = torch.einsum('bkf,nkf->bkn', [obj_emb, all_ent])
            x = self.gamma - (x2.unsqueeze(2) + y2.t() -  2 * xy)

        elif self.p.gamma_method == 'fix':
            x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=3).transpose(1, 2)
        # start to attention on prediction
        if self.p.score_order == 'before':
            x = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.sigmoid(x)
        elif self.p.score_order == 'after':
            x = torch.sigmoid(x)
            pred = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.clamp(pred, min=0., max=1.0)
        return pred, corr


class DisenKGAT_DistMult(CapsuleBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)
        self.rel_weight = self.conv_ls[-1].rel_weight

    def lld_best(self, sub, rel):
        return self.lld_bst(sub, rel, self.drop)

    def forward(self, sub, rel, neg_ents=None, mode='train'):
        if mode == 'train' and self.p.mi_train:
            sub_emb, rel_emb, all_ent, corr = self.forward_base(sub, rel, self.drop, self.drop, mode)
            sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        else:
            sub_emb, rel_emb, all_ent, corr = self.test_base(sub, rel, self.drop, self.drop, mode)
        rel_weight = torch.index_select(self.rel_weight, 0, rel) 
        rel_emb = rel_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        if self.p.score_method == 'dot_rel':
            sub_rel_emb = sub_emb * rel_weight
            rel_emb = rel_emb
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb])) # B K
        elif self.p.score_method == 'dot_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb])) # B K
        elif self.p.score_method == 'cat_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze()) # B K
        elif self.p.score_method == 'cat_sub':
            sub_rel_emb = sub_emb 
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze()) # B K
        elif self.p.score_method == 'learn':
            att_rel = torch.index_select(self.fc_att, 0, rel)
            attention = self.leakyrelu(att_rel) # [B K]
        attention = nn.Softmax(dim=-1)(attention)
        # calculate the score 
        obj_emb = sub_emb * rel_emb
        x = torch.einsum('bkf,nkf->bkn', [obj_emb, all_ent])
        x += self.bias.expand_as(x)
        # start to attention on prediction
        if self.p.score_order == 'before':
            x = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.sigmoid(x)
        elif self.p.score_order == 'after':
            x = torch.sigmoid(x)
            pred = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.clamp(pred, min=0., max=1.0)

        return pred, corr


class DisenKGAT_ConvE(CapsuleBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.embed_dim = self.p.embed_dim

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                       stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)
        if self.p.score_method.startswith('cat'):
            self.fc_a = nn.Linear(2 * self.p.gcn_dim, 1)
        elif self.p.score_method == 'learn':
            self.fc_att = get_param((2 * self.p.num_rel, self.p.num_factors))
        self.rel_weight = self.conv_ls[-1].rel_weight

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def lld_best(self, sub, rel):
        return self.lld_bst(sub, rel, self.hidden_drop)

    def forward(self, sub, rel, neg_ents=None, mode='train'):
        if mode == 'train' and self.p.mi_train:
            sub_emb, rel_emb, all_ent, corr = self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, mode)
            # sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        else:
            sub_emb, rel_emb, all_ent, corr = self.test_base(sub, rel, self.hidden_drop, self.feature_drop, mode)
        sub_emb = sub_emb.view(-1, self.p.gcn_dim)
        rel_emb = rel_emb.view(-1, self.p.gcn_dim)
        all_ent = all_ent.view(-1, self.p.num_factors, self.p.gcn_dim)

        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(-1, self.p.num_factors, self.p.gcn_dim)
        # start to calculate the attention
        rel_weight = torch.index_select(self.rel_weight, 0, rel)       # B K F
        rel_emb = rel_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim) # B K F
        if self.p.score_method == 'dot_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb])) # B K
        elif self.p.score_method == 'dot_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb])) # B K
        elif self.p.score_method == 'cat_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze()) # B K
        elif self.p.score_method == 'cat_sub':
            sub_rel_emb = sub_emb 
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze()) # B K
        elif self.p.score_method == 'learn':
            att_rel = torch.index_select(self.fc_att, 0, rel)
            attention = self.leakyrelu(att_rel) # [B K]
        attention = nn.Softmax(dim=-1)(attention)
        x = torch.einsum('bkf,nkf->bkn', [x, all_ent])
        x += self.bias.expand_as(x)

        if self.p.score_order == 'before':
            x = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.sigmoid(x)
        elif self.p.score_order == 'after':
            x = torch.sigmoid(x)
            pred = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.clamp(pred, min=0., max=1.0)
        return pred, corr


class DisenKGAT_InteractE(CapsuleBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.inp_drop = torch.nn.Dropout(self.p.iinp_drop)
        self.feature_map_drop = torch.nn.Dropout2d(self.p.ifeat_drop)
        self.hidden_drop = torch.nn.Dropout(self.p.ihid_drop)

        self.hidden_drop_gcn = torch.nn.Dropout(0)

        self.bn0 = torch.nn.BatchNorm2d(self.p.iperm)

        flat_sz_h = self.p.ik_h
        flat_sz_w = 2 * self.p.ik_w
        self.padding = 0

        self.bn1 = torch.nn.BatchNorm2d(self.p.inum_filt * self.p.iperm)
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.inum_filt * self.p.iperm

        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
        self.chequer_perm = self.get_chequer_perm()
        if self.p.score_method.startswith('cat'):
            self.fc_a = nn.Linear(2 * self.p.gcn_dim, 1)
        elif self.p.score_method == 'learn':
            self.fc_att = get_param((2 * self.p.num_rel, self.p.num_factors))
        self.rel_weight = self.conv_ls[-1].rel_weight
        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.register_parameter('conv_filt',
                                Parameter(torch.zeros(self.p.inum_filt, 1, self.p.iker_sz, self.p.iker_sz)))
        xavier_normal_(self.conv_filt)

    def circular_padding_chw(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def lld_best(self, sub, rel):
        return self.lld_bst(sub, rel, self.inp_drop)

    def forward(self, sub, rel, neg_ents=None, mode='train'):
        if mode == 'train' and self.p.mi_train:
            sub_emb, rel_emb, all_ent, corr = self.forward_base(sub, rel, self.inp_drop, self.hidden_drop_gcn, mode)
            # sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        else:
            sub_emb, rel_emb, all_ent, corr = self.test_base(sub, rel, self.inp_drop, self.hidden_drop_gcn, mode)
        sub_emb = sub_emb.view(-1, self.p.gcn_dim)
        rel_emb = rel_emb.view(-1, self.p.gcn_dim)
        all_ent = all_ent.view(-1, self.p.num_factors, self.p.gcn_dim)
        # sub: [B K F]  
        # rel: [B K F] 
        # all_ent: [N K F]
        comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.p.iperm, 2 * self.p.ik_w, self.p.ik_h))
        stack_inp = self.bn0(stack_inp)
        x = stack_inp
        x = self.circular_padding_chw(x, self.p.iker_sz // 2)
        x = F.conv2d(x, self.conv_filt.repeat(self.p.iperm, 1, 1, 1), padding=self.padding, groups=self.p.iperm)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x) # [B*K F]
        x = x.view(-1, self.p.num_factors, self.p.gcn_dim)
        # start to calculate the attention
        rel_weight = torch.index_select(self.rel_weight, 0, rel)       # B K F
        rel_emb = rel_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim) # B K F
        if self.p.score_method == 'dot_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb])) # B K
        elif self.p.score_method == 'dot_sub':
            sub_rel_emb = sub_emb
            attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [sub_rel_emb, rel_emb])) # B K
        elif self.p.score_method == 'cat_rel':
            sub_rel_emb = sub_emb * rel_weight
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze()) # B K
        elif self.p.score_method == 'cat_sub':
            sub_rel_emb = sub_emb 
            attention = self.leakyrelu(self.fc_a(torch.cat([sub_rel_emb, rel_emb], dim=2)).squeeze()) # B K
        elif self.p.score_method == 'learn':
            att_rel = torch.index_select(self.fc_att, 0, rel)
            attention = self.leakyrelu(att_rel) # [B K]
        attention = nn.Softmax(dim=-1)(attention)
        if self.p.strategy == 'one_to_n' or neg_ents is None:
            x = torch.einsum('bkf,nkf->bkn', [x, all_ent])
            x += self.bias.expand_as(x)
        else:
            x = torch.mul(x.unsqueeze(1), all_ent[neg_ents]).sum(dim=-1)
            x += self.bias[neg_ents]
        if self.p.score_order == 'before':
            x = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.sigmoid(x)
        elif self.p.score_order == 'after':
            x = torch.sigmoid(x)
            pred = torch.einsum('bk,bkn->bn', [attention, x])
            pred = torch.clamp(pred, min=0., max=1.0)
        return pred, corr

    def get_chequer_perm(self):
        """
        Function to generate the chequer permutation required for InteractE model

        Parameters
        ----------

        Returns
        -------

        """
        ent_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])
        rel_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])

        comb_idx = []
        for k in range(self.p.iperm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.p.ik_h):
                for j in range(self.p.ik_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm





def get_combined_results(left_results, right_results):
    results = {}
    count = float(left_results['count'])

    results['left_mr'] = round(left_results['mr'] / count, 5)
    results['left_mrr'] = round(left_results['mrr'] / count, 5)
    results['right_mr'] = round(right_results['mr'] / count, 5)
    results['right_mrr'] = round(right_results['mrr'] / count, 5)
    results['mr'] = round((left_results['mr'] + right_results['mr']) / (2 * count), 5)
    results['mrr'] = round((left_results['mrr'] + right_results['mrr']) / (2 * count), 5)

    for k in range(10):
        results['left_hits@{}'.format(k + 1)] = round(left_results['hits@{}'.format(k + 1)] / count, 5)
        results['right_hits@{}'.format(k + 1)] = round(right_results['hits@{}'.format(k + 1)] / count, 5)
        results['hits@{}'.format(k + 1)] = round(
            (left_results['hits@{}'.format(k + 1)] + right_results['hits@{}'.format(k + 1)]) / (2 * count), 5)
    return results

def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    xavier_normal_(param.data)
    return param

def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

def conj(a):
    a[..., 1] = -a[..., 1]
    return a

def cconv(a, b):
    return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True
# sys.path.append('./')


