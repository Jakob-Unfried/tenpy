r"""A module to handle charge conservation in tensor networks.

A detailed introduction (including notations) can be found in :doc:`../IntroNpc`.

This module `np_conserved` implements an class :class:`Array`
designed to make use of charge conservation in tensor networks.
The idea is that the `Array` class is used in a fashion very similar to
the `numpy.ndarray`, e.g you can call the functions :func:`tensordot` or :func:`svd`
(of this module) on them.
The structure of the algorithms (as DMRG) is thus the same as with basic numpy ndarrays.

Internally, an :class:`Array` saves charge meta data to keep track of blocks which are nonzero.
All possible operations (e.g. tensordot, svd, ...) on such arrays preserve the total charge
structure. In addition, these operations make use of the charges to figure out which of the blocks
it hase to use/combine - this is the basis for the speed-up.


See also
--------
tenpy.linalg.charges : Implementation of :class:`~tenpy.linalg.charge.ChargeInfo`
    and :class:`~tenpy.linalg.charge.LegCharge` with additional documentation.


.. todo ::
   usage and examples, Routine listing
   update ``from charges import``
   write example section
"""
# Examples
# --------
# >>> import numpy as np
# >>> import tenpy.linalg.np_conserved as npc
# >>> Sz = np.array([[0., 1.], [1., 0.]])
# >>> Sz_c = npc.Array.from_ndarray_trivial(Sz)  # convert to npc array with trivial charge
# >>> Sz_c
# <npc.Array shape=(2, 2)>
# >>> sx = npc.ndarray.from_ndarray([[0., 1.], [1., 0.]])  # trivial charge conservation
# >>> b = npc.ndarray.from_ndarray([[0., 1.], [1., 0.]])  # trivial charge conservation
# >>>
# >>> print a[0, -1]
# >>> c = npc.tensordot(a, b, axes=([1], [0]))

from __future__ import division

import numpy as np
import copy as copy_
import warnings
import itertools

# import public API from charges
from .charges import (QDTYPE, ChargeInfo, LegCharge, LegPipe,
                      reverse_sort_perm)
from . import charges   # for private functions

from ..tools.math import toiterable

"""A cutoff to ignore machine precision rounding errors when determining charges"""
QCUTOFF = np.finfo(np.float64).eps * 10


class Array(object):
    r"""A multidimensional array (=tensor) for using charge conservation.

    An `Array` represents a multi-dimensional tensor,
    together with the charge structure of its legs (for abelian charges).
    Further information can be found in :doc:`../IntroNpc`.

    The default :meth:`__init__` (i.e. ``Array(...)``) does not insert any data,
    and thus yields an Array 'full' of zeros, equivalent to :func:`zeros()`.
    Further, new arrays can be created with one of :meth:`from_ndarray_trivial`,
    :meth:`from_ndarray`, or :meth:`from_npfunc`, and of course by copying/tensordot/svd etc.

    Parameters
    ----------
    chargeinfo : :class:`~tenpy.linalg.charges.ChargeInfo`
        the nature of the charge, used as self.chinfo.
    legs : list of :class:`~tenpy.linalg.charges.LegCharge`
        the leg charges for each of the legs.
    dtype : type or string
        the data type of the array entries. Defaults to np.float64.


    Attributes
    ----------
    shape : tuple(int)
        the number of indices for each of the legs
    dtype : np.dtype
        the data type of the entries
    chinfo : :class:`~tenpy.linalg.charges.ChargeInfo`
        the nature of the charge
    qtotal : charge values
        the total charge of the tensor.
    legs : list of :class:`~tenpy.linalg.charges.LegCharge`
        the leg charges for each of the legs.
    labels : dict (string -> int)
        labels for the different legs
    _data : list of arrays
        the actual entries of the tensor
    _qdata : 2D array (len(_data), rank)
        for each of the _data entries the qind of the different legs.
    _qdata_sorted : Bool
        whether self._qdata is lexsorted. Defaults to `True`,
        but *must* be set to `False` by algorithms changing _qdata.

    Notes
    -----
    The Array

    .. todo ::

        - Somehow, including `rank` to the list of attributes breaks
          the sphinx build for this class...
        - What about size and stored_blocks?
        - Methods section
        - should _qdata be of type np.intp ?
    """
    def __init__(self, chargeinfo, legcharges, dtype=np.float64, qtotal=None):
        """see help(self)"""
        self.chinfo = chargeinfo
        self.legs = list(legcharges)
        self._set_shape()
        self.dtype = np.dtype(dtype)
        self.qtotal = self.chinfo.make_valid(qtotal)
        self.labels = {}
        self._data = []
        self._qdata = np.empty((0, self.rank), QDTYPE)
        self._qdata_sorted = True
        self.test_sanity()

    def copy(self, deep=False):
        """Return a (deep or shallow) copy of self.

        **Both** deep and shallow copies will share ``chinfo`` and the `LegCharges` in ``legs``.
        In contrast to a deep copy, the shallow copy will also share the tensor entries.
        """
        if deep:
            cp = copy_.deepcopy(self)
        else:
            cp = copy_.copy(self)
            # some things should be copied even for shallow copies
            cp._set_shape()
            cp.qtotal = cp.qtotal.copy()
        # even deep copies can share chargeinfo and legs
        cp.chinfo = self.chinfo
        cp.legs = self.legs[:]
        return cp

    @classmethod
    def from_ndarray_trivial(cls, data_flat, dtype=np.float64):
        """convert a flat numpy ndarray to an Array with trivial charge conservation.

        Parameters
        ----------
        data_flat : array_like
            the data to be converted to a Array
        dtype : type | string
            the data type of the array entries. Defaults to ``np.float64``.

        Returns
        -------
        res : :class:`Array`
            an Array with data of data_flat
        """
        data_flat = np.array(data_flat, dtype)
        chinfo = ChargeInfo()
        legs = [LegCharge.from_trivial(s, chinfo) for s in data_flat.shape]
        res = cls(chinfo, legs, dtype)
        res._data = [data_flat]
        res._qdata = np.zeros((1, res.rank), QDTYPE)
        res._qdata_sorted = True
        res.test_sanity()
        return res

    @classmethod
    def from_ndarray(cls, data_flat, chargeinfo, legcharges, dtype=np.float64, qtotal=None,
                     cutoff=0.):
        """convert a flat (numpy) ndarray to an Array.

        Parameters
        ----------
        data_flat : array_like
            the flat ndarray which should be converted to a npc `Array`.
            The shape has to be compatible with legcharges.
        chargeinfo : ChargeInfo
            the nature of the charge
        legcharges : list of LegCharge
            a LegCharge for each of the legs.
        dtype : type | string
            the data type of the array entries. Defaults to np.float64.
        qtotal : None | charges
            the total charge of the new array.
        cutoff : float
            A cutoff to exclude rounding errors of machine precision. Defaults to QCUTOFF.

        Returns
        -------
        res : :class:`Array`
            an Array with data of `data_flat`.

        See also
        --------
        detect_ndarray_qtotal : used to detect the total charge of the flat array.
        """
        if cutoff is None:
            cutoff = QCUTOFF
        res = cls(chargeinfo, legcharges, dtype, qtotal)  # without any data
        if res.shape != data_flat.shape:
            raise ValueError("Incompatible shapes: legcharges {0!s} vs flat {1!s} ".format(
                res.shape, data_flat.shape))
        if qtotal is None:
            res.qtotal = qtotal = res.detect_ndarray_qtotal(data_flat, cutoff)
        data = []
        qdata = []
        for qindices in res._iter_all_blocks():
            sl = res._get_block_slices(qindices)
            if np.all(res._get_block_charge(qindices) == qtotal):
                data.append(np.array(data_flat[sl], dtype=res.dtype))   # copy data
                qdata.append(qindices)
            elif np.any(np.abs(data_flat[sl]) > cutoff):
                warnings.warn("flat array has non-zero entries in blocks incompatible with charge")
        res._data = data
        if len(qdata) == 0:
            res._qdata = np.empty((0, res.rank), QDTYPE)
        else:
            res._qdata = np.array(qdata, dtype=QDTYPE)
        res._qdata_sorted = True
        res.test_sanity()
        return res

    @classmethod
    def from_func(cls, func, chargeinfo, legcharges, dtype=np.float64, qtotal=None,
                  func_args=(), func_kwargs={}, shape_kw=None):
        """Create an Array from a numpy func.

        This function creates an array and fills the blocks *compatible* with the charges
        using `func`, where `func` is a function returning a `array_like` when given a shape,
        e.g. one of ``np.ones`` or ``np.random.standard_normal``.

        Parameters
        ----------
        func : callable
            a function-like object which is called to generate the data blocks.
            We expect that `func` returns a flat array of the given `shape` convertible to `dtype`.
            If no `shape_kw` is given, it is called like ``func(shape, *fargs, **fkwargs)``,
            otherwise as ``func(*fargs, `shape_kw`=shape, **fkwargs)``.
            `shape` is a tuple of int.
        chargeinfo : ChargeInfo
            the nature of the charge
        legcharges : list of LegCharge
            a LegCharge for each of the legs.
        dtype : type | string
            the data type of the output entries. Defaults to np.float64.
            Note that this argument is not given to func, but rather a type conversion
            is performed afterwards. You might want to set a `dtype` in `func_kwargs` as well.
        qtotal : None | charges
            the total charge of the new array. Defaults to charge 0.
        func_args : iterable
            additional arguments given to `func`
        func_kwargs : dict
            additional keyword arguments given to `func`
        shape_kw : None | str
            If given, the keyword with which shape is given to `func`.

        Returns
        -------
        res : :class:`Array`
            an Array with blocks filled using `func`.
        """
        res = cls(chargeinfo, legcharges, dtype, qtotal)  # without any data yet.
        data = []
        qdata = []
        for qindices in res._iter_all_blocks():
            if np.any(res._get_block_charge(qindices) != res.qtotal):
                continue
            shape = res._get_block_shape(qindices)
            if shape_kw is None:
                block = func(shape, *func_args, **func_kwargs)
            else:
                kws = func_kwargs.copy()
                kws[shape_kw] = shape
                block = func(*func_args, **kws)
            block = np.asarray(block, dtype=res.dtype)
            data.append(block)
            qdata.append(qindices)
        res._data = data
        if len(qdata) == 0:
            res._qdata = np.empty((0, res.rank), QDTYPE)
        else:
            res._qdata = np.array(qdata, dtype=QDTYPE)
        res._qdata_sorted = True  # _iter_all_blocks is in lexiographic order
        res.test_sanity()
        return res

    def zeros_like(self):
        """return a shallow copy of self with only zeros as entries, containing no `_data`"""
        res = self.copy(deep=False)
        res._data = []
        res._qdata = np.empty((0, res.rank), QDTYPE)
        res._qdata_sorted = True
        return res

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        if self.shape != tuple([lc.ind_len for lc in self.legs]):
            raise ValueError("shape mismatch with LegCharges\n self.shape={0!s} != {1!s}".format(
                self.shape, tuple([lc.ind_len for lc in self.legs])))
        if any([self.dtype != d.dtype for d in self._data]):
            raise ValueError("wrong dtype: {0!s} vs\n {1!s}".format(
                self.dtype, [self.dtype != d.dtype for d in self._data]))
        for l in self.legs:
            l.test_sanity()
            if l.chinfo != self.chinfo:
                raise ValueError("leg has different ChargeInfo:\n{0!s}\n vs {1!s}".format(
                    l.chinfo, self.chinfo))
        if self._qdata.shape != (self.stored_blocks, self.rank):
            raise ValueError("_qdata shape wrong")
        if np.any(self._qdata < 0) or np.any(self._qdata >= [l.block_number for l in self.legs]):
            raise ValueError("invalid qind in _qdata")
        if self._qdata_sorted:
            perm = np.lexsort(self._qdata.T)
            if np.any(perm != np.arange(len(perm))):
                raise ValueError("_qdata_sorted == True, but _qdata is not sorted")

    # properties ==============================================================

    @property
    def rank(self):
        """the number of legs"""
        return len(self.shape)

    @property
    def size(self):
        """the number of dtype-objects stored"""
        return np.sum([t.size for t in self._data], dtype=np.int_)

    @property
    def stored_blocks(self):
        """the number of (non-zero) blocks stored in self._data"""
        return len(self._data)

    # accessing entries =======================================================

    def to_ndarray(self):
        """convert self to a dense numpy ndarray."""
        res = np.zeros(self.shape, self.dtype)
        for block, slices, _, _ in self:  # that's elegant! :)
            res[slices] = block
        return res

    def __iter__(self):
        """Allow to iterate over the non-zero blocks, giving all `_data`.

        Yields
        ------
        block : ndarray
            the actual entries of a charge block
        blockslices : tuple of slices
            a slice giving the range of the block in the original tensor for each of the legs
        charges : list of charges
            the charge value(s) for each of the legs
        qdat : ndarray
            the qind for each of the legs
        """
        for block, qdat in itertools.izip(self._data, self._qdata):
            qind = [l.qind[qi] for (qi, l) in itertools.izip(qdat, self.legs)]
            blockslices = tuple([slice(qi[0], qi[1]) for qi in qind])
            qs = [qi[2:] for qi in qind]
            yield block, blockslices, qs, qdat

    def __getitem__(self, inds):
        """acces entries with ``self[inds]``

        Parameters
        ----------
        inds : tuple
            A tuple specifying the `index` for each leg.
            An ``Ellipsis`` (written as ``...``) replaces ``slice(None)`` for missing axes.
            For a single `index`, we currently support:

            - A single integer, choosing an index of the axis,
              reducing the dimension of the resulting array.
            - A ``slice(None)`` specifying the complete axis.
            - A ``slice``, which acts like a `mask` in :meth:`iproject`.
            - A 1D array(bool): acts like a `mask` in :meth:`iproject`.
            - A 1D array(int) of ``len < leg.index_len``: acts like a `mask` in :meth:`iproject`.

        Returns
        -------
        res : `dtype`
            only returned, if a single integer is given for all legs.
            It is the entry specified by `inds`, giving ``0.`` for non-saved blocks.
        or
        sliced : :class:`Array`
            a copy with some of the data removed by :meth:`take_slice` and/or :meth:`project`

        Notes
        -----
        ``self[i]`` is equivalent to ``self[i, ...]``.
        ``self[i, ..., j]`` is syntactic sugar for ``self[(i, Ellipsis, i2)]``

        Raises
        ------
        IndexError
            If the number of indices is too large, or
            if an index is out of range.

        """
        int_only, inds = self._pre_indexing(inds)
        if int_only:
            pos = np.array([l.get_qindex(i) for i, l in zip(inds, self.legs)])
            block = self._get_block(pos[:, 0])
            if block is None:
                return self.dtype.type(0)
            else:
                return block[tuple(pos[:, 1])]
        return self._advanced_getitem(inds)

    def __setitem__(self, inds, val):
        """``self[i1, i2, ... ,irank] = val`` sets value of corresponding entry"""
        int_only, inds = self._pre_indexing(inds)
        if int_only:
            pos = np.array([l.get_qindex(i) for i, l in zip(inds, self.legs)])
            block = self._get_block(pos[:, 0], insert=True, raise_incomp_q=True)
            block[tuple(pos[:, 1])] = val
            return
        raise NotImplementedError("slicing for setitem not implemented")  # TODO

    def take_slice(self, indices, axes):
        """Return a copy of self fixing `indices` along one or multiple `axes`.

        For a rank-4 Array ``A.take_slice([i, j], [1,2])`` is equivalent to ``A[:, i, j, :]``.

        Parameters
        ----------
        indices : (iterable of) int
            the (flat) index for each of the legs specified by `axes`
        axes : (iterable of) str/int
            leg labels or indices to specify the legs for which the indices are given.

        Returns
        -------
        slided_self : :class:`Array`
            a copy of self, equivalent to taking slices with indices inserted in axes.
        """
        return self._take_slice_shallow(indices, axes, write=False)

    # handling of charges =====================================================

    def detect_ndarray_qtotal(self, flat_array, cutoff=None):
        """ Returns the total charge of first non-zero sector found in `a`.

        Charge information is taken from self.
        If you have only the charge data, create an empty Array(chinf, legcharges).

        Parameters
        ----------
        flat_array : array
            the flat numpy array from which you want to detect the charges
        chinfo : ChargeInfo
            the nature of the charge
        legcharges : list of LegCharge
            for each leg the LegCharge
        cutoff : float
            defaults to QCUTOFF

        Returns
        -------
        qtotal : charge
            the total charge fo the first non-zero (i.e. > cutoff) charge block
        """
        if cutoff is None:
            cutoff = QCUTOFF
        for qindices in self._iter_all_blocks():
            sl = self._get_block_slices(qindices)
            if np.any(np.abs(flat_array[sl]) > cutoff):
                return self._get_block_charge(qindices)
        warnings.warn("can't detect total charge: no entry larger than cutoff. Return 0 charge.")
        return self.chinfo.make_valid()

    def gauge_total_charge(self, leg, newqtotal=None):
        """changes the total charge of an Array `A` inplace by adjusting the charge on a certain leg.

        The total charge is given by finding a nonzero entry [i1, i2, ...] and calculating::

            qtotal = sum([l.qconj[i] * l.qind[il] for il in zip([i1,i2,...], self.legs)])

        Thus, the total charge can be changed by redefining the leg charge of a given leg.
        This is exaclty what this function does.

        Parameters
        ----------
        leg : int or string
            the new leg (index or label), for which the charge is changed
        newqtotal : charge values, defaults to 0
            the new total charge
        """
        leg = self.get_leg_index(leg)
        newqtotal = self.chinfo.make_valid(newqtotal)  # converts to array, default zero
        chdiff = newqtotal - self.qtotal
        newleg = copy_.copy(self.legs[leg])  # shallow copy of the LegCharge
        newleg.qind = newleg.qind.copy()
        newleg.qind[:, 2:] = self.chinfo.make_valid(newleg.qind[:, 2:] + newleg.qconj * chdiff)
        self.legs[leg] = newleg
        self.qtotal = newqtotal

    def is_completely_blocked(self):
        """returns bool wheter all legs are blocked by charge"""
        return all([l.is_blocked() for l in self.legs])

    def sort_legcharge(self, sort=True, bunch=True):
        """Return a copy with one ore all legs sorted by charges.

        Sort/bunch one or multiple of the LegCharges.
        Note that legs which are sorted *and* bunched
        are guaranteed to be blocked by charge.

        Parameters
        ----------
        sort : True | False | list of {True, False, perm}
            A single bool holds for all legs, default=True.
            Else, `sort` should contain one entry for each leg, with a bool for sort/don't sort,
            or a 1D array perm for a given permuation to apply to a leg.
        bunch : True | False | list of {True, False}
            A single bool holds for all legs, default=True.
            whether or not to bunch at each leg, i.e. combine contiguous blocks with equal charges.

        Returns
        -------
        perm : tuple of 1D arrays
            the permutation applied to each of the legs.
            cp.to_ndarray() = self.to_ndarray(perm)
        result : Array
            a shallow copy of self, with legs sorted/bunched
        """
        if sort is False or sort is True:  # ``sort in [False, True]`` doesn't work...
            sort = [sort]*self.rank
        if bunch is False or bunch is True:
            bunch = [bunch]*self.rank
        if not len(sort) == len(bunch) == self.rank:
            raise ValueError("Wrong len for bunch or sort")
        cp = self.copy(deep=False)
        cp._qdata = cp._qdata.copy(self)
        for li in xrange(self.rank):
            if sort[li] is not False:
                if sort[li] is True:
                    p_flat, p_qind, newleg = cp.legs[li].sort(bunch=False)
                    sort[li] = p_flat
                    cp.legs[li] = newleg
                else:
                    # TODO: catch exception when something like self.permute(axis=...)
                    # is implemented for general permutations
                    # try:
                    p_qind = charges._perm_qind_from_perm_flat(sort[li], self.legs[li])
                    # except ValueError:
                    #     cp = cp.ipermute(sort[li], axes=[li]) # implement...
                    #     continue
                cp._perm_qind(p_qind, li)
            else:
                sort[li] = np.arange(cp.shape[li])
        if any(bunch):
            cp = cp._bunch(bunch)  # bunch does not permute...
        return tuple(sort), cp

    def sort_qdata(self):
        """(lex)sort ``self._qdata``. In place.

        Lexsort ``self._qdata`` and ``self._data`` and set ``self._qdata_sorted = True``.
        """
        if self._qdata_sorted:
            return
        if len(self._qdata) < 2:
            self._qdata_sorted = True
            return
        perm = np.lexsort(self._qdata.T)
        self._qdata = self._qdata[perm, :]
        self._data = [self._data[p] for p in perm]
        self._qdata_sorted = True

    # data manipulation =======================================================

    def astype(self, dtype):
        """Return (deep) copy with new dtype, upcasting all blocks in ``_data``.

        Parameters
        ----------
        dtype : convertible to a np.dtype
            the new data type.
            If None, deduce the new dtype as common type of ``self._data``.

        Returns
        -------
        copy : :class:`Array`
            deep copy of self with new dtype
        """
        cp = self.copy(deep=False)  # manual deep copy: don't copy every block twice
        cp._qdata = cp._qdata.copy()
        if dtype is None:
            dtype = np.common_dtype(*self._data)
        cp.dtype = np.dtype(dtype)
        cp._data = [d.astype(self.dtype, copy=True) for d in self._data]
        return cp

    def iproject(self, mask, axes):
        """Applying masks to one or multiple axes. In place.

        This function is similar as `np.compress` with boolean arrays
        For each specified axis, a boolean 1D array `mask` can be given,
        which chooses the indices to keep.

        Parameters
        ----------
        mask : (list of) 1D array(bool|int)
            for each axis specified by `axes` a mask, which indices of the axes should be kept.
            If `mask` is a bool array, keep the indices where `mask` is True.
            If `mask` is an int array, keep the indices listed in the mask, *ignoring* the
            order or multiplicity.
        axes : (list of) int | string
            The `i`th entry in this list specifies the axis for the `i`th entry of `mask`,
            either as an int, or with a leg label.
            If axes is just a single int/string, specify just one mask.

        Returns
        -------
        map_qind : list of 1D arrays
            the mapping of qindices for each of the specified axes.
        """
        axes = self.get_leg_indices(toiterable(axes))
        mask = [np.asarray(m) for m in toiterable(mask)]
        if len(axes) != len(mask):
            raise ValueError("len(axes) != len(mask)")
        if len(axes) == 0:
            return  # nothing to do.
        for i, m in enumerate(mask):
            # convert integer masks to bool masks
            if m.dtype != np.bool_:
                mask[i] = np.zeros(self.shape[axes[i]], dtype=np.bool_)
                np.put(mask[i], m, True)
        block_masks = []
        proj_data = np.arange(self.stored_blocks)
        map_qind = []
        for m, a in zip(mask, axes):
            l = self.legs[a]
            m_qind, bm, self.legs[a] = l.project(m)
            map_qind.append(m_qind)
            block_masks.append(bm)
            q = self._qdata[:, a] = m_qind[self._qdata[:, a]]
            piv = (q >= 0)
            self._qdata = self._qdata[piv]
            # self._qdata_sorted is preserved
            proj_data = proj_data[piv]
        self._set_shape()
        # finally project out the blocks
        data = []
        for i, iold in enumerate(proj_data):
            block = self._data[iold]
            for m, a in zip(block_masks, axes):
                block = np.compress(m[self._qdata[i, a]], block, axis=a)
            data.append(block)
        self._data = data
        return map_qind

    def ipermute(self, perm, axis):
        """apply a permutation in the indices of an axis. In place.

        Similar as np.take with a 1D array.

        Parameters
        ----------
        perm : array_like 1D int
            The permutation which should be applied to the leg given by `axis`
        axis : str | int
            a leg label or index.

        .. todo ::
            not implemented
        """
        # permute legcharge, then do something like:
        # for i in xrange(len(perm)): cp[p[i], ...] = self[i, ...]
        # this is probably terribly slow but still fairly efficient.
        # to optimize it: figure out which qind are actually split?
        raise NotImplementedError() # TODO

    def itranspose(self, axes=None):
        """Transpose axes like `np.transpose`. In place.

        Parameters
        ----------
        axes: iterable (int|string), len ``rank`` | None
            the new order of the axes. By default (None), reverse axes.
        """
        if axes is None:
            axes = tuple(reversed(xrange(self.rank)))
        else:
            axes = tuple(self.get_leg_indices(axes))
            if len(axes) != self.rank or len(set(axes)) != self.rank:
                raise ValueError("axes has wrong length: " + str(axes))
        axes_arr = np.array(axes)
        self.legs = [self.legs[a] for a in axes]
        self._set_shape()
        labs = self.get_leg_labels()
        self.set_leg_labels([labs[a] for a in axes])
        self._qdata = self._qdata[:, axes_arr]
        self._qdata_sorted = False
        self._data = [np.transpose(block, axes) for block in self._data]

    def transpose(self, axes=None):
        """Like :meth:`itranspose`, but on a deep copy."""
        cp = self.copy(deep=True)
        cp.itranspose(axes)
        return cp

    # labels ==================================================================

    def get_leg_index(self, label):
        """translate a leg-index or leg-label to a leg-index.

        Parameters
        ----------
        label : int | string
            eather the leg-index directly or a label (string) set before.

        Returns
        -------
        leg_index : int
            the index of the label

        See also
        --------
        get_leg_indices : calls get_leg_index for a list of labels
        set_leg_labels : set the labels of different legs.
        """
        res = self.labels.get(label, label)
        if res > self.rank:
            raise ValueError("axis {0:d} out of rank {1:d}".format(res, self.rank))
        elif res < 0:
            res += self.rank
        return res

    def get_leg_indices(self, labels):
        """Translate a list of leg-indices or leg-labels to leg indices.

        Parameters
        ----------
        labels : iterable of string/int
            The leg-labels (or directly indices) to be translated in leg-indices

        Returns
        -------
        leg_indices : list of int
            the translated labels.

        See also
        --------
        get_leg_index : used to translate each of the single entries.
        set_leg_labels : set the labels of different legs.
        """
        return [self.get_leg_index(l) for l in labels]

    def set_leg_labels(self, labels):
        """Return labels for the legs.

        Introduction to leg labeling can be found in :doc:`../IntroNpc`.

        Parameters
        ----------
        labels : iterable (strings | None), len=self.rank
            One label for each of the legs.
            An entry can be None for an anonymous leg.

        See also
        --------
        get_leg: translate the labels to indices
        get_legs: calls get_legs for an iterable of labels
        """
        if len(labels) != self.rank:
            raise ValueError("Need one leg label for each of the legs.")
        self.labels = {}
        for i, l in enumerate(labels):
            if l is not None:
                self.labels[l] = i

    def get_leg_labels(self):
        """Return tuple of the leg labels, with `None` for anonymous legs."""
        lb = [None] * self.rank
        for k, v in self.labels.iteritems():
            lb[v] = k
        return tuple(lb)

    # string output ===========================================================
    def __repr__(self):
        return "<npc.array shape={0!s} charge={1!s} labels={2!s}>".format(
            self.shape, self.chinfo, self.get_leg_labels())

    def __str__(self):
        res = "\n".join([repr(self)[:-1], str(self.to_ndarray()), ">"])
        return res

    def sparse_stats(self):
        """Returns a string detailing the sparse statistics"""
        total = np.prod(self.shape)
        if total is 0:
            return "Array without entries, one axis is empty."
        nblocks = self.stored_blocks
        stored = self.size
        nonzero = np.sum([np.count_nonzero(t) for t in self._data], dtype=np.int_)
        bs = np.array([t.s for t in self._data], dtype=np.float)
        if nblocks > 0:
            bs1 = (np.sum(bs**0.5)/nblocks)**2
            bs2 = np.sum(bs)/nblocks
            bs3 = (np.sum(bs**2.0)/nblocks)**0.5
            captsparse = float(nonzero) / stored
        else:
            captsparse = 1.
            bs1, bs2, bs3 = 0, 0, 0
        res = "{nonzero:d} of {total:d} entries (={nztotal:g}) nonzero,\n" \
            "stored in {nblocks:d} blocks with {stored:d} entries.\n" \
            "Captured sparsity: {captsparse:g}\n"  \
            "Effective block sizes (second entry=mean): [{bs1:.2f}, {bs2:.2f}, {bs3:.2f}]"

        return res.format(nonzero=nonzero, total=total, nztotal=nonzero/total, nblocks=nblocks,
                          stored=stored, captspares=captsparse, bs1=bs1, bs2=bs2, bs3=bs3)
    # private functions =======================================================

    def _set_shape(self):
        """deduce self.shape from self.legs"""
        self.shape = tuple([lc.ind_len for lc in self.legs])

    def _iter_all_blocks(self):
        """generator to iterate over all combinations of qindices in lexiographic order.

        Yields
        ------
        qindices : tuple of int
            a qindex for each of the legs
        """
        for block_inds in itertools.product(*[xrange(l.block_number)
                                              for l in reversed(self.legs)]):
            # loop over all charge sectors in lex order (last leg most siginificant)
            yield tuple(block_inds[::-1])   # back to legs in correct order

    def _get_block_charge(self, qindices):
        """returns the charge of a block selected by `qindices`

        The charge of a single block is defined as ::

            qtotal = sum_{legs l} legs[l].qind[qindices[l], 2:] * legs[l].qconj() modulo qmod
        """
        q = np.sum([l.qind[qi, 2:]*l.qconj for l, qi in itertools.izip(self.legs, qindices)],
                   axis=0)
        return self.chinfo.make_valid(q)

    def _get_block_slices(self, qindices):
        """returns tuple of slices for a block selected by `qindices`"""
        return tuple([slice(l.qind[j, 0], l.qind[j, 1])
                      for l, j in itertools.izip(self.legs, qindices)])

    def _get_block_shape(self, qindices):
        """return shape for the block given by qindices"""
        return tuple([(l.qind[qi, 1] - l.qind[qi, 0]) for l, qi in
                      itertools.izip(self.legs, qindices)])

    def _get_block(self, qindices, insert=False, raise_incomp_q=False):
        """return the ndarray in ``_data`` representing the block corresponding to `qindices`.

        Parameters
        ----------
        qindices : 1D array of QDTYPE
            the qindices, for which we need to look in _qdata
        insert : bool
            If True, insert a new (zero) block, if `qindices` is not existent in ``self._data``.
            Else: just return ``None`` in that case.
        raise_incomp_q : bool
            Raise an IndexError if the charge is incompatible.

        Returns
        -------
        block: ndarray
            the block in ``_data`` corresponding to qindices
            If `insert`=False and there is not block with qindices, return ``False``

        Raises
        ------
        IndexError
            If qindices are incompatible with charge and `raise_incomp_q`
        """
        if not np.all(self._get_block_charge(qindices) == self.qtotal):
            if raise_incomp_q:
                raise IndexError("trying to get block for qindices incompatible with charges")
            return None
        # find qindices in self._qdata
        match = np.argwhere(np.all(self._qdata == qindices, axis=1))[:, 0]
        if len(match) == 0:
            if insert:
                res = np.zeros(self._get_block_shape(qindices), dtype=self.dtype)
                self._data.append(res)
                self._qdata = self._qdata.append(qindices)
                self._qdata_sorted = False
                return res
            else:
                return None
        return self._data[match[0]]

    def _bunch(self, bunch_legs):
        """Return copy and bunch the qind for one or multiple legs

        Parameters
        ----------
        bunch : list of {True, False}
            one entry for each leg, whether the leg should be bunched.

        See also
        --------
        sort_legcharge: public API calling this function.
        """
        cp = self.copy(deep=False)
        # lists for each leg:
        new_to_old_idx = [None]*cp.rank     # the `idx` returned by cp.legs[li].bunch()
        map_qindex = [None]*cp.rank         # array mapping old qindex to new qindex, such that
        # new_leg.qind[m_qindex[i]] == old_leg.qind[i]  # (except the second column entry)
        bunch_qindex = [None]*cp.rank       # bool array wheter the *new* qind was bunched
        for li, bunch in enumerate(bunch_legs):
            idx, new_leg = cp.legs[li].bunch()
            cp.legs[li] = new_leg
            new_to_old_idx[li] = idx
            # generate entries in map_qindex and bunch_qdindex
            idx = np.append(idx, [self.shape[li]])
            m_qindex = []
            bunch_qindex[li] = b_qindex = np.empty(idx.shape, dtype=np.bool_)
            for inew in xrange(len(idx)-1):
                old_blocks = idx[inew+1] - idx[inew]
                m_qindex.append([inew]*old_blocks)
                b_qindex[inew] = (old_blocks > 1)
            map_qindex[li] = np.concatenate(m_qindex, axis=0)

        # now map _data and _qdata
        bunched_blocks = {}     # new qindices -> index in new _data
        new_data = []
        new_qdata = []
        for old_block, old_qindices in itertools.izip(self._data, self._qdata):
            new_qindices = tuple([m[qi] for m, qi in itertools.izip(map_qindex, old_qindices)])
            bunch = any([b[qi] for b, qi in itertools.izip(bunch_qindex, new_qindices)])
            if bunch:
                if new_qindices not in bunched_blocks:
                    # create enlarged block
                    bunched_blocks[new_qindices] = len(new_data)
                    # cp has new legs and thus gives the new shape
                    new_block = np.zeros(cp._get_block_shape(new_qindices), dtype=cp.dtype)
                    new_data.append(new_block)
                    new_qdata.append(new_qindices)
                else:
                    new_block = new_data[bunched_blocks[new_qindices]]
                # figure out where to insert the in the new bunched_blocks
                old_slbeg = [l.qind[qi, 0] for l, qi in itertools.izip(self.legs, old_qindices)]
                new_slbeg = [l.qind[qi, 0] for l, qi in itertools.izip(cp.legs, new_qindices)]
                slbeg = [(o-n) for o, n in itertools.izip(old_slbeg, new_slbeg)]
                sl = [slice(beg, beg+l) for beg, l in itertools.izip(slbeg, old_block.shape)]
                # insert the old block into larger new block
                new_block[tuple(sl)] = old_block
            else:
                # just copy the old block
                new_data.append(old_block.copy())
                new_qdata.append(new_qindices)
        cp._data = new_data
        cp._qdata = np.array(new_qdata, dtype=QDTYPE)
        cp._qsorted = False
        return cp

    def _perm_qind(self, p_qind, leg):
        """Apply a permutation `p_qind` of the qindices in leg `leg` to _qdata. In place."""
        # entry ``b`` of of old old._qdata[:, leg] refers to old ``old.legs[leg][b]``.
        # since new ``new.legs[leg][i] == old.legs[leg][p_qind[i]]``,
        # we have new ``new.legs[leg][reverse_sort_perm(p_qind)[b]] == old.legs[leg][b]``
        # thus we replace an entry `b` in ``_qdata[:, leg]``with reverse_sort_perm(q_ind)[b].
        p_qind_r = reverse_sort_perm(p_qind)
        self._qdata[:, leg] = p_qind_r[self._qdata[:, leg]]  # equivalent to
        # self._qdata[:, leg] = [p_qind_r[i] for i in self._qdata[:, leg]]
        self._qdata_sorted = False

    def _advanced_getitem(self, inds, write=False):
        """self[inds] for non-integer `inds`.
        This function is called by self.__getitem__(inds) with ``write=False``,
        and _advanced_setitem_npc with ``write=True``."""
        # non-integer inds -> slicing / projection
        slice_inds = []  # arguments for `take_slice`
        slice_axes = []
        project_masks = []  # arguments for `iproject`
        project_axes = []
        for a, i in enumerate(inds):
            if isinstance(i, slice):
                if i != slice(None):
                    m = np.zeros(self.shape[a], dtype=np.bool_)
                    m[i] = True
                    project_masks.append(m)
                    project_axes.append(a)
            else:
                if isinstance(i, np.ndarray):
                    project_masks.append(i)
                    project_axes.append(a)
                else:  # should be an integer
                    slice_inds.append(i)
                    slice_axes.append(a)
        res = self._take_slice_shallow(slice_inds, slice_axes, write=write)
        map_axes = np.add.accumulate([(a not in slice_axes) for a in xrange(self.rank)]) - 1
        project_map_qinds = res.iproject(project_masks, [map_axes[p] for p in project_axes])
        if not write:
            return res
        else:
            map_qinds = [None] * self.rank
            for a, m_qind in zip(project_axes, project_map_qinds):
                map_qinds[a] = m
            return map_qinds, res

    def _advanced_setitem_npc(self, inds, other):
        """self[inds] = other  for non-integer `inds` and Array `other`.
        This function is called by self.__setitem__(inds, other)."""
        map_qinds, self_part = self._advanced_getitem(inds, write=True)
        # test compatibility with `other`
        if self_part.rank != other.rank:
            raise IndexError("wrong number of indices")
        for sl, ol in zip(self_part.legs, other.legs):
            sl.test_contractible(ol.conj())
        new_blocks = []
        new_qindices = []
        for o_block, o_qindices in itertools.izip(other._data, other._qdata):
            block = self_part._get_block(insert=False, raise_incomp_q=True)
            # a block exists in self_part, if and only if its extended version exists in self.
            if block is not None:  # block exists in self_part
                block[:] = o_block  # overwrites data in self
            else:  # data still needs to be copied
                new_blocks.append(o_block)
                new_qindices.append(o_qindices)
        if len(new_blocks) == 0:
            return  # we had no new blocks, so all data was copied
        # instead of figuring out the originial blocks, we:
        # 1) create an empty copy of self
        cpy = self.zeros_like()
        # 2) add the blocks in the copy. Therefore
        # 2.1) figure out axes removed and kept in _advanced_getitem
        removed_axes = [a for a, i in enumerate(inds)
                        if not (isinstance(i, slice) or isinstance(i, np.ndarray))]
        part_axes = np.array([a for a in xrange(self.rank) if a not in removed_axes])
        # 2.2) figure out the qindices to insert into the copy,
        # such that ``cpy._advanced_getitem(inds)`` has the desired `new_qindices`
        full_qindices = np.empty((len(new_blocks), self.rank), dtype=QDTYPE)
        for a in removed_axes:
            full_qindices[:, a] = self.legs[a].get_qindex(inds[a])[0]
        new_qindices = np.array(new_qindices, dtype=QDTYPE)
        for ax_part, ax_full in enumerate(part_axes):
            # 2.3) revert the map_qind to go back from projected qind to the original
            rev_map_qind = np.nonzero(map_qinds[ax_full] >= 0)[0]
            full_qindices[:, ax_full] = rev_map_qind[new_qindices[:, ax_part]]
        # 2.4) insert the blocks
        for qind in full_qindices:
            cpy._get_block(full_qindices, insert=True)
        # 3.) get a projected version
        _, cpy_part = cpy._advanced_getitem(inds, write=True)
        # 4.) write the data into cpy_part and thus into cpy
        for cpy_block, new_block in itertools.izip(cpy_part._data, new_blocks):
            cpy_block[:] = new_block
        # 5.) and finally copy the blocks from cpy to self
        self._data = self._data + cpy._data
        self._qdata = np.concatenate(self._qdata, cpy._qdata)
        self._qdata_sorted = False
        # TODO should be all, but have to test it
        raise NotImplementedError() # TODO

    def _take_slice_shallow(self, indices, axes, write=False):
        """same as :meth:`take_slice`, but additional parameter `write`.
        If write=True, writing in _data blocks of the result will write into the original array.
        By default, write=False for a good reason. Consider a 2D Array `a` and ::

            b = a.take_slice([0], [1])
            b[0] = 4    # this will write in a, if the block of index [0,0] existed in a.
            #             *but* new inserted blocks will not be written to b
        """
        axes = self.get_leg_indices(toiterable(axes))
        indices = np.asarray(toiterable(indices), dtype=np.intp)
        if len(axes) != len(indices):
            raise ValueError("len(axes) != len(indices)")
        if indices.ndim != 1:
            raise ValueError("indices may only contain ints")
        res = self.copy(deep=(not write))
        if len(axes) == 0:
            return res  # nothing to do
        # qindex and index_within_block for each of the axes
        pos = np.array([self.legs[a].get_qindex(i) for a, i in zip(axes, indices)])
        # which axes to keep
        keep_axes = [a for a in xrange(self.rank) if a not in axes]
        res.legs = [self.legs[a] for a in keep_axes]
        res._set_shape()
        labels = self.get_leg_labels()
        res.set_leg_labels([labels[a] for a in keep_axes])
        # calculate new total charge
        for a, (qi, _) in zip(axes, pos):
            res.qtotal -= self.legs[a].qind[qi, 2:] * self.legs[a].qconj
        res.qtotal = self.chinfo.make_valid(res.qtotal)
        # which blocks to keep
        axes = np.array(axes, dtype=np.intp)
        keep_axes = np.array(keep_axes, dtype=np.intp)
        keep_blocks = np.all(self._qdata[:, axes] == pos[:, 0], axis=1)
        res._qdata = self._qdata[np.ix_(keep_blocks, keep_axes)].copy()
        # res._qdata_sorted is not changed
        # determine the slices to take on _data
        sl = [slice(None)] * self.rank
        for a, ri in zip(axes, pos[:, 1]):
            sl[a] = ri  # the indices within the blocks
        sl = tuple(sl)
        # finally take slices on _data
        res._data = [block[sl] for block, k in itertools.izip(res._data, keep_blocks) if k]
        return res

    def _pre_indexing(self, inds):
        """check if `inds` are valid indices for ``self[inds]`` and replaces Ellipsis by slices.

        Returns
        -------
        only_integer : bool
            whether all of `inds` are (convertible to) np.intp
        inds : tuple, len=self.rank
            `inds`, where ``Ellipsis`` is replaced by the correct number of slice(None).
        """
        if type(inds) != tuple:  # for rank 1
            inds = tuple(inds)
        if len(inds) < self.rank:
            inds = inds + (Ellipsis, )
        if any([(i is Ellipsis) for i in inds]):
            fill = tuple([slice(None)] * (self.rank - len(inds)+1))
            e = inds.index(Ellipsis)
            inds = inds[:e] + fill + inds[e+1:]
        if len(inds) > self.rank:
            raise IndexError("too many indices for Array")
        # do we have only integer entries in `inds`?
        try:
            np.array(inds, dtype=np.intp)
        except:
            return False, inds
        else:
            return True, inds

# functions ====================================================================


def zeros(chargeinfo, legcharges, qtotal=None):
    """create a npc array full of zeros (with no _data).

    This is just a wrapper around ``Array(...)``,
    detailed documentation can be found in the class doc-string of :class:`Array`."""
    return Array(chargeinfo, legcharges, qtotal)