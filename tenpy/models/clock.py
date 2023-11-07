"""Quantum Clock model.

Generalization of transverse field Ising model to higher dimensional on-site Hilbert space.
"""
# Copyright 2023 TeNPy Developers, GNU GPLv3

import numpy as np
from .model import CouplingMPOModel, NearestNeighborModel
from .lattice import Chain
from ..networks.site import ClockSite


__all__ = ['ClockModel', 'ClockChain']


class ClockModel(CouplingMPOModel):
    r"""q-state Quantum clock model on a general lattice

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i,j\rangle, i < j} \mathtt{J} (X_i X_j^\dagger + \mathrm{ h.c.})
            - \sum_{i} \mathtt{g} (Z_i + \mathrm{ h.c.})

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs, each pair appearing
    exactly once.
    The operators :math:`X_i` and :math:`Z_i` are :math:`N \times N` generalizations of
    the Pauli X and Z operators, see :class:`~tenpy.networks.site.ClockSite`.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`ClockModel` below.

    Options
    -------
    .. cfg:config :: ClockModel
        :include: CouplingMPOModel

        conserve : None | 'Z'
            What should be conserved. See :class:`~tenpy.networks.Site.ClockSite`.
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.
        q : int
            The number of states per site.
        J, g : float | array
            Couplings as defined for the Hamiltonian above.

    """

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'Z')
        if conserve == 'best':
            conserve = 'Z'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        q = model_params.get('q', None)
        if q is None:
            raise ValueError('Need to specify q.')
        sort_charge = model_params.get('sort_charge', None)
        return ClockSite(q=q, conserve=conserve, sort_charge=sort_charge)

    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1.))
        g = np.asarray(model_params.get('g', 1.))
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Z', plus_hc=True)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'X', u2, 'Xhc', dx, plus_hc=True)


class ClockChain(ClockModel, NearestNeighborModel):
    """The :class:`ClockModel` on a Chain, suitable for TEBD.

    See the :class:`ClockModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True


class NNNClockModel(CouplingMPOModel):
    r"""Generalization of :class:`ClockModel`, including next-nearest-neighbor terms
    
    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i,j\rangle, i < j} \mathtt{J1} (X_i X_j^\dagger + \mathrm{ h.c.})
        H = - \sum_{\langle\langle i,j\rangle\rangle, i < j} \mathtt{J2} (X_i X_j^\dagger + \mathrm{ h.c.})
            - \sum_{i} \mathtt{g} (Z_i + \mathrm{ h.c.})

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs, each pair appearing
    exactly once, and :math:`\langle\langle i,j \rangle\rangle, i< j` pairs of next-nearest neighbors.
    The operators :math:`X_i` and :math:`Z_i` are :math:`N \times N` generalizations of
    the Pauli X and Z operators, see :class:`~tenpy.networks.site.ClockSite`.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`NNNClockModel` below.

    Options
    -------
    .. cfg:config :: NNNClockModel
        :include: CouplingMPOModel

        conserve : None | 'Z'
            What should be conserved. See :class:`~tenpy.networks.Site.ClockSite`.
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.
        q : int
            The number of states per site.
        J1, J2, g : float | array
            Couplings as defined for the Hamiltonian above.

    """

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'Z')
        if conserve == 'best':
            conserve = 'Z'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        q = model_params.get('q', None)
        if q is None:
            raise ValueError('Need to specify q.')
        sort_charge = model_params.get('sort_charge', None)
        return ClockSite(q=q, conserve=conserve, sort_charge=sort_charge)

    def init_terms(self, model_params):
        J1 = np.asarray(model_params.get('J1', 1.))
        J2 = np.asarray(model_params.get('J2', 0.))
        g = np.asarray(model_params.get('g', 1.))
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Z', plus_hc=True)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J1, u1, 'X', u2, 'Xhc', dx, plus_hc=True)
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(-J2, u1, 'X', u2, 'Xhc', dx, plus_hc=True)


class NNNClockChain(ClockModel):
    """:class:`NNNClockModel` on a :class:`Chain`."""
    default_lattice = Chain
    force_default_lattice = True
