'''
This is an added module that is similar to the acts of analysis.py file but specifically for the network within HPVsim
We will use a graph delta method to efficiently store changes within the network of the simulation as it runs, for now it uses sciris to be inline with other HPVsim files
A few important classes introduced here are the network base class, and the delta class.

Design summary
--------------
Rather than storing a full copy of the contact network at every timestep (expensive in both
memory and time for large populations), the ``NetworkBackend`` drives the same demographic and
partnership-formation/dissolution logic that HPVsim always ran, but instruments it to record only
what *changed* this timestep: which nodes (people) were added or removed, and which edges
(partnerships) were added or removed. This is captured in a ``NetworkDelta`` each timestep.

Edge identity is tracked with a persistent integer edge id ('eid'), assigned once per partnership
by ``BasePeople.add_contacts`` (see base.py) and carried along by the normal Layer machinery
(pop_inds/append/concatenate). This lets a delta unambiguously say "this specific partnership
ended", even if a new partnership between the same two people is formed later.

The ``NetworkBackend`` itself is *not* an Analyzer -- it is required infrastructure that sim.step()
depends on every timestep (it performs the same dissolve/create-partnerships work the old inline
code did), so it is stored directly on the Sim as ``sim.network_backend`` rather than being routed
through the analyzers list. To actually retain the per-timestep deltas produced by the backend
(available each step as ``sim.network_delta``), attach the ``network_history`` Analyzer (see
analysis.py), which simply copies ``sim.network_delta`` into its own storage each timestep.

Other changes regarding this are:
- added network backend and delta attributes on Sim (see sim.py)
- added 'eid' tracking to Layer/Contacts and BasePeople.add_contacts (see base.py)
- added network_added_node_log/network_removed_node_log/network_added_edges/network_dissolved_edges
  bookkeeping to People, populated by add_births/remove_people/create_partnerships/dissolve_partnerships
  (see people.py)
'''
import numpy as np
import sciris as sc

from . import defaults as hpd
from . import utils as hpu


__all__ = [
    "NetworkDelta",
    "NetworkBackend",
    "DefaultNetworkBackend",
    "make_network_backend",
    "stack_edges",
]


# Node entry kinds
ENTRY_INITIAL     = 0
ENTRY_BIRTH       = 1
ENTRY_MIGRATION   = 2
ENTRY_MULTISCALE  = 3  # Extra-resolution cancer agent cloned via People.set_severity's ms_agent_ratio mechanism

# Node removal reasons
NODE_DEATH_OTHER  = 0
NODE_DEATH_CANCER = 1
NODE_DEATH_HIV    = 2
NODE_EMIGRATION   = 3

# Edge removal reasons
EDGE_EXPIRED      = 0
EDGE_NODE_REMOVED = 1
EDGE_DISSOLVED    = 2

# Maps the 'cause' string used by People.remove_people() to a node removal reason code
_CAUSE_TO_REASON = {
    'other':      NODE_DEATH_OTHER,
    'cancer':     NODE_DEATH_CANCER,
    'hiv':        NODE_DEATH_HIV,
    'emigration': NODE_EMIGRATION,
}


class NetworkDelta(sc.prettyobj):
    '''
    A compact record of everything that changed in the partnership network during a single
    timestep (or, for the backend's initial_snapshot, everything present at t=0).

    Args:
        t (int): the timestep this delta corresponds to
        added_nodes (objdict): uid, sex, age, cluster, entry_kind arrays for nodes added this step
        removed_nodes (objdict): uid, reason arrays for nodes removed this step
        added_edges (objdict): eid, f, m, layer arrays for edges added this step
        removed_edges (objdict): eid, f, m, layer, reason arrays for edges removed this step
    '''

    def __init__(
        self,
        t,
        added_nodes=None,
        removed_nodes=None,
        added_edges=None,
        removed_edges=None,
    ):
        self.t = int(t)

        self.added_nodes = (
            added_nodes
            if added_nodes is not None
            else self.empty_added_nodes()
        )

        self.removed_nodes = (
            removed_nodes
            if removed_nodes is not None
            else self.empty_removed_nodes()
        )

        self.added_edges = (
            added_edges
            if added_edges is not None
            else self.empty_added_edges()
        )

        self.removed_edges = (
            removed_edges
            if removed_edges is not None
            else self.empty_removed_edges()
        )

        return

    def is_empty(self):
        ''' True if this delta contains no changes at all '''
        return (
            len(self.added_nodes.uid) == 0
            and len(self.removed_nodes.uid) == 0
            and len(self.added_edges.eid) == 0
            and len(self.removed_edges.eid) == 0
        )

    @staticmethod
    def empty_added_nodes():
        return sc.objdict(
            uid=np.empty(0, dtype=hpd.default_int),
            sex=np.empty(0, dtype=np.int8),
            age=np.empty(0, dtype=hpd.default_float),
            cluster=np.empty(0, dtype=hpd.default_int),
            entry_kind=np.empty(0, dtype=np.int8),
        )

    @staticmethod
    def empty_removed_nodes():
        return sc.objdict(
            uid=np.empty(0, dtype=hpd.default_int),
            reason=np.empty(0, dtype=np.int8),
        )

    @staticmethod
    def empty_added_edges():
        return sc.objdict(
            eid=np.empty(0, dtype=hpd.default_int),
            f=np.empty(0, dtype=hpd.default_int),
            m=np.empty(0, dtype=hpd.default_int),
            layer=np.empty(0, dtype=np.int8),
        )

    @staticmethod
    def empty_removed_edges():
        return sc.objdict(
            eid=np.empty(0, dtype=hpd.default_int),
            f=np.empty(0, dtype=hpd.default_int),
            m=np.empty(0, dtype=hpd.default_int),
            layer=np.empty(0, dtype=np.int8),
            reason=np.empty(0, dtype=np.int8),
        )


class NetworkBackend(sc.prettyobj):
    """
    Base class for HPVsim network backends. A network backend is responsible for advancing the
    partnership network by one timestep (dissolving and creating partnerships, alongside the
    demographic updates that add/remove nodes) and for reporting exactly what changed as a
    NetworkDelta. Concrete subclasses implement this for a specific network model (e.g. the
    default/random sexual network).
    """

    def __init__(self, label=None):
        self.label = label or self.__class__.__name__
        self.initialized = False
        self.finalized = False
        self.layer_map = None   # Ordered list of layer keys, index = the 'layer' code used in NetworkDelta edges
        self.initial_snapshot = None  # NetworkDelta describing the network as of initialization
        self.delta = None       # The NetworkDelta being built/most recently finalized for the current timestep
        return

    def initialize(self, sim):
        ''' Capture the initial network state; must be called after sim.people exists '''
        self.initialized = True
        self.delta = None
        return

    def begin_step(self, sim):
        '''
        Reset per-timestep bookkeeping. Must be called before anything this timestep has a
        chance to add/remove nodes or edges (i.e. before the HIV substep, if any).
        '''
        people = sim.people
        people.network_added_node_log = []
        people.network_removed_node_log = []
        people.network_added_edges = {}
        people.network_dissolved_edges = {}
        self.delta = None
        return

    def step(self, sim):
        ''' Advance the network by one timestep (dissolve/create partnerships, etc) '''
        raise NotImplementedError

    def finalize_step(self, sim):
        ''' Assemble and return the NetworkDelta for the current timestep '''
        raise NotImplementedError

    def finalize(self, sim):
        ''' Called once from sim.finalize() '''
        self.finalized = True
        return


def stack_edges(layer_dict, layer_code, include_reason=False):
    '''
    Combine per-layer edge dict-likes (each with at least 'eid', 'f', 'm', and, if
    include_reason, 'reason') into a single set of arrays tagged with a 'layer' code.

    Shared by every NetworkBackend implementation that needs to turn a {lkey: dict-like}
    mapping (e.g. people.network_added_edges / people.network_dissolved_edges, or a full
    people.contacts) into the flat eid/f/m/layer[/reason] arrays a NetworkDelta expects.

    Args:
        layer_dict (dict or Contacts): mapping of layer key -> dict-like of arrays
        layer_code (dict): mapping of layer key -> int8 code (see NetworkBackend.layer_map)
        include_reason (bool): whether entries also carry a 'reason' array (removed edges)
    '''
    eids, fs, ms, layers, reasons = [], [], [], [], []

    for lkey, data in layer_dict.items():
        # Layers with no edges this step may not carry an 'eid' key at all (add_contacts
        # skips populating columns for empty layers), so check defensively.
        eid_arr = data.get('eid', None) if hasattr(data, 'get') else data['eid']
        n = 0 if eid_arr is None else len(eid_arr)
        if n == 0:
            continue
        code = layer_code[lkey]
        eids.append(np.asarray(eid_arr, dtype=hpd.default_int))
        fs.append(np.asarray(data['f'], dtype=hpd.default_int))
        ms.append(np.asarray(data['m'], dtype=hpd.default_int))
        layers.append(np.full(n, code, dtype=np.int8))
        if include_reason:
            reasons.append(np.asarray(data['reason'], dtype=np.int8))

    if not eids:
        return NetworkDelta.empty_removed_edges() if include_reason else NetworkDelta.empty_added_edges()

    out = sc.objdict(
        eid=np.concatenate(eids),
        f=np.concatenate(fs),
        m=np.concatenate(ms),
        layer=np.concatenate(layers),
    )
    if include_reason:
        out['reason'] = np.concatenate(reasons)
    return out


class DefaultNetworkBackend(NetworkBackend):
    """
    Network backend for HPVsim's default sexual-partnership network model (also used for the
    'random' network, which shares the same create_partnerships/dissolve_partnerships machinery).

    This backend doesn't reimplement partnership formation/dissolution -- that logic already
    lives in People (update_states_pre, dissolve_partnerships, create_partnerships) and
    population.make_contacts. Instead it drives those same calls and reads back the small amount
    of bookkeeping those methods now leave behind (see people.py) to build a NetworkDelta each step.
    """

    def initialize(self, sim):
        people = sim.people

        self.layer_map = list(people.layer_keys())
        self._layer_code = {lkey: np.int8(i) for i, lkey in enumerate(self.layer_map)}

        # Reset per-step bookkeeping so initialization is safe to call more than once
        people.network_added_node_log = []
        people.network_removed_node_log = []
        people.network_added_edges = {}
        people.network_dissolved_edges = {}

        # Capture the initial network as a delta of everything "added" at t=0
        alive_inds = hpu.true(people.alive)
        added_nodes = sc.objdict(
            uid=alive_inds.astype(hpd.default_int),
            sex=people.sex[alive_inds].astype(np.int8),
            age=people.age[alive_inds].astype(hpd.default_float),
            cluster=people.cluster[alive_inds].astype(hpd.default_int),
            entry_kind=np.full(len(alive_inds), ENTRY_INITIAL, dtype=np.int8),
        )
        added_edges = stack_edges(people.contacts, self._layer_code)

        self.initial_snapshot = NetworkDelta(
            t=sim.t,
            added_nodes=added_nodes,
            added_edges=added_edges,
        )

        self.delta = None
        self.initialized = True
        return

    def step(self, sim):
        '''
        Perform the demographic and partnership updates that used to run inline in Sim.step():
        age people, apply deaths/births/migration, dissolve expired/orphaned partnerships, and
        form new ones. Node/edge changes are logged onto sim.people as this runs (see people.py);
        finalize_step() reads that bookkeeping to build this timestep's NetworkDelta.
        '''
        people = sim.people
        t = sim.t
        year = sim.yearvec[t]

        people.update_states_pre(t=t, year=year)  # Ages people, applies deaths, generates new births/migration
        people.dissolve_partnerships(t=t)          # Dissolve expired partnerships and those involving now-dead people

        tind = sim.yearvec[t] - sim['start']
        people.create_partnerships(
            tind,
            sim['mixing'],
            sim['layer_probs'],
            sim['f_cross_layer'],
            sim['m_cross_layer'],
            sim['dur_pship'],
            sim['acts'],
            sim['age_act_pars'],
        )

        return

    def finalize_step(self, sim):
        people = sim.people
        t = sim.t

        # --- Nodes ---
        added_log = people.network_added_node_log
        if len(added_log):
            add_inds = np.concatenate([inds for inds, _ in added_log])
            add_kinds = np.concatenate([
                np.full(len(inds), kind, dtype=np.int8) for inds, kind in added_log
            ])
        else:
            add_inds = np.empty(0, dtype=hpd.default_int)
            add_kinds = np.empty(0, dtype=np.int8)

        added_nodes = sc.objdict(
            uid=add_inds.astype(hpd.default_int),
            sex=people.sex[add_inds].astype(np.int8),
            age=people.age[add_inds].astype(hpd.default_float),
            cluster=people.cluster[add_inds].astype(hpd.default_int),
            entry_kind=add_kinds,
        )

        removed_log = people.network_removed_node_log
        if len(removed_log):
            rem_inds = np.concatenate([inds for inds, _ in removed_log])
            rem_reasons = np.concatenate([
                np.full(len(inds), _CAUSE_TO_REASON[cause], dtype=np.int8)
                for inds, cause in removed_log
            ])
        else:
            rem_inds = np.empty(0, dtype=hpd.default_int)
            rem_reasons = np.empty(0, dtype=np.int8)

        removed_nodes = sc.objdict(
            uid=rem_inds.astype(hpd.default_int),
            reason=rem_reasons,
        )

        # --- Edges ---
        added_edges = stack_edges(people.network_added_edges, self._layer_code)
        removed_edges = stack_edges(people.network_dissolved_edges, self._layer_code, include_reason=True)

        delta = NetworkDelta(
            t=t,
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            added_edges=added_edges,
            removed_edges=removed_edges,
        )
        self.delta = delta
        return delta


def make_network_backend(sim=None, network=None):
    '''
    Factory returning the appropriate NetworkBackend for a given network choice.

    Args:
        sim (Sim): the simulation the backend will be attached to (used to infer the network choice)
        network (str): explicit network choice, overrides sim['network'] if provided
    '''
    if network is None and sim is not None:
        network = sim['network']
    if network == 'francesco':
        from . import francesco_network as hpfn  # Local import: avoids a module-load cycle with network.py
        return hpfn.FrancescoNetworkBackend()
    if network == 'community':
        from . import community_network as hpcn  # Local import: avoids a module-load cycle with network.py
        return hpcn.CommunityNetworkBackend()
    return DefaultNetworkBackend()
