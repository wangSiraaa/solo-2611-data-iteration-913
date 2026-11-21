"""Cross-region verification: anonymous per-region hydraulic products.

This module layers *custody regions* on top of the frozen-slice water rules in
``slabwater.solver`` (four-neighbour moves, minimax drainage thresholds, ceiling
``C``).  The composite grid is split by horizontal/vertical seams into
rectangular custody regions.  Each region pre-issues a standalone **anonymous
hydraulic product** that answers exactly two kinds of questions:

* connectivity -- "are two ports connected at a given water level?";
* storage -- "given a set of ports with drainage levels, how much water stays
  in this region?".

The product never carries per-cell interior heights or coordinates.  Interior
cells are merged by water level and connected component into anonymous blobs
that keep only cell counts and height frequencies; boundary port blocks are
public by contract, and the region's area, ground total and whole-region height
frequencies are explicitly allowed public information.  No cryptographic
hiding is used -- secrecy comes purely from this aggregation, and anything the
product reveals is information a curator could extract anyway through the two
query kinds the contract requires.

Curator side: the curator holds only the regions' products plus the public
seam list (adjacent boundary ports and the crossing sill).  From these it
computes cross-region water levels and per-region storage, and searches outlet
schemes (at most ``opens`` opened candidate outlets, each with its designated
drain level) that keep every region's storage inside its approved
``[lower, upper]`` band.  Water may leave a region and re-enter through
another port, so levels are solved globally on a merged bottleneck tree --
never by visiting each region once.

Structure of the module:

``build_product`` / ``RegionProduct``
    Region side: freeze a rectangular region into its anonymous product.
    The product is a merge (Kruskal reconstruction) tree over water levels;
    every subtree without ports collapses into one anonymous histogram blob.

``Curator``
    Curator side: joins products and seams into one global merge tree, then
    evaluates schemes, answers connectivity, and searches signable schemes.

``issue_certificate`` / ``recheck_certificate``
    Certificates bind the exact (region, version) vector that was used, so a
    region may re-version independently while old certificates remain
    re-checkable against the archived products they actually combined.

``full_levels`` / ``full_volumes`` / ``full_connected``
    The complete old solver (whole-map minimax Dijkstra, extended with seam
    sills).  Used to audit small maps before release.

All quantities remain integer synthetic-model quantities; not for field
hydraulic decisions.
"""

from __future__ import annotations

import heapq
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .model import Cell, sha256_hex

_INF = 2**62
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))

HEIGHT_LEVELS = 64  # 底高取值 0..63
MAX_GRID = 32  # 格面最大 32x32
MAX_BOUNDARY_PORTS = 12  # 每条区界最多 12 个端口


def _cell(raw: Sequence[int]) -> Cell:
    return (int(raw[0]), int(raw[1]))


def _one_hot(h: int) -> list[int]:
    v = [0] * HEIGHT_LEVELS
    v[h] = 1
    return v


# ---------------------------------------------------------------------------
# Public seam (contract data)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Seam:
    """Adjacency between two boundary ports across a custody seam.

    ``sill`` is the crossing threshold (过缝门槛): a route using this adjacency
    takes ``max(route bottleneck, sill)``, exactly the solver's edge rule.
    """

    a: Cell
    b: Cell
    sill: int

    def to_dict(self) -> dict[str, Any]:
        return {"a": [self.a[0], self.a[1]], "b": [self.b[0], self.b[1]], "sill": self.sill}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Seam":
        return cls(a=_cell(raw["a"]), b=_cell(raw["b"]), sill=int(raw["sill"]))


def _sill_map(seams: Sequence[Seam]) -> dict[frozenset, int]:
    return {frozenset((s.a, s.b)): s.sill for s in seams}


# ---------------------------------------------------------------------------
# Region side: the anonymous hydraulic product
# ---------------------------------------------------------------------------
# Product tree nodes (JSON-serialisable):
#   ("P", (r, c), h)        public boundary port block (boundary blocks are open)
#   ("B", (f0..f63))        anonymous interior blob: cells of one water-level
#                           connected component merged into height frequencies
#   ("N", level, (children)) merge node: its sides connect at this water level
#                           (the lowest bottleneck between them)


@dataclass(frozen=True)
class RegionProduct:
    """Standalone anonymous hydraulic product issued by one custody region."""

    region: str
    version: int
    area: int
    ground_total: int  # 总底高
    height_freq: tuple[int, ...]  # 全区高度频数(合同允许公开)
    ports: tuple[tuple[Cell, int], ...]  # 公开边界端口块 (cell, height)
    nodes: tuple[tuple[Any, ...], ...]
    root: int
    bottlenecks: tuple[tuple[Cell, Cell, int], ...]  # 端口两两最低瓶颈水位
    product_hash: str
    _lazy: dict = field(default_factory=dict, compare=False, repr=False)

    # -- Q1: 端口间在给定水位是否已经连通 ----------------------------------
    def connected(self, p: Cell, q: Cell, level: int) -> bool:
        table = self._bottleneck_table()
        if p == q:
            return table["heights"][p] <= level
        w = table["pairs"].get(_pair_key(p, q))
        return w is not None and w <= level

    def port_bottleneck(self, p: Cell, q: Cell) -> int | None:
        return self._bottleneck_table()["pairs"].get(_pair_key(p, q))

    # -- Q2: 给定一组端口排水水位时本区会留下多少水 --------------------------
    def storage(self, port_levels: Mapping[Cell, int], ceiling: int) -> int:
        """Water left in this region when the given ports drain at given levels.

        Ports not present in ``port_levels`` are closed (not drainage sources);
        every interior cell sits at ``min(C, min over open ports of
        max(port level, lowest bottleneck to that port))``.
        """

        nodes = self.nodes
        g = [_INF] * len(nodes)
        order: list[int] = []
        stack = [self.root]
        while stack:
            u = stack.pop()
            order.append(u)
            nd = nodes[u]
            if nd[0] == "N":
                stack.extend(nd[2])
        for u in reversed(order):
            nd = nodes[u]
            if nd[0] == "P":
                g[u] = port_levels.get(nd[1], _INF)
            elif nd[0] == "B":
                g[u] = _INF
            else:
                g[u] = min(g[ch] for ch in nd[2])
        total = 0
        stack = [(self.root, _INF)]
        while stack:
            u, inherited = stack.pop()
            nd = nodes[u]
            kind = nd[0]
            if kind == "P":
                h = nd[2]
                level = inherited
                if g[u] < _INF:
                    cand = g[u] if g[u] > h else h
                    if cand < level:
                        level = cand
                if level > ceiling:
                    level = ceiling
                if level > h:
                    total += level - h
            elif kind == "B":
                level = inherited if inherited < ceiling else ceiling
                if level > 0:
                    cnt_le, sum_le = self._blob_prefix(u)
                    total += level * cnt_le[level - 1] - sum_le[level - 1]
            else:
                lv = nd[1]
                chs = nd[2]
                for i, ch in enumerate(chs):
                    sib = _INF
                    for j, other in enumerate(chs):
                        if j != i and g[other] < sib:
                            sib = g[other]
                    inh = inherited
                    if sib < _INF:
                        cand = sib if sib > lv else lv
                        if cand < inh:
                            inh = cand
                    stack.append((ch, inh))
        return total

    # -- transport ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "version": self.version,
            "area": self.area,
            "ground_total": self.ground_total,
            "height_freq": list(self.height_freq),
            "ports": [[[c[0], c[1]], h] for c, h in self.ports],
            "nodes": [_node_to_json(nd) for nd in self.nodes],
            "root": self.root,
            "bottlenecks": [
                [[p[0], p[1]], [q[0], q[1]], w] for p, q, w in self.bottlenecks
            ],
            "product_hash": self.product_hash,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RegionProduct":
        product = cls(
            region=str(raw["region"]),
            version=int(raw["version"]),
            area=int(raw["area"]),
            ground_total=int(raw["ground_total"]),
            height_freq=tuple(int(v) for v in raw["height_freq"]),
            ports=tuple(sorted((_cell(c), int(h)) for c, h in raw["ports"])),
            nodes=tuple(_node_from_json(nd) for nd in raw["nodes"]),
            root=int(raw["root"]),
            bottlenecks=tuple(
                sorted((_cell(p), _cell(q), int(w)) for p, q, w in raw["bottlenecks"])
            ),
            product_hash=str(raw["product_hash"]),
        )
        if product.product_hash != _product_hash(product):
            raise ValueError("product_hash does not match product content")
        return product

    # -- internals ----------------------------------------------------------
    def _bottleneck_table(self) -> dict[str, Any]:
        table = self._lazy.get("bottleneck_table")
        if table is None:
            table = {
                "pairs": {_pair_key(p, q): w for p, q, w in self.bottlenecks},
                "heights": {c: h for c, h in self.ports},
            }
            self._lazy["bottleneck_table"] = table
        return table

    def _blob_prefix(self, u: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        cache = self._lazy.setdefault("blob_prefix", {})
        hit = cache.get(u)
        if hit is None:
            hist = self.nodes[u][1]
            cnt_le: list[int] = []
            sum_le: list[int] = []
            c = s = 0
            for h in range(HEIGHT_LEVELS):
                c += hist[h]
                s += hist[h] * h
                cnt_le.append(c)
                sum_le.append(s)
            hit = (tuple(cnt_le), tuple(sum_le))
            cache[u] = hit
        return hit


def _pair_key(p: Cell, q: Cell) -> tuple[Cell, Cell]:
    return (p, q) if p <= q else (q, p)


def _node_to_json(nd: tuple[Any, ...]) -> list[Any]:
    if nd[0] == "P":
        return ["P", [nd[1][0], nd[1][1]], nd[2]]
    if nd[0] == "B":
        return ["B", list(nd[1])]
    return ["N", nd[1], list(nd[2])]


def _node_from_json(raw: Sequence[Any]) -> tuple[Any, ...]:
    if raw[0] == "P":
        return ("P", _cell(raw[1]), int(raw[2]))
    if raw[0] == "B":
        return ("B", tuple(int(v) for v in raw[1]))
    return ("N", int(raw[1]), tuple(int(v) for v in raw[2]))


def _product_hash(product: RegionProduct) -> str:
    return sha256_hex(
        {
            "region": product.region,
            "version": product.version,
            "ports": [[[c[0], c[1]], h] for c, h in product.ports],
            "nodes": [_node_to_json(nd) for nd in product.nodes],
            "root": product.root,
        }
    )


def build_product(
    region: str,
    version: int,
    heights: Sequence[Sequence[int]],
    cells: Sequence[Cell],
    ports: Sequence[Cell],
) -> RegionProduct:
    """Freeze one rectangular custody region into its anonymous product.

    ``heights`` is the composite grid (the region reads only its own cells);
    ``cells`` must form one rectangle (a horizontal/vertical seam cut);
    ``ports`` are the region's public boundary port blocks.
    """

    cellset = {(_cell(c)) for c in cells}
    if not cellset:
        raise ValueError("region must contain at least one cell")
    rows = [r for r, _ in cellset]
    cols = [c for _, c in cellset]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    if len(cellset) != (r1 - r0 + 1) * (c1 - c0 + 1):
        raise ValueError("custody region must be one rectangle")
    for r, c in cellset:
        h = heights[r][c]
        if not (0 <= h < HEIGHT_LEVELS):
            raise ValueError(f"cell {(r, c)} height {h} outside 0..{HEIGHT_LEVELS - 1}")
    portset = {_cell(p) for p in ports}
    if not portset <= cellset:
        raise ValueError("ports must be cells of the region")
    for r, c in sorted(portset):
        if r not in (r0, r1) and c not in (c0, c1):
            raise ValueError(f"port {(r, c)} is not on the region boundary")

    # Kruskal reconstruction over the region's own four-neighbour grid:
    # activating cells in height order, each union creates a merge node whose
    # level is the lowest bottleneck between the two sides.
    nodes: list[list[Any]] = []
    uf: dict[Cell, Cell] = {}
    comp: dict[Cell, int] = {}

    def find(x: Cell) -> Cell:
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    for r, c in sorted(cellset, key=lambda z: (heights[z[0]][z[1]], z)):
        h = heights[r][c]
        leaf = len(nodes)
        nodes.append(["P", (r, c), h] if (r, c) in portset else ["B", _one_hot(h)])
        uf[(r, c)] = (r, c)
        comp[(r, c)] = leaf
        for dr, dc in _NEIGHBOURS:
            nb = (r + dr, c + dc)
            if nb in uf:
                ra, rb = find((r, c)), find(nb)
                if ra != rb:
                    nodes.append(["N", h, [comp[ra], comp[rb]]])
                    uf[rb] = ra
                    comp[ra] = len(nodes) - 1
    root = comp[find(next(iter(cellset)))]

    # Squeeze: any subtree without ports merges into one anonymous blob
    # (counts and height frequencies merged by water level and component).
    order: list[int] = []
    stack = [root]
    while stack:
        u = stack.pop()
        order.append(u)
        if nodes[u][0] == "N":
            stack.extend(nodes[u][2])
    hist: list[list[int] | None] = [None] * len(nodes)
    has_port = [False] * len(nodes)
    for u in reversed(order):
        nd = nodes[u]
        if nd[0] == "P":
            hist[u] = _one_hot(nd[2])
            has_port[u] = True
        elif nd[0] == "B":
            hist[u] = list(nd[1])
        else:
            acc = [0] * HEIGHT_LEVELS
            for ch in nd[2]:
                child = hist[ch]
                assert child is not None
                for i in range(HEIGHT_LEVELS):
                    acc[i] += child[i]
                has_port[u] = has_port[u] or has_port[ch]
            hist[u] = acc
            if not has_port[u]:
                nodes[u] = ["B", acc]

    # Compact: drop nodes left unreachable by the squeeze and re-index.
    reachable: list[int] = []
    stack = [root]
    seen = {root}
    while stack:
        u = stack.pop()
        reachable.append(u)
        if nodes[u][0] == "N":
            for ch in nodes[u][2]:
                if ch not in seen:
                    seen.add(ch)
                    stack.append(ch)
    remap = {old: new for new, old in enumerate(sorted(reachable))}
    compact_nodes: list[list[Any]] = []
    for old in sorted(reachable):
        nd = nodes[old]
        if nd[0] == "N":
            nd = ["N", nd[1], [remap[ch] for ch in nd[2]]]
        compact_nodes.append(nd)
    root = remap[root]

    # Port-to-port lowest bottlenecks inside the region (contract-inherent:
    # they are exactly what the connectivity query must answer).
    bottlenecks: dict[tuple[Cell, Cell], int] = {}
    plist: list[list[Cell] | None] = [None] * len(compact_nodes)
    order: list[int] = []
    stack = [root]
    while stack:
        u = stack.pop()
        order.append(u)
        if compact_nodes[u][0] == "N":
            stack.extend(compact_nodes[u][2])
    for u in reversed(order):
        nd = compact_nodes[u]
        if nd[0] == "P":
            plist[u] = [nd[1]]
        elif nd[0] == "B":
            plist[u] = []
        else:
            lists = [plist[ch] for ch in nd[2]]
            merged: list[Cell] = []
            for sub in lists:
                assert sub is not None
                merged.extend(sub)
            for i in range(len(lists)):
                for j in range(i + 1, len(lists)):
                    for p in lists[i]:
                        for q in lists[j]:
                            bottlenecks[_pair_key(p, q)] = nd[1]
            plist[u] = merged

    freq = [0] * HEIGHT_LEVELS
    for u in range(len(compact_nodes)):
        nd = compact_nodes[u]
        if nd[0] == "P":
            freq[nd[2]] += 1
        elif nd[0] == "B":
            for h, n in enumerate(nd[1]):
                freq[h] += n
    product = RegionProduct(
        region=region,
        version=version,
        area=len(cellset),
        ground_total=sum(h * n for h, n in enumerate(freq)),
        height_freq=tuple(freq),
        ports=tuple(sorted((p, heights[p[0]][p[1]]) for p in portset)),
        nodes=tuple(_node_to_json_tuple(nd) for nd in compact_nodes),
        root=root,
        bottlenecks=tuple(sorted((p, q, w) for (p, q), w in bottlenecks.items())),
        product_hash="",
    )
    object.__setattr__(product, "product_hash", _product_hash(product))
    return product


def _node_to_json_tuple(nd: Sequence[Any]) -> tuple[Any, ...]:
    if nd[0] == "P":
        return ("P", (nd[1][0], nd[1][1]), nd[2])
    if nd[0] == "B":
        return ("B", tuple(nd[1]))
    return ("N", nd[1], tuple(nd[2]))


# ---------------------------------------------------------------------------
# Curator side
# ---------------------------------------------------------------------------
class Curator:
    """Holds only region products and the public seam list.

    The global merge tree is built once and reused across every outlet
    combination; interior blocks of the regions are never needed (and may be
    removed by the regions after issuing their products).
    """

    def __init__(
        self,
        products: Sequence[RegionProduct],
        seams: Sequence[Seam],
        ceiling: int,
    ):
        if not products:
            raise ValueError("curator needs at least one product")
        self.products = tuple(products)
        self.seams = tuple(sorted(seams, key=lambda s: (s.a, s.b)))
        self.ceiling = ceiling
        self._port_heights: dict[Cell, int] = {}
        for product in self.products:
            for cell, h in product.ports:
                self._port_heights[cell] = h
        for seam in self.seams:
            for cell in (seam.a, seam.b):
                if cell not in self._port_heights:
                    raise ValueError(f"seam endpoint {cell} is not a public boundary port")
        self._tnodes, self._roots = self._build_global_tree()
        self._matrix: dict[tuple[Cell, Cell], int] | None = None

    # -- global merge tree ---------------------------------------------------
    def _build_global_tree(self) -> tuple[list[list[Any]], list[int]]:
        """Merge region products and seams into one bottleneck tree.

        Vertices: ("P", cell) ports weighted by their public block height;
        ("S", i) seam dummies weighted by the crossing sill; ("J", region, u)
        join dummies weighted by a region-internal merge level, linked to one
        representative port of each port-bearing child subtree.  Activating
        vertices in weight order (Kruskal reconstruction) yields a tree whose
        LCA levels are the true cross-region lowest bottlenecks -- including
        routes that leave a region and re-enter it through another port.
        """

        verts: dict[Any, int] = {}
        adj: dict[Any, set] = {}
        for product in self.products:
            for cell, h in product.ports:
                verts[("P", cell)] = h
            nodes = product.nodes
            order: list[int] = []
            stack = [product.root]
            while stack:
                u = stack.pop()
                order.append(u)
                if nodes[u][0] == "N":
                    stack.extend(nodes[u][2])
            plist: dict[int, list[Cell]] = {}
            for u in reversed(order):
                nd = nodes[u]
                if nd[0] == "P":
                    plist[u] = [nd[1]]
                elif nd[0] == "B":
                    plist[u] = []
                else:
                    merged: list[Cell] = []
                    reps: list[Cell] = []
                    for ch in nd[2]:
                        sub = plist[ch]
                        if sub:
                            reps.append(sub[0])
                            merged.extend(sub)
                    if len(reps) >= 2:
                        key = ("J", product.region, u)
                        verts[key] = nd[1]
                        for rep in reps:
                            adj.setdefault(key, set()).add(("P", rep))
                            adj.setdefault(("P", rep), set()).add(key)
                    plist[u] = merged
        for i, seam in enumerate(self.seams):
            key = ("S", i)
            verts[key] = seam.sill
            for cell in (seam.a, seam.b):
                adj.setdefault(key, set()).add(("P", cell))
                adj.setdefault(("P", cell), set()).add(key)

        tnodes: list[list[Any]] = []
        uf: dict[Any, Any] = {}
        comp: dict[Any, int] = {}

        def find(x: Any) -> Any:
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x

        for key in sorted(verts, key=lambda k: (verts[k], repr(k))):
            w = verts[key]
            leaf = len(tnodes)
            tnodes.append(["L", w, key])
            uf[key] = key
            comp[key] = leaf
            for nb in adj.get(key, ()):
                if nb in uf:
                    ra, rb = find(key), find(nb)
                    if ra != rb:
                        tnodes.append(["M", w, [comp[ra], comp[rb]]])
                        uf[rb] = ra
                        comp[ra] = len(tnodes) - 1
        roots = sorted({comp[find(k)] for k in verts})
        return tnodes, roots

    def _port_levels(self, outlets: Mapping[Cell, int]) -> dict[Cell, int]:
        """Level of every port: min over opened outlets of max(drain, bottleneck).

        Components without any opened outlet sit at the ceiling.
        """

        sources = {}
        for cell, drain in outlets.items():
            cell = _cell(cell)
            if cell not in self._port_heights:
                raise ValueError(f"outlet {cell} is not a public boundary port")
            sources[("P", cell)] = drain
        tnodes = self._tnodes
        g = [_INF] * len(tnodes)
        order: list[int] = []
        for root in self._roots:
            stack = [root]
            while stack:
                u = stack.pop()
                order.append(u)
                if tnodes[u][0] == "M":
                    stack.extend(tnodes[u][2])
        for u in reversed(order):
            nd = tnodes[u]
            if nd[0] == "L":
                g[u] = sources.get(nd[2], _INF)
            else:
                g[u] = min(g[ch] for ch in nd[2])
        levels: dict[Cell, int] = {}
        for root in self._roots:
            stack = [(root, _INF)]
            while stack:
                u, inherited = stack.pop()
                nd = tnodes[u]
                if nd[0] == "L":
                    level = inherited
                    if g[u] < _INF:
                        cand = g[u] if g[u] > nd[1] else nd[1]
                        if cand < level:
                            level = cand
                    key = nd[2]
                    if key[0] == "P":
                        levels[key[1]] = level if level < self.ceiling else self.ceiling
                else:
                    lv = nd[1]
                    chs = nd[2]
                    for i, ch in enumerate(chs):
                        sib = _INF
                        for j, other in enumerate(chs):
                            if j != i and g[other] < sib:
                                sib = g[other]
                        inh = inherited
                        if sib < _INF:
                            cand = sib if sib > lv else lv
                            if cand < inh:
                                inh = cand
                        stack.append((ch, inh))
        return levels

    # -- queries --------------------------------------------------------------
    def evaluate(self, outlets: Mapping[Cell, int]) -> tuple[dict[str, int], dict[Cell, int]]:
        """Per-region storage and port levels for one opening scheme."""

        levels = self._port_levels(outlets)
        volumes = {}
        for product in self.products:
            volumes[product.region] = product.storage(
                {cell: levels[cell] for cell, _ in product.ports}, self.ceiling
            )
        return volumes, levels

    def port_bottlenecks(self) -> dict[tuple[Cell, Cell], int]:
        if self._matrix is None:
            matrix: dict[tuple[Cell, Cell], int] = {}
            tnodes = self._tnodes

            def collect(u: int) -> list[Cell]:
                nd = tnodes[u]
                if nd[0] == "L":
                    return [nd[2][1]] if nd[2][0] == "P" else []
                lists = [collect(ch) for ch in nd[2]]
                for i in range(len(lists)):
                    for j in range(i + 1, len(lists)):
                        for p in lists[i]:
                            for q in lists[j]:
                                matrix[_pair_key(p, q)] = nd[1]
                return [c for sub in lists for c in sub]

            sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))
            for root in self._roots:
                collect(root)
            self._matrix = matrix
        return self._matrix

    def connected(self, p: Cell, q: Cell, level: int) -> bool:
        p, q = _cell(p), _cell(q)
        if p == q:
            return self._port_heights[p] <= level
        w = self.port_bottlenecks().get(_pair_key(p, q))
        return w is not None and w <= level

    # -- scheme search ---------------------------------------------------------
    def find_scheme(
        self,
        candidates: Mapping[Cell, int],
        opens: int,
        bounds: Mapping[str, tuple[int, int]],
        port_requirements: Mapping[Cell, tuple[int, int]] | None = None,
        max_subsets: int = 4096,
    ) -> "SchemeResult":
        """Search outlet sets (size <= opens) for one keeping every region in band.

        ``candidates`` maps each openable public boundary port to its designated
        drain level (指定端口水位); ``bounds`` maps each region to its approved
        [lower, upper] storage band; ``port_requirements`` optionally constrains
        the resulting water level of designated ports.  Storage is monotone
        non-increasing in the opened set, which gives two immediate impossibility
        proofs (empty set below a lower bound, or all candidates opened still
        above an upper bound) and a safe pruning rule; every evaluated attempt
        is kept as evidence when all schemes fail.
        """

        cand: dict[Cell, int] = {}
        for cell, drain in candidates.items():
            cell = _cell(cell)
            if cell not in self._port_heights:
                raise ValueError(f"candidate outlet {cell} is not a public boundary port")
            cand[cell] = int(drain)
        cells = sorted(cand)
        req = {_cell(c): tuple(v) for c, v in (port_requirements or {}).items()}
        n_subsets = sum(
            math.comb(len(cells), k) for k in range(0, min(opens, len(cells)) + 1)
        )
        if n_subsets > max_subsets:
            return SchemeResult(
                ok=False,
                reason="search_space_too_large",
                scheme=None,
                volumes=None,
                port_levels=None,
                attempts=[],
                exhaustive=False,
            )

        attempts: list[SchemeAttempt] = []
        evaluated: dict[tuple[Cell, ...], tuple[dict[str, int], dict[Cell, int], list[dict]]] = {}

        def evaluate(sub: tuple[Cell, ...]):
            if sub not in evaluated:
                volumes, levels = self.evaluate({c: cand[c] for c in sub})
                violations: list[dict[str, Any]] = []
                for region in sorted(bounds):
                    lo, hi = bounds[region]
                    actual = volumes[region]
                    if actual < lo:
                        violations.append(
                            {"region": region, "constraint": "lower", "limit": lo, "actual": actual}
                        )
                    elif actual > hi:
                        violations.append(
                            {"region": region, "constraint": "upper", "limit": hi, "actual": actual}
                        )
                for cell in sorted(req):
                    lo, hi = req[cell]
                    actual = levels[cell]
                    if not lo <= actual <= hi:
                        violations.append(
                            {
                                "port": [cell[0], cell[1]],
                                "constraint": "level",
                                "limit": [lo, hi],
                                "actual": actual,
                            }
                        )
                evaluated[sub] = (volumes, levels, violations)
            return evaluated[sub]

        found: tuple[tuple[Cell, ...], dict[str, int], dict[Cell, int]] | None = None

        def dfs(i: int, sub: list[Cell]) -> None:
            nonlocal found
            if found is not None:
                return
            key = tuple(sub)
            volumes, levels, violations = evaluate(key)
            attempts.append(
                SchemeAttempt(opened=key, feasible=not violations, volumes=volumes, violations=violations)
            )
            if not violations:
                found = (key, volumes, levels)
                return
            if any(v.get("constraint") == "lower" for v in violations):
                # Storage only decreases when more outlets open: every superset
                # of this set is too dry as well.
                attempts[-1].note = "pruned: supersets are drier still"
                return
            if len(sub) == opens:
                return
            for j in range(i, len(cells)):
                dfs(j + 1, sub + [cells[j]])

        # Immediate impossibility proofs before the walk.
        v_empty, _, viol_empty = evaluate(())
        if any(v.get("constraint") == "lower" for v in viol_empty):
            attempts.append(
                SchemeAttempt((), False, v_empty, viol_empty, note="empty set already below lower bound")
            )
            return SchemeResult(False, "empty_below_lower", None, None, None, attempts, True)
        full = tuple(cells)
        v_full, _, viol_full = evaluate(full)
        if any(v.get("constraint") == "upper" for v in viol_full):
            attempts.append(
                SchemeAttempt(full, False, v_full, viol_full, note="all candidates open still above upper bound")
            )
            return SchemeResult(False, "full_above_upper", None, None, None, attempts, True)

        dfs(0, [])
        if found is not None:
            sub, volumes, levels = found
            return SchemeResult(
                ok=True,
                reason=None,
                scheme={c: cand[c] for c in sub},
                volumes=volumes,
                port_levels=levels,
                attempts=attempts,
                exhaustive=False,
            )
        return SchemeResult(False, "no_feasible_scheme", None, None, None, attempts, True)


@dataclass
class SchemeAttempt:
    opened: tuple[Cell, ...]
    feasible: bool
    volumes: dict[str, int]
    violations: list[dict[str, Any]]
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "opened": [[c[0], c[1]] for c in self.opened],
            "feasible": self.feasible,
            "volumes": self.volumes,
            "violations": self.violations,
            "note": self.note,
        }


@dataclass
class SchemeResult:
    ok: bool
    reason: str | None
    scheme: dict[Cell, int] | None
    volumes: dict[str, int] | None
    port_levels: dict[Cell, int] | None
    attempts: list[SchemeAttempt]
    exhaustive: bool

    def evidence(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "scheme": (
                {f"{c[0]},{c[1]}": v for c, v in self.scheme.items()} if self.scheme else None
            ),
            "volumes": self.volumes,
            "exhaustive": self.exhaustive,
            "attempts": [a.to_dict() for a in self.attempts],
        }


# ---------------------------------------------------------------------------
# Certificates: bind the exact (region, version) vector that was evaluated
# ---------------------------------------------------------------------------
def issue_certificate(
    curator: Curator,
    scheme: Mapping[Cell, int],
    bounds: Mapping[str, tuple[int, int]],
    port_requirements: Mapping[Cell, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Sign one opening scheme; the certificate cites every product version used."""

    volumes, levels = curator.evaluate(scheme)
    ok = all(bounds[r][0] <= volumes[r] <= bounds[r][1] for r in bounds)
    if port_requirements:
        for cell, (lo, hi) in port_requirements.items():
            ok = ok and lo <= levels[_cell(cell)] <= hi
    cert: dict[str, Any] = {
        "products": {
            p.region: {"version": p.version, "product_hash": p.product_hash}
            for p in curator.products
        },
        "seams": [s.to_dict() for s in curator.seams],
        "ceiling": curator.ceiling,
        "outlets": {f"{c[0]},{c[1]}": int(v) for c, v in sorted(scheme.items())},
        "bounds": {r: [lo, hi] for r, (lo, hi) in sorted(bounds.items())},
        "port_requirements": {
            f"{c[0]},{c[1]}": [lo, hi]
            for c, (lo, hi) in sorted((port_requirements or {}).items())
        },
        "volumes": volumes,
        "port_levels": {f"{c[0]},{c[1]}": v for c, v in sorted(levels.items())},
        "ok": ok,
    }
    cert["cert_hash"] = sha256_hex(cert)
    return cert


@dataclass
class RecheckResult:
    ok: bool
    checks: dict[str, bool]
    messages: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": self.checks, "messages": self.messages}


def recheck_certificate(
    cert: Mapping[str, Any],
    archive: Mapping[tuple[str, int], RegionProduct],
) -> RecheckResult:
    """Re-check a certificate against the archived products it actually used.

    Regions may re-version independently; the archive keeps every issued
    product, so an old certificate is always re-evaluated with the exact
    (region, version) combination it cites.  Only products and the public seam
    list are needed -- the regions' interior blocks may be long removed.
    """

    checks: dict[str, bool] = {}
    messages: list[str] = []
    products: list[RegionProduct] = []
    available = True
    hashes_match = True
    for region, ref in sorted(cert["products"].items()):
        product = archive.get((region, ref["version"]))
        if product is None:
            available = False
            messages.append(f"product {region}@v{ref['version']} missing from archive")
            continue
        if product.product_hash != ref["product_hash"]:
            hashes_match = False
            messages.append(f"product {region}@v{ref['version']} hash mismatch")
        products.append(product)
    checks["products_available"] = available
    checks["product_hashes_match"] = hashes_match
    if not available:
        return RecheckResult(False, checks, messages)
    curator = Curator(products, [Seam.from_dict(s) for s in cert["seams"]], cert["ceiling"])
    scheme = {_cell(k.split(",")): v for k, v in cert["outlets"].items()}
    volumes, levels = curator.evaluate(scheme)
    checks["volumes_match"] = volumes == cert["volumes"]
    checks["levels_match"] = levels == {
        _cell(k.split(",")): v for k, v in cert["port_levels"].items()
    }
    ok = all(cert["bounds"][r][0] <= volumes[r] <= cert["bounds"][r][1] for r in cert["bounds"])
    for key, (lo, hi) in cert.get("port_requirements", {}).items():
        ok = ok and lo <= levels[_cell(key.split(","))] <= hi
    checks["ok_reproduced"] = ok == cert["ok"]
    if not checks["volumes_match"] or not checks["levels_match"]:
        messages.append("recomputed water numbers differ from the certificate")
    return RecheckResult(all(checks.values()), checks, messages)


# ---------------------------------------------------------------------------
# Complete old solver on the composite map (audit baseline for small maps)
# ---------------------------------------------------------------------------
def full_levels(
    heights: Sequence[Sequence[int]],
    seams: Sequence[Seam],
    outlets: Mapping[Cell, int],
    ceiling: int,
    cellset: set[Cell] | None = None,
    owner: Mapping[Cell, Any] | None = None,
) -> list[list[int]]:
    """Whole-map minimax Dijkstra, the solver's rule extended with seam sills.

    A route's threshold is the max of every cell height on it, every crossed
    seam sill, and the outlet's drain level; a cell's level is ``min(C, best
    threshold)`` and ``C`` when no outlet is reachable.  When ``owner`` (cell
    -> region) is given, adjacencies between different regions are sealed
    unless a public seam connects exactly that port pair -- the custody
    boundary is a wall except at the published ports.
    """

    rows, cols = len(heights), len(heights[0])
    sill = _sill_map(seams)

    def edge_sill(a: Cell, b: Cell) -> int | None:
        if owner is None or owner[a] == owner[b]:
            return 0
        return sill.get(frozenset((a, b)))  # None: sealed custody boundary

    dist = [[_INF] * cols for _ in range(rows)]
    heap: list[tuple[int, int, int]] = []
    for cell, drain in outlets.items():
        r, c = _cell(cell)
        if cellset is not None and (r, c) not in cellset:
            continue
        start = max(drain, heights[r][c])
        if start < dist[r][c]:
            dist[r][c] = start
            heapq.heappush(heap, (start, r, c))
    while heap:
        d, r, c = heapq.heappop(heap)
        if d != dist[r][c]:
            continue
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (cellset is None or (nr, nc) in cellset):
                s = edge_sill((r, c), (nr, nc))
                if s is None:
                    continue
                nd = max(d, heights[nr][nc], s)
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))
    return [
        [ceiling if dist[r][c] > ceiling else dist[r][c] for c in range(cols)]
        for r in range(rows)
    ]


def full_volumes(
    heights: Sequence[Sequence[int]],
    regions: Mapping[str, Sequence[Cell]],
    seams: Sequence[Seam],
    outlets: Mapping[Cell, int],
    ceiling: int,
) -> dict[str, int]:
    owner = {cell: region for region, cells in regions.items() for cell in cells}
    levels = full_levels(heights, seams, outlets, ceiling, owner=owner)
    return {
        region: sum(max(0, levels[r][c] - heights[r][c]) for r, c in cells)
        for region, cells in regions.items()
    }


def full_connected(
    heights: Sequence[Sequence[int]],
    seams: Sequence[Seam],
    p: Cell,
    q: Cell,
    level: int,
    cellset: set[Cell] | None = None,
    owner: Mapping[Cell, Any] | None = None,
) -> bool:
    rows, cols = len(heights), len(heights[0])
    sill = _sill_map(seams)
    p, q = _cell(p), _cell(q)
    if heights[p[0]][p[1]] > level or heights[q[0]][q[1]] > level:
        return False
    seen = {p}
    stack = [p]
    while stack:
        r, c = stack.pop()
        if (r, c) == q:
            return True
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or (nr, nc) in seen:
                continue
            if heights[nr][nc] > level or (cellset is not None and (nr, nc) not in cellset):
                continue
            if owner is not None and owner[(r, c)] != owner[(nr, nc)]:
                s = sill.get(frozenset(((r, c), (nr, nc))))
                if s is None or s > level:
                    continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return False


# ---------------------------------------------------------------------------
# Worked examples (see README): python -m slabwater.xregion
# ---------------------------------------------------------------------------
def _example_one() -> None:
    print("例1: 一行切左右两区 (C=4, 出口仅在右端, 接缝门槛0)")
    seams = [Seam((0, 2), (0, 3), 0)]
    regions = {"L": [(0, 0), (0, 1), (0, 2)], "R": [(0, 3), (0, 4)]}
    ports = {"L": [(0, 2)], "R": [(0, 3), (0, 4)]}
    outlets = {(0, 4): 0}
    for tag, row in (("v1a", [0, 3, 0, 0, 0]), ("v1b", [0, 0, 3, 0, 0])):
        grid = [row]
        pl = build_product("L", 1, grid, regions["L"], ports["L"])
        pr = build_product("R", 1, grid, regions["R"], ports["R"])
        curator = Curator([pl, pr], seams, 4)
        volumes, _ = curator.evaluate(outlets)
        full = full_volumes(grid, regions, seams, outlets, 4)
        blobs = [nd[1] for nd in pl.nodes if nd[0] == "B"]
        print(
            f"  左区 {row[:3]}: 面积={pl.area} 总底高={pl.ground_total} "
            f"高度频数={ {h: n for h, n in enumerate(pl.height_freq) if n} } "
            f"匿名块={ [{h: n for h, n in enumerate(b) if n} for b in blobs] }"
        )
        print(f"    产品+接缝 水量={volumes}   完整旧求解器={full}")
        for cap in (4, 6):
            cert = issue_certificate(curator, outlets, {"L": (0, cap), "R": (0, 0)})
            print(
                f"    左区核准上限 {cap}: 左区水量 {volumes['L']} -> "
                f"{'可签' if cert['ok'] else '拒签'}"
            )


def _example_two() -> None:
    print("例2: 两条跨区通路、必须绕行回区 (C=10)")
    grid = [[0, 9, 9, 0], [0, 0, 0, 0]]
    regions = {"A": [(0, i) for i in range(4)], "B": [(1, i) for i in range(4)]}
    ports = {"A": list(regions["A"]), "B": list(regions["B"])}
    outlets = {(0, 0): 0}
    print("  A 行 [0,9,9,0] 为山脊, B 行 [0,0,0,0] 为走廊; 出口在 (0,0)")
    for t3 in (0, 3, 5, 9, 12):
        seams = [Seam((0, i), (1, i), s) for i, s in enumerate((0, 0, 0, t3))]
        pa = build_product("A", 1, grid, regions["A"], ports["A"])
        pb = build_product("B", 1, grid, regions["B"], ports["B"])
        volumes, _ = Curator([pa, pb], seams, 10).evaluate(outlets)
        full = full_volumes(grid, regions, seams, outlets, 10)
        naive = pa.storage({(0, 0): 0}, 10)  # 反面教材: A 区孤立只求一次
        print(
            f"  列3门槛={t3:2d}: 产品+接缝 A={volumes['A']}  完整旧求解器 A={full['A']}"
            f"  (A区孤立算={naive})"
        )
    seams = [Seam((0, i), (1, i), s) for i, s in enumerate((0, 0, 0, 3))]
    pa = build_product("A", 1, grid, regions["A"], ports["A"])
    pb = build_product("B", 1, grid, regions["B"], ports["B"])
    curator = Curator([pa, pb], seams, 10)
    inside = pa.port_bottleneck((0, 0), (0, 3))
    across = curator.port_bottlenecks()[_pair_key((0, 0), (0, 3))]
    print(f"  A区两端 (0,0)-(0,3): 区内连通水位={inside}, 跨区绕行后={across}")


def main() -> int:
    _example_one()
    _example_two()
    return 0


if __name__ == "__main__":
    sys.exit(main())
