"""Tests for the cross-region anonymous-product layer (slabwater.xregion).

Run with::

    python -m unittest discover -s tests -v

The two worked examples are the ones from the task statement: a one-row split
whose two left-region variants share area, height frequencies and ground total
yet store 3 vs 6 water; and a ridge/corridor map where water must leave region
A and re-enter it through a second seam crossing.
"""

from __future__ import annotations

import random
import unittest

from slabwater.xregion import (
    Curator,
    Seam,
    build_product,
    full_connected,
    full_levels,
    full_volumes,
    issue_certificate,
    recheck_certificate,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def one_row(row, ceiling=4):
    """The one-row map: left region cols 0..2, right region cols 3..4."""
    grid = [row]
    regions = {"L": [(0, 0), (0, 1), (0, 2)], "R": [(0, 3), (0, 4)]}
    ports = {"L": [(0, 2)], "R": [(0, 3), (0, 4)]}
    seams = [Seam((0, 2), (0, 3), 0)]
    products = {
        rid: build_product(rid, 1, grid, cells, ports[rid])
        for rid, cells in regions.items()
    }
    return grid, regions, ports, seams, products, ceiling


def loopback(t3=3, ceiling=10):
    """Ridge row A over corridor row B; outlet at A's left end."""
    grid = [[0, 9, 9, 0], [0, 0, 0, 0]]
    regions = {"A": [(0, i) for i in range(4)], "B": [(1, i) for i in range(4)]}
    ports = {"A": list(regions["A"]), "B": list(regions["B"])}
    seams = [Seam((0, i), (1, i), s) for i, s in enumerate((0, 0, 0, t3))]
    products = {
        rid: build_product(rid, 1, grid, cells, ports[rid])
        for rid, cells in regions.items()
    }
    return grid, regions, ports, seams, products, ceiling


class ExampleOneTests(unittest.TestCase):
    def test_water_3_vs_6_with_identical_public_statistics(self):
        grid_a, regions, ports, seams, prods_a, ceiling = one_row([0, 3, 0, 0, 0])
        _, _, _, _, prods_b, _ = one_row([0, 0, 3, 0, 0])
        la, lb = prods_a["L"], prods_b["L"]
        # Area, whole-region height frequencies and ground total are unchanged.
        self.assertEqual(la.area, lb.area)
        self.assertEqual(la.height_freq, lb.height_freq)
        self.assertEqual(la.ground_total, lb.ground_total)
        # ...yet the (level, component)-merged products differ and so does water.
        self.assertNotEqual(la.product_hash, lb.product_hash)
        outlets = {(0, 4): 0}
        va, _ = Curator([prods_a["L"], prods_a["R"]], seams, ceiling).evaluate(outlets)
        vb, _ = Curator([prods_b["L"], prods_b["R"]], seams, ceiling).evaluate(outlets)
        self.assertEqual(va["L"], 3)
        self.assertEqual(vb["L"], 6)
        # ...and both agree with the complete old solver on the composite map.
        self.assertEqual(full_volumes(grid_a, regions, seams, outlets, ceiling)["L"], 3)
        self.assertEqual(
            full_volumes([[0, 0, 3, 0, 0]], regions, seams, outlets, ceiling)["L"], 6,
        )

    def test_signing_caps_4_and_6(self):
        _, _, _, seams, prods_a, ceiling = one_row([0, 3, 0, 0, 0])
        _, _, _, _, prods_b, _ = one_row([0, 0, 3, 0, 0])
        outlets = {(0, 4): 0}
        results = {}
        for tag, prods in (("a", prods_a), ("b", prods_b)):
            curator = Curator([prods["L"], prods["R"]], seams, ceiling)
            for cap in (4, 6):
                cert = issue_certificate(curator, outlets, {"L": (0, cap), "R": (0, 0)})
                results[(tag, cap)] = cert["ok"]
        self.assertEqual(
            results,
            {("a", 4): True, ("a", 6): True, ("b", 4): False, ("b", 6): True},
        )

    def test_connectivity_query_on_product(self):
        _, _, _, _, prods, ceiling = one_row([0, 3, 0, 0, 0])
        pr = prods["R"]
        self.assertTrue(pr.connected((0, 3), (0, 4), 0))
        # Left region's only port is trivially connected to itself at its height.
        self.assertTrue(prods["L"].connected((0, 2), (0, 2), 0))


class ExampleTwoTests(unittest.TestCase):
    def test_loopback_matches_full_solver_and_beats_naive(self):
        grid, regions, ports, seams, prods, ceiling = loopback(t3=3)
        outlets = {(0, 0): 0}
        volumes, _ = Curator([prods["A"], prods["B"]], seams, ceiling).evaluate(outlets)
        self.assertEqual(volumes, {"A": 3, "B": 0})
        self.assertEqual(full_volumes(grid, regions, seams, outlets, ceiling), volumes)
        # The per-region-once shortcut (treating seam ports as closed) is wrong:
        # it never sees the corridor and reports 9 instead of 3.
        self.assertEqual(prods["A"].storage({(0, 0): 0}, ceiling), 9)

    def test_seam_sill_sweep(self):
        expected = {0: 0, 3: 3, 5: 5, 9: 9, 12: 9}
        for t3, want in expected.items():
            grid, regions, _, seams, prods, ceiling = loopback(t3=t3)
            volumes, _ = Curator([prods["A"], prods["B"]], seams, ceiling).evaluate({(0, 0): 0})
            self.assertEqual(volumes["A"], want, f"t3={t3}")
            self.assertEqual(
                full_volumes(grid, regions, seams, {(0, 0): 0}, ceiling)["A"], want
            )

    def test_regional_vs_global_connectivity(self):
        _, _, _, seams, prods, ceiling = loopback(t3=3)
        curator = Curator([prods["A"], prods["B"]], seams, ceiling)
        # Inside region A the two ends meet only over the 9-high ridge...
        self.assertEqual(prods["A"].port_bottleneck((0, 0), (0, 3)), 9)
        self.assertFalse(prods["A"].connected((0, 0), (0, 3), 8))
        # ...globally they meet at level 3 by detouring through region B.
        self.assertTrue(curator.connected((0, 0), (0, 3), 3))
        self.assertFalse(curator.connected((0, 0), (0, 3), 2))


class SchemeAndCertificateTests(unittest.TestCase):
    def setUp(self):
        self.grid, self.regions, _, self.seams, self.prods, self.ceiling = loopback(t3=3)
        self.curator = Curator([self.prods["A"], self.prods["B"]], self.seams, self.ceiling)
        self.candidates = {(0, 0): 0, (0, 3): 0, (1, 1): 0}
        self.bounds = {"A": (0, 0), "B": (0, 5)}

    def test_cap_one_fails_cap_two_signs(self):
        fail = self.curator.find_scheme(self.candidates, 1, self.bounds)
        self.assertFalse(fail.ok)
        self.assertEqual(fail.reason, "no_feasible_scheme")
        self.assertTrue(fail.exhaustive)
        self.assertTrue(fail.attempts)  # 失败留证
        ok = self.curator.find_scheme(self.candidates, 2, self.bounds)
        self.assertTrue(ok.ok)
        self.assertEqual(ok.scheme, {(0, 0): 0, (0, 3): 0})
        self.assertEqual(ok.volumes, {"A": 0, "B": 0})
        # The signed scheme agrees with the complete old solver.
        self.assertEqual(
            full_volumes(self.grid, self.regions, self.seams, ok.scheme, self.ceiling),
            ok.volumes,
        )

    def test_impossibility_proofs(self):
        # Lower bound above the maximum possible storage (A holds 22 when fully
        # closed): every scheme is drier still, so all fail at once.
        low = self.curator.find_scheme(self.candidates, 2, {"A": (23, 99), "B": (0, 99)})
        self.assertEqual(low.reason, "empty_below_lower")
        # Upper bound below the minimum possible storage (no outlets available,
        # A keeps 22): opening nothing still exceeds the band.
        high = self.curator.find_scheme({}, 0, {"A": (0, 21), "B": (0, 99)})
        self.assertEqual(high.reason, "full_above_upper")
        # Designated port level impossible: A at 0 forces both A-end outlets,
        # which drain the corridor port (1,3) to 0, never to [4, 9].
        negative = self.curator.find_scheme(
            self.candidates, 2, {"A": (0, 0), "B": (0, 0)}, port_requirements={(1, 3): (4, 9)}
        )
        self.assertFalse(negative.ok)

    def test_certificate_recheck_and_tamper(self):
        found = self.curator.find_scheme(self.candidates, 2, self.bounds)
        self.assertTrue(found.ok)
        cert = issue_certificate(self.curator, found.scheme, self.bounds)
        archive = {(p.region, p.version): p for p in self.prods.values()}
        result = recheck_certificate(cert, archive)
        self.assertTrue(result.ok, result.messages)
        tampered = dict(cert)
        tampered["volumes"] = dict(cert["volumes"], A=1)
        self.assertFalse(recheck_certificate(tampered, archive).ok)

    def test_region_reversion_keeps_old_certificate_checkable(self):
        found = self.curator.find_scheme(self.candidates, 2, self.bounds)
        cert = issue_certificate(self.curator, found.scheme, self.bounds)
        archive = {(p.region, p.version): p for p in self.prods.values()}
        # Region A re-versions alone (interior ridge 9 -> 8); B keeps v1.
        grid2 = [[0, 8, 9, 0], [0, 0, 0, 0]]
        pa2 = build_product("A", 2, grid2, self.regions["A"], self.regions["A"])
        archive[("A", 2)] = pa2
        # Old certificate still re-checks against the archived v1 combination.
        self.assertTrue(recheck_certificate(cert, archive).ok)
        # The curator can swap in the new product without any interior blocks.
        curator2 = Curator([pa2, self.prods["B"]], self.seams, self.ceiling)
        volumes2, _ = curator2.evaluate(found.scheme)
        self.assertEqual(volumes2, {"A": 0, "B": 0})

    def test_curator_needs_no_interior_blocks(self):
        # The curator's whole pipeline consumes products and seams only; once
        # products are issued the interior grids can be dropped entirely.
        found = self.curator.find_scheme(self.candidates, 2, self.bounds)
        cert = issue_certificate(self.curator, found.scheme, self.bounds)
        archive = {(p.region, p.version): p for p in self.prods.values()}
        del self.grid, self.regions  # 复开时移走分馆内部块
        self.assertTrue(recheck_certificate(cert, archive).ok)


class RandomizedAuditTests(unittest.TestCase):
    """Small maps: products + seams must reproduce the complete old solver."""

    def random_map(self, rng):
        rows, cols = rng.randint(1, 5), rng.randint(1, 5)
        ceiling = 8
        grid = [[rng.randint(0, 7) for _ in range(cols)] for _ in range(rows)]
        rects = [(0, 0, rows, cols)]
        target = rng.randint(1, 4)
        while len(rects) < target:
            splittable = [i for i, (_, _, h, w) in enumerate(rects) if h > 1 or w > 1]
            if not splittable:
                break
            i = rng.choice(splittable)
            r0, c0, h, w = rects.pop(i)
            if h > 1 and (w == 1 or rng.random() < 0.5):
                t = rng.randint(1, h - 1)
                rects += [(r0, c0, t, w), (r0 + t, c0, h - t, w)]
            else:
                t = rng.randint(1, w - 1)
                rects += [(r0, c0, h, t), (r0, c0 + t, h, w - t)]
        regions, owner = {}, {}
        for rid, (r0, c0, h, w) in enumerate(rects):
            cells = [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
            regions[rid] = cells
            for cell in cells:
                owner[cell] = rid
        seams = []
        portsets = {rid: set() for rid in regions}
        for r in range(rows):
            for c in range(cols):
                for dr, dc in ((1, 0), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and owner[(r, c)] != owner[(nr, nc)]:
                        seams.append(Seam((r, c), (nr, nc), rng.randint(0, ceiling)))
                        portsets[owner[(r, c)]].add((r, c))
                        portsets[owner[(nr, nc)]].add((nr, nc))
        # A few extra public boundary ports on the outer edge (outlet candidates).
        for rid, cells in regions.items():
            edge = [cell for cell in cells if cell[0] in (0, rows - 1) or cell[1] in (0, cols - 1)]
            for cell in rng.sample(edge, min(len(edge), rng.randint(0, 2))):
                portsets[rid].add(cell)
        products = {
            rid: build_product(rid, 1, grid, cells, sorted(portsets[rid]))
            for rid, cells in regions.items()
        }
        return grid, regions, seams, products, ceiling

    def test_audit_200_maps(self):
        rng = random.Random(20261121)
        checks = 0
        for _ in range(200):
            grid, regions, seams, products, ceiling = self.random_map(rng)
            owner = {cell: rid for rid, cells in regions.items() for cell in cells}
            curator = Curator(list(products.values()), seams, ceiling)
            all_ports = [cell for p in products.values() for cell, _ in p.ports]
            matrix = curator.port_bottlenecks()
            for _ in range(8):
                opened = [c for c in all_ports if rng.random() < 0.4]
                outlets = {c: rng.randint(0, ceiling) for c in opened}
                # Cross-region volumes: curator vs complete old solver.
                got, levels = curator.evaluate(outlets)
                want = full_volumes(grid, regions, seams, outlets, ceiling)
                self.assertEqual(got, want, f"outlets={outlets} seams={seams} grid={grid}")
                checks += 1
                # Cross-region connectivity at a random level.
                if len(all_ports) >= 2:
                    p, q = rng.sample(all_ports, 2)
                    w = rng.randint(0, ceiling)
                    via_full = full_connected(grid, seams, p, q, w, owner=owner)
                    key = (p, q) if p <= q else (q, p)
                    via_tree = key in matrix and matrix[key] <= w
                    self.assertEqual(via_full, via_tree)
                    checks += 1
                # Standalone product Q2: region alone, random open ports.
                rid = rng.choice(list(regions))
                cellset = set(regions[rid])
                product = products[rid]
                own_ports = [c for c, _ in product.ports]
                if own_ports:
                    open_own = {c: rng.randint(0, ceiling) for c in own_ports if rng.random() < 0.5}
                    masked = full_levels(grid, [], open_own, ceiling, cellset)
                    want_q2 = sum(
                        max(0, masked[r][c] - grid[r][c]) for r, c in cellset
                    )
                    self.assertEqual(product.storage(open_own, ceiling), want_q2)
                    checks += 1
                # Standalone product Q1.
                if len(own_ports) >= 2:
                    p, q = rng.sample(own_ports, 2)
                    w = rng.randint(0, ceiling)
                    self.assertEqual(
                        product.connected(p, q, w),
                        full_connected(grid, [], p, q, w, cellset),
                    )
                    checks += 1
        self.assertGreater(checks, 2000)


class ScaleTests(unittest.TestCase):
    """32x32 grids, heights 0..63, <=12 ports per boundary."""

    def build(self, grid, ports_per_edge=11):
        n = len(grid)
        cells = [(r, c) for r in range(n) for c in range(n)]
        step = max(1, (n - 1) // (ports_per_edge - 1))
        ports = {(0, i * step) for i in range(ports_per_edge)}
        ports |= {(n - 1, i * step) for i in range(ports_per_edge)}
        return build_product("Z", 1, grid, cells, sorted(ports))

    def test_flat_three_heights(self):
        rng = random.Random(3)
        grid = [[rng.choice([10, 20, 30]) for _ in range(32)] for _ in range(32)]
        product = self.build(grid)
        self.assertLessEqual(len(product.nodes), 2 * 32 * 32 - 1)
        # Flat floors compress hard: few distinct levels, big merged blobs.
        blob_cells = sum(sum(nd[1]) for nd in product.nodes if nd[0] == "B")
        self.assertGreater(blob_cells, 32 * 32 - 100)
        self.exercise(product)

    def test_staircase_all_levels(self):
        rng = random.Random(63)
        grid = [[rng.randint(0, 63) for _ in range(32)] for _ in range(32)]
        product = self.build(grid)
        self.assertLessEqual(len(product.nodes), 2 * 32 * 32 - 1)
        self.exercise(product)

    def exercise(self, product):
        rng = random.Random(7)
        ports = [c for c, _ in product.ports]
        for _ in range(300):
            levels = {c: rng.randint(0, 63) for c in ports if rng.random() < 0.5}
            self.assertIsInstance(product.storage(levels, 63), int)
        # Product survives a JSON round trip unchanged (multi-curator reuse).
        import json
        from slabwater.xregion import RegionProduct

        clone = RegionProduct.from_dict(json.loads(json.dumps(product.to_dict())))
        self.assertEqual(clone.product_hash, product.product_hash)
        self.assertEqual(clone.storage({ports[0]: 0}, 63), product.storage({ports[0]: 0}, 63))

    def test_product_hides_interior_cells(self):
        grid = [[(r * 7 + c * 13) % 64 for c in range(8)] for r in range(8)]
        product = self.build(grid, ports_per_edge=4)
        # Only public port blocks carry coordinates; blobs carry frequencies.
        port_cells = {c for c, _ in product.ports}
        for nd in product.nodes:
            if nd[0] == "P":
                self.assertIn(nd[1], port_cells)
            elif nd[0] == "B":
                self.assertEqual(len(nd), 2)
                self.assertEqual(len(nd[1]), 64)
        # In the serialized product the only coordinate pairs anywhere are the
        # public port blocks; interior cells appear nowhere.
        import json

        raw = json.loads(json.dumps(product.to_dict()))
        seen = {tuple(c) for c, _ in raw["ports"]}
        for nd in raw["nodes"]:
            if nd[0] == "P":
                seen.add(tuple(nd[1]))
            else:
                self.assertNotEqual(nd[0], "P")
        self.assertEqual(seen, port_cells)
        interior = [
            (r, c)
            for r in range(8)
            for c in range(8)
            if (r, c) not in port_cells
        ]
        self.assertTrue(interior)  # the grid really has interior cells
        self.assertTrue(all(cell not in seen for cell in interior))

    def test_four_region_composite_matches_full_solver(self):
        rng = random.Random(11)
        n = 32
        grid = [[rng.randint(0, 63) for _ in range(n)] for _ in range(n)]
        quads = {
            "NW": (0, 0, 16, 16),
            "NE": (0, 16, 16, 16),
            "SW": (16, 0, 16, 16),
            "SE": (16, 16, 16, 16),
        }
        regions = {
            rid: [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
            for rid, (r0, c0, h, w) in quads.items()
        }
        owner = {cell: rid for rid, cells in regions.items() for cell in cells}
        seams, portsets = [], {rid: set() for rid in regions}
        for r in range(0, n, 3):  # 11 crossings per seam: <=12 ports per boundary
            seams.append(Seam((r, 15), (r, 16), rng.randint(0, 63)))
        for c in range(0, n, 3):
            seams.append(Seam((15, c), (16, c), rng.randint(0, 63)))
        for seam in seams:
            portsets[owner[seam.a]].add(seam.a)
            portsets[owner[seam.b]].add(seam.b)
        for rid, cells in regions.items():
            self.assertLessEqual(len(portsets[rid]), 24)  # <=12 per boundary
        products = {
            rid: build_product(rid, 1, grid, cells, sorted(portsets[rid]))
            for rid, cells in regions.items()
        }
        curator = Curator(list(products.values()), seams, 63)
        all_ports = [c for p in products.values() for c, _ in p.ports]
        for _ in range(30):
            outlets = {c: rng.randint(0, 63) for c in all_ports if rng.random() < 0.3}
            got, _ = curator.evaluate(outlets)
            want = full_volumes(grid, regions, seams, outlets, 63)
            self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
