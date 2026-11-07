"""Offline source and numerical checks; does not compile LaTeX or run simulations."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import tempfile
import unittest

import yaml

from snapshot_data import ARMS, digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


class BundleChecks(unittest.TestCase):
    """Catch data substitution, pooling errors, missing inputs, and overstated pairing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load("data/plot_data.json")
        cls.manifest = load("data/source_manifest.json")
        cls.tex_files = sorted(ROOT.glob("*.tex")) + sorted((ROOT / "sections").glob("*.tex")) + sorted((ROOT / "tables").glob("*.tex"))
        cls.tex = "\n".join(p.read_text() for p in cls.tex_files)

    def test_snapshot_hashes(self) -> None:
        for record in self.manifest["records"]:
            if "snapshot_sha256" in record:
                with self.subTest(source=record["source"]):
                    self.assertEqual(digest(ROOT / record["snapshot"]), record["snapshot_sha256"])
        self.assertEqual(digest(ROOT / "data/plot_data.json"), self.manifest["plot_data_sha256"])

    def test_completed_sources_only(self) -> None:
        for record in self.manifest["records"]:
            source = record["source"]
            self.assertNotIn("alkane_repair", source)
            self.assertTrue("alkane_hotspots" in source or "alkane_artifacts/pilot" in source, source)
        for arm in ARMS:
            for rep in range(2):
                self.assertEqual(load(f"data/evidence/arms/{arm}_rep{rep}.json")["status"], "complete")

    def test_calibration_from_per_atom_errors(self) -> None:
        rows = list(csv.DictReader((ROOT / "data/evidence/c16_calibration.csv").read_text().splitlines()))
        for setting in self.data["calibration"]:
            errors = [float(r["force_error"]) for r in rows if r["setting_id"] == setting["setting_id"]]
            with self.subTest(setting=setting["setting_id"]):
                self.assertAlmostEqual(math.sqrt(sum(e * e for e in errors) / (3 * len(errors))), setting["pooled_core_force_rmse"], places=12)
                self.assertAlmostEqual(max(errors), setting["max_core_force_error"], places=12)

    def test_transfer_from_per_atom_errors(self) -> None:
        for key, name in [("c16", "c16_held-out.csv"), ("c32", "c32_transfer.csv")]:
            rows = list(csv.DictReader((ROOT / "data/evidence" / name).read_text().splitlines()))
            errors = [float(r["force_error"]) for r in rows]
            self.assertAlmostEqual(math.sqrt(sum(e * e for e in errors) / (3 * len(errors))), self.data[key]["pooled_core_force_rmse"], places=12)
            self.assertAlmostEqual(max(errors), self.data[key]["max_core_force_error"], places=12)

    def test_gate_preserves_failure_and_transfer_margin(self) -> None:
        targets = self.data["targets"]
        for row in self.data["calibration"] + [self.data["c16"], self.data["c32"]]:
            expected = row["pooled_core_force_rmse"] <= targets["core_force_rmse"] and row["max_core_force_error"] <= targets["max_core_force_error"] and row["complete"] and not row["failures"]
            self.assertEqual(row["passes"], expected)
        graph_two = next(s for s in self.data["calibration"] if s["setting_id"] == "graph-h2")
        self.assertFalse(graph_two["passes"])

    def test_four_label_comparison_and_cost_ratio(self) -> None:
        means = {}
        for arm in ARMS[:4]:
            rows = [rep["rounds"][1] for rep in self.data["arms"][arm]]
            self.assertEqual([r["n_labels"] for r in rows], [4, 4])
            means[arm] = {k: sum(r[k] for r in rows) / 2 for k in ["force_rmse", "cumulative_cpu_core_s"]}
        self.assertEqual(round(means["phal_local"]["force_rmse"], 3), .479)
        self.assertEqual(round(means["phal_full_system"]["force_rmse"], 3), .502)
        ratio = means["phal_local"]["cumulative_cpu_core_s"] / means["phal_full_system"]["cumulative_cpu_core_s"]
        self.assertEqual(round(ratio, 3), .194)

    def test_acquisitions_not_fully_paired(self) -> None:
        for rep in range(2):
            local = self.data["arms"]["phal_local"][rep]["rounds"][1]["acquisitions"]
            full = self.data["arms"]["phal_full_system"][rep]["rounds"][1]["acquisitions"]
            left = [(a["parent_id"], a["candidate_id"]) for a in local]
            right = [(a["parent_id"], a["candidate_id"]) for a in full]
            self.assertEqual(left == right, rep == 0)

    def test_costs_are_sums_of_acquisitions(self) -> None:
        for arm, reps in self.data["arms"].items():
            for rep in reps:
                costs = 0.0
                labels = 0
                for row in rep["rounds"]:
                    costs += sum(a["cpu_core_s"] for a in row["acquisitions"])
                    labels += len(row["acquisitions"])
                    self.assertAlmostEqual(costs, row["cumulative_cpu_core_s"], places=7)
                    self.assertEqual(labels, row["n_labels"])

    def test_targets_are_configured_and_unreached(self) -> None:
        config = yaml.safe_load((ROOT / "data/configs/arms_pilot_v2c.yaml").read_text())
        self.assertEqual(self.data["accuracy_targets"], config["evaluation"]["accuracy_targets"])
        target = max(self.data["accuracy_targets"])
        for reps in self.data["arms"].values():
            for rep in reps:
                self.assertTrue(all(row["force_rmse"] > target for row in rep["rounds"]))

    def test_closed_loop_accounting(self) -> None:
        d = self.data["closed_loop"]
        self.assertEqual(len(d["event_steps"]), 10)
        self.assertEqual(len(d["labels"]), 3)
        self.assertEqual(sorted(row["n_atoms"] for row in d["labels"]), [26, 29, 44])
        self.assertAlmostEqual(sum(row["cpu_core_s"] for row in d["labels"]) / 3600, d["dft_core_h"], places=10)
        before, _, after = d["validation_force_rmse"]
        self.assertEqual(round(100 * (before - after) / before, 1), 21.4)

    def test_topology_fractions_preserve_seed_denominators(self) -> None:
        for name, count, percent in [("initial", 166, 36.6), ("final", 447, 98.7)]:
            block = self.data["closed_loop"]["frozen_evaluation"][name]
            self.assertEqual([r["n_frames"] for r in block["per_seed"]], [151] * 3)
            self.assertEqual(sum(r["n_intact"] for r in block["per_seed"]), count)
            self.assertEqual(block["n_intact"], count)
            self.assertEqual(block["n_frames"], 453)
            self.assertEqual(round(100 * count / block["n_frames"], 1), percent)

    def test_geometry_illustrations_are_actual_held_out_c16(self) -> None:
        self.assertEqual({g["family"] for g in self.data["geometries"]}, {"normal", "torsion", "bend", "contact"})
        for g in self.data["geometries"]:
            self.assertTrue(g["parent_id"].startswith("C16-b02-"))
            self.assertEqual(len(g["positions"]), 50)
            self.assertEqual(g["symbols"].count("C"), 16)
            self.assertEqual(g["symbols"].count("H"), 34)
            self.assertEqual(len(g["bonds"]), 49)

    def test_latex_inputs_and_graphics_exist(self) -> None:
        for relative in re.findall(r"\\input\{([^}]+)\}", self.tex):
            self.assertTrue((ROOT / (relative if relative.endswith(".tex") else relative + ".tex")).is_file(), relative)
        figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", self.tex)
        self.assertEqual(len(figures), 6)
        for relative in figures:
            stem = ROOT / relative
            for ext in [".pdf", ".svg", ".png"]:
                self.assertGreater(stem.with_suffix(ext).stat().st_size, 1000)
            self.assertTrue(stem.read_bytes().startswith(b"%PDF"))

    def test_citations_and_bibtex_keys_match(self) -> None:
        citations = {key.strip() for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", self.tex) for key in group.split(",")}
        entries = re.findall(r"\\bibitem\{([^}]+)\}", self.tex)
        bibkeys = re.findall(r"@article\{([^,]+),", (ROOT / "references.bib").read_text())
        self.assertEqual(citations, set(entries))
        self.assertEqual(citations, set(bibkeys))
        self.assertEqual(len(entries), len(set(entries)))
        self.assertEqual(len(entries), 11)

    def test_cross_references_resolve(self) -> None:
        labels = re.findall(r"\\label\{([^}]+)\}", self.tex)
        refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", self.tex)
        self.assertEqual(len(labels), len(set(labels)))
        self.assertFalse(set(refs) - set(labels))

    def test_latex_structure_without_compilation(self) -> None:
        for path in self.tex_files:
            text = re.sub(r"(?<!\\)%.*", "", path.read_text())
            level = 0
            for bracket in re.findall(r"(?<!\\)[{}]", text):
                level += 1 if bracket == "{" else -1
                self.assertGreaterEqual(level, 0, str(path))
            self.assertEqual(level, 0, str(path))
            stack = []
            for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", text):
                if kind == "begin":
                    stack.append(name)
                else:
                    self.assertTrue(stack, str(path))
                    self.assertEqual(stack.pop(), name, str(path))
            self.assertFalse(stack, str(path))

    def test_author_placeholders_are_preserved(self) -> None:
        text = (ROOT / "metadata.tex").read_text()
        for required in ["Author names to be supplied", "Affiliations to be supplied", "Corresponding-author email to be supplied"]:
            self.assertIn(required, text)

    def test_snapshot_json_helpers_and_nonfinite_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "probe.json"
            write_json(path, {"force": [1.0, -2.0, 3.0]})
            self.assertEqual(read_json(path), {"force": [1.0, -2.0, 3.0]})
            self.assertEqual(digest(path), hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(ValueError):
                write_json(path, {"force": float("nan")})

    def test_generated_tables_reproduce(self) -> None:
        from make_figures import tables
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            tables(self.data, out)
            for name in ["fidelity_calibration.tex", "acquisition_four_labels.tex"]:
                self.assertEqual((out / "tables" / name).read_bytes(), (ROOT / "tables" / name).read_bytes())


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main(verbosity=2)
