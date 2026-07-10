import unittest

import yaml

from scripts.run_cluster_e2e import render_manifest


class ClusterE2EScriptTest(unittest.TestCase):
  def test_render_manifest(self) -> None:
    manifest = render_manifest(
      "fft-gsm8k-rl",
      "steps=2 base_model=google/gemma-4-e4b",
      "example/client:revision-a",
    )

    document = yaml.safe_load(manifest)
    self.assertEqual(document["metadata"]["generateName"], "open-rl-e2e-")
    self.assertEqual(document["spec"]["template"]["spec"]["containers"][0]["image"], "example/client:revision-a")
    self.assertNotIn("E2E-", manifest)


if __name__ == "__main__":
  unittest.main()
