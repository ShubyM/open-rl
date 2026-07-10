import unittest

import yaml

from scripts import run_cluster_e2e


class ClusterE2EScriptTest(unittest.TestCase):
  def test_job_names_are_dns_safe_and_bounded(self) -> None:
    job_name = run_cluster_e2e.make_job_name("FFT/GSM8K_RL" + "x" * 100, suffix="20260709-120000-abcd")

    self.assertLessEqual(len(job_name), 63)
    self.assertRegex(job_name, r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
    self.assertTrue(job_name.endswith("-20260709-120000-abcd"))

  def test_render_manifest_stamps_unique_job_and_removes_empty_extra_arg(self) -> None:
    manifest = run_cluster_e2e.render_manifest(
      "open-rl-e2e-test-1",
      "fft-gsm8k-rl",
      "steps=2 base_model=google/gemma-4-e4b",
      "example/client:revision-a",
    )

    self.assertIn("name: open-rl-e2e-test-1", manifest)
    self.assertIn("image: example/client:revision-a", manifest)
    self.assertIn('        - "steps=2"', manifest)
    self.assertNotIn("E2E-IMAGE", manifest)
    self.assertNotIn("E2E-SCENARIO", manifest)
    self.assertNotIn("E2E-EXTRA-ARGS", manifest)
    self.assertEqual(yaml.safe_load(manifest)["metadata"]["name"], "open-rl-e2e-test-1")

  def test_pod_failure_surfaces_image_pull_reason(self) -> None:
    pod = {
      "status": {
        "phase": "Pending",
        "containerStatuses": [
          {
            "name": "e2e-client",
            "state": {"waiting": {"reason": "ImagePullBackOff", "message": "denied"}},
          }
        ],
      }
    }

    self.assertEqual(
      run_cluster_e2e.pod_failure(pod),
      "container e2e-client is waiting with ImagePullBackOff: denied",
    )


if __name__ == "__main__":
  unittest.main()
