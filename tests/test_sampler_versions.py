import os
import tempfile
import time
import unittest

from server.sampler_versions import version_chain


def make_versions(folder, names, full=()):
  paths = []
  for i, name in enumerate(names):
    path = os.path.join(folder, name)
    os.makedirs(path)
    if name in full:
      os.makedirs(os.path.join(path, "full"))
    os.utime(path, (1000 + i, 1000 + i))
    paths.append(path)
  return paths


class VersionChainTest(unittest.TestCase):
  def test_the_next_delta_is_the_whole_chain_in_steady_state(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      v = make_versions(tmp, ["sampler-1789", "sampler-1", "sampler-2"])
      self.assertEqual(version_chain(v[1], v[2]), [v[2]])

  def test_a_sampler_that_fell_behind_applies_what_it_missed_in_order(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      v = make_versions(tmp, ["sampler-1789", "sampler-1", "sampler-2", "sampler-3"])
      self.assertEqual(version_chain(v[0], v[3]), v[1:])

  def test_cold_start_begins_at_the_last_full_snapshot(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      v = make_versions(tmp, ["sampler-9", "sampler-10", "sampler-11", "sampler-19", "sampler-20"], full=("sampler-9", "sampler-19"))
      self.assertEqual(version_chain(None, v[2]), [os.path.join(v[0], "full"), v[1], v[2]])
      self.assertEqual(version_chain(None, v[4]), [os.path.join(v[3], "full"), v[4]])

  def test_cold_start_without_a_full_replays_from_the_first_delta(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      v = make_versions(tmp, ["sampler-1789", "sampler-1", "sampler-2"])
      self.assertEqual(version_chain(None, v[2]), v)

  def test_a_target_the_listing_does_not_show_yet_is_applied_alone(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      v = make_versions(tmp, ["sampler-1"])
      unseen = os.path.join(tmp, "sampler-2")
      self.assertEqual(version_chain(v[0], unseen), [unseen])
      self.assertEqual(version_chain(None, unseen), [unseen])

  def test_ordering_is_by_write_time_not_name(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      v = make_versions(tmp, ["sampler-1789161243718", "sampler-1"])
      time.sleep(0)
      self.assertEqual(version_chain(None, v[1]), v)


if __name__ == "__main__":
  unittest.main()
