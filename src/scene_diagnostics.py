"""Report scene scale before native compilation/model transfer can fail."""

from contextlib import contextmanager


def scene_memory_estimates(ngeom, num_envs):
  pairs = ngeom * (ngeom - 1) // 2
  return {
    # Local MJWarp put_model creates two int64 index vectors, then stacks them.
    # This excludes its many masks, body-id arrays, and other allocations.
    "pair_index_bytes": pairs * 32,
    # MJWarp Data stores vec3 positions and mat33 rotations for every world.
    "geom_pose_bytes": num_envs * ngeom * 12 * 4,
  }


@contextmanager
def report_scene_initialization(env_cfg):
  """Temporarily compose with the existing hook; never serialize the closure."""
  original = env_cfg.scene.spec_fn

  def report(spec):
    if original is not None:
      original(spec)
    count = len(spec.geoms)
    sizes = scene_memory_estimates(count, env_cfg.scene.num_envs)
    gib = 1024**3
    print(f"[init] scene assembled: geoms={count}, envs={env_cfg.scene.num_envs}; "
          "compiling model and allocating physics buffers...", flush=True)
    print(f"[init] MJWarp dense-pair index estimate={sizes['pair_index_bytes']/gib:.2f} GiB host; "
          f"geom poses={sizes['geom_pose_bytes']/gib:.2f} GiB device. "
          "These exclude other allocations; pair indexing depends on backend version.", flush=True)

  env_cfg.scene.spec_fn = report
  try:
    yield
  finally:
    env_cfg.scene.spec_fn = original
