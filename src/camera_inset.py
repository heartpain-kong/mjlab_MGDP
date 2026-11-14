"""First-person RGB picture-in-picture for MuJoCo's native viewer."""

import time
import mujoco
from glfw import KEY_F8
from src.config.lgdog1 import FRONT_CAMERA_NAME


def find_camera(model, name=FRONT_CAMERA_NAME):
  matches = [i for i in range(model.ncam) if model.camera(i).name == name
             or model.camera(i).name.endswith("/" + name)]
  if len(matches) != 1:
    raise ValueError(f"Expected one camera {name!r}, found {len(matches)}")
  return matches[0]


class CameraInset:
  def __init__(self, model, size=256, fps=10., corner="bottom-left"):
    if not 64 <= size <= 512 or not 1 <= fps <= 60:
      raise ValueError("Camera inset size must be 64..512 and fps 1..60")
    if corner not in ("bottom-left", "bottom-right", "top-left", "top-right"):
      raise ValueError("Unknown camera inset corner")
    self.camera = find_camera(model)
    width,height=model.cam_resolution[self.camera]
    self.aspect=float(width/height) if width>0 and height>0 else 1.
    self.size, self.fps, self.corner = size, fps, corner
    self.enabled, self.dirty = True, True
    self.last_update = float("-inf")
    self.renderer = None
    self.model = model
    # MuJoCo scales near clip by scene extent. Large terrain grids otherwise
    # clip out close obstacles seen from a camera only ~0.385m above ground.
    model.vis.map.znear = min(model.vis.map.znear, .01/max(model.stat.extent, .01))

  def toggle(self):
    # Key callbacks run on the UI thread: defer rendering to simulation thread.
    self.enabled = not self.enabled
    self.dirty = True

  def update(self, viewer, data, force=False):
    if not viewer.is_running():
      return
    if not self.enabled:
      if self.dirty:
        viewer.clear_images()
        self.dirty = False
      return
    now = time.monotonic()
    if not force and not self.dirty and now-self.last_update < 1/self.fps:
      return
    viewport = viewer.viewport
    if viewport is None:
      return
    width = min(self.size, viewport.width//3, int(viewport.height//3*self.aspect))
    height = max(1,round(width/self.aspect))
    if min(width,height) < 32:
      return
    if self.renderer is None or self.renderer.width != width or self.renderer.height != height:
      if self.renderer is not None:
        self.renderer.close()
      # Stock robot XML may only allocate a 640x480 offscreen framebuffer.
      self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
      self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
      self.renderer = mujoco.Renderer(self.model, height=height, width=width,
                                      max_geom=max(10000, self.model.ngeom+100))
    self.renderer.update_scene(data, camera=self.camera)
    image = self.renderer.render().copy()
    image[:3] = image[-3:] = (25, 190, 175)
    image[:, :3] = image[:, -3:] = (25, 190, 175)
    x = viewport.left + (12 if self.corner.endswith("left") else viewport.width-width-12)
    y = viewport.bottom + (12 if self.corner.startswith("bottom") else viewport.height-height-12)
    if viewer.is_running():
      viewer.set_images((mujoco.MjrRect(x, y, width, height), image))
    self.last_update, self.dirty = now, False

  def close(self):
    if self.renderer is not None:
      self.renderer.close()
      self.renderer = None


def native_viewer_with_inset(env, policy, enabled=True, size=256, fps=10., corner="bottom-left"):
  from mjlab.viewer import NativeMujocoViewer

  class InsetNativeViewer(NativeMujocoViewer):
    inset = None

    def setup(self):
      if enabled:
        model = self.env.unwrapped.sim.mj_model
        try:
          find_camera(model)
        except ValueError:
          pass  # Other robots may not define this LgDog camera.
        else:
          self.inset = CameraInset(model, size, fps, corner)
      try:
        super().setup()
      except BaseException:
        self.close()
        raise

    def _safe_key_callback(self, key):
      if key == KEY_F8 and self.inset is not None:
        self.inset.toggle()
      else:
        super()._safe_key_callback(key)

    def sync_env_to_viewer(self):
      super().sync_env_to_viewer()
      if self.inset is not None and self.viewer.is_running():
        with self._mj_lock:
          self.inset.update(self.viewer, self.mjd)

    def close(self):
      try:
        if self.inset is not None:
          self.inset.close()
      finally:
        super().close()

  return InsetNativeViewer(env, policy)
