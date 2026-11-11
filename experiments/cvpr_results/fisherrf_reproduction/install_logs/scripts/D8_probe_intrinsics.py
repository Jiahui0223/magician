# CPU-only probe of PyTorch3D FoVPerspectiveCameras(fov 60) pixel convention at 256 x 456 (rows x cols).
import torch, numpy as np
from pytorch3d.renderer import FoVPerspectiveCameras, RasterizationSettings, MeshRasterizer
from pytorch3d.structures import Meshes
H, W = 256, 456
cam = FoVPerspectiveCameras(device="cpu", R=torch.eye(3)[None], T=torch.zeros(1, 3), zfar=750.0)  # fov 60 default, znear 1
# fit u = fx_s * (-x/z) + cx_s, v = fy_s * (-y/z) + cy_s on screen coords (PyTorch3D screen space, corner-origin)
pts = torch.tensor([[0., 0., 10.], [1., 0., 10.], [0., 1., 10.], [-2., -1., 10.]])[None]
scr = cam.transform_points_screen(pts, image_size=((H, W),))[0, :, :2].numpy()
print("screen xy:", scr.tolist())
fx = (scr[1, 0] - scr[0, 0]) / (-1. / 10.)
fy = (scr[2, 1] - scr[0, 1]) / (-1. / 10.)
print("fx_screen %.4f fy_screen %.4f  principal (corner-origin) %.4f %.4f  -> pixel-centre index cx %.4f cy %.4f" % (fx, fy, scr[0, 0], scr[0, 1], scr[0, 0] - 0.5, scr[0, 1] - 0.5))
# rasterize a fronto-parallel plane z=10 and a point check: which pixel row/col gets which ray
v = torch.tensor([[-50., -50., 10.], [50., -50., 10.], [50., 50., 10.], [-50., 50., 10.]])
f = torch.tensor([[0, 1, 2], [0, 2, 3]])
rast = MeshRasterizer(cameras=cam, raster_settings=RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1))
frag = rast(Meshes(verts=[v], faces=[f]))
z = frag.zbuf[0, ..., 0]
print("zbuf plane z=10: min %.4f max %.4f (z-depth, not ray length, if constant)" % (z.min().item(), z.max().item()))
# tilted plane: z = 10 + 0.1*x (view coords) -> check zbuf at a pixel against both pixel conventions
v2 = torch.tensor([[-50., -50., 5.], [50., -50., 15.], [50., 50., 15.], [-50., 50., 5.]])
z2 = rast(Meshes(verts=[v2], faces=[f])).zbuf[0, ..., 0].numpy()
S = 1.0 / np.tan(np.radians(30.0)); m = min(H, W)
for (u, r) in [(10, 128), (227, 128), (440, 128)]:
    # plane z = 10 + 0.1 x, x = -(u - cx)/f * z  => z = 10 / (1 + 0.1 (u - cx)/f)
    for name, fpx, cx in [("pytorch3d centre f=S*m/2, cx=(W-1)/2", S * m / 2, (W - 1) / 2), ("MAGICIAN table f=S*(m-1)/2, cx=W(m-1)/(2m)", S * (m - 1) / 2, W * (m - 1) / (2 * m))]:
        zp = 10.0 / (1.0 + 0.1 * (u - cx) / fpx)
        print("u=%3d zbuf=%.5f  pred[%s]=%.5f" % (u, z2[r, u], name, zp))
