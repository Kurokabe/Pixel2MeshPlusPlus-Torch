import torch
from PIL import Image, ImageDraw


def normal(v):
    norm = torch.norm(v, dim=-1, keepdim=True)
    if norm.all() == 0:
        return v
    return torch.divide(v, norm)


def camera_mat(param):
    azimuth = param[0]
    elevation = param[1]
    yaw = param[2]
    distance = param[3]
    fov = param[4]

    theta = azimuth * torch.pi / 180.0
    camy = distance * torch.sin(elevation * torch.pi / 180.0)
    lens = distance * torch.cos(elevation * torch.pi / 180.0)
    camx = lens * torch.cos(theta)
    camz = lens * torch.sin(theta)

    Z = torch.stack([camx, camy, camz])

    x = camy * torch.cos(theta + torch.pi)
    z = camy * torch.sin(theta + torch.pi)
    Y = torch.stack([x, lens, z], dim=-1)
    X = torch.linalg.cross(Y, Z)

    cm_mat = torch.stack([normal(X), normal(Y), normal(Z)])
    return cm_mat, Z


def camera_trans(camera_metadata, points):
    c, o = camera_mat(camera_metadata)
    pt_trans = points - o
    transpose_c = torch.transpose(c, 1, 0)
    pt_trans = torch.matmul(pt_trans, transpose_c)
    return pt_trans


def batch_camera_trans(camera_metadata, points):
    c, o = batch_camera_mat(camera_metadata)
    o = o.unsqueeze(1)
    pt_trans = points - o
    transpose_c = torch.transpose(c, 2, 1)
    pt_trans = torch.matmul(pt_trans, transpose_c)
    return pt_trans


def camera_trans_inv(camera_metadata, xyz):
    c, o = camera_mat(camera_metadata)
    transpose_c = torch.transpose(c, 1, 0)
    inv_transpose_c = torch.linalg.inv(transpose_c)
    inv_xyz = torch.matmul(xyz, inv_transpose_c)
    inv_xyz = inv_xyz + o
    return inv_xyz


def batch_camera_trans_inv(camera_metadata, xyz):
    c, o = batch_camera_mat(camera_metadata)
    transpose_c = torch.transpose(c, 2, 1)
    inv_transpose_c = torch.linalg.inv(transpose_c)
    inv_xyz = torch.matmul(xyz, inv_transpose_c)
    o = o.unsqueeze(1)
    inv_xyz = inv_xyz + o
    return inv_xyz


def batch_camera_mat(param):
    azimuth = param[:, 0]
    elevation = param[:, 1]
    yaw = param[:, 2]
    distance = param[:, 3]
    fov = param[:, 4]

    theta = azimuth * torch.pi / 180.0
    camy = distance * torch.sin(elevation * torch.pi / 180.0)
    lens = distance * torch.cos(elevation * torch.pi / 180.0)
    camx = lens * torch.cos(theta)
    camz = lens * torch.sin(theta)

    Z = torch.stack([camx, camy, camz], dim=1)

    x = camy * torch.cos(theta + torch.pi)
    z = camy * torch.sin(theta + torch.pi)
    Y = torch.stack([x, lens, z], dim=1)
    X = torch.linalg.cross(Y, Z)

    cm_mat = torch.stack([normal(X), normal(Y), normal(Z)], dim=1)
    return cm_mat, Z


def draw_image_from_points_projection(h, w):
    # create a new image
    img = Image.new("RGB", (224, 224), color="white")

    # create a draw object
    draw = ImageDraw.Draw(img)

    # iterate over each h and w coordinate and draw a red dot
    for i in range(len(h)):
        draw.ellipse((w[i] - 1, h[i] - 1, w[i] + 1, h[i] + 1), fill="red")

    # display the image
    img.show()
