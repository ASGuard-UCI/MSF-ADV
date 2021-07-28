'''
Use mitsuba renderer to obtain a depth and a reflectance image, given the
camera's rotation parameters and the file path of the object to be rendered.
'''
import numpy as np
import uuid
import os
import cv2
import subprocess
import shutil
from scipy.signal import medfilt2d

# import config
from pytorch.utils.utils import makedir_if_not_exist


render_template = \
r'''<?xml version="1.0" encoding="UTF-8"?>
<scene version="0.5.0">
    <integrator type="multichannel">
        <integrator type="field">
            <string name="field" value="{field}"/>
            <spectrum name="undefined" value="{undefined}"/>
        </integrator>
    </integrator>
    <sensor type="orthographic">
        <transform name="toWorld">
            <scale x="{sensor_scale}" y="{sensor_scale}"/>
            <lookat origin="{origin_str}" target="{target_str}" up="{up_str}"/>
        </transform>
        <sampler type="halton">
            <integer name="sampleCount" value="{sample_count}"/>
        </sampler>
        <film type="mfilm">
            <integer name="height" value="{height}"/>
            <integer name="width" value="{width}"/>
            <string name="fileFormat" value="numpy"/>
            <string name="pixelFormat" value="{pixel_format}"/>
        </film>
    </sensor>

    <shape type="shapenet">
        <string name="filename" value="{obj_path}"/>
        <float name="maxSmoothAngle" value="30"/>
    </shape>
    <!--<shape type="sphere"> <float name="radius" value="0.08"/> </shape>-->
</scene>
'''


def render_depth_refl(obj_path, theta, phi, psi, sample_count=16, height=128,
        width=128, focal_length=128, sensor_scale=1, cache_dir='./',
        cleanup=True):
    '''
    Render the depth and reflectance given those parameters.
    Axis:
       y
       |
       | /
       |/   theta
       /--------- x
      /
     /
    z

    :param obj_path: the path to the shape in wavefront obj format.
    :param theta: azimuth, in degrees
    :param phi: elevation, in degrees
    :param psi: in-plane rotation, in degrees
    :param sample_count: the halton samples for each pixel.
    :param height: image height
    :param width: image width
    :param focal_length: the distance between the camera and the origin
    :param sensor_scale: the scale of the screen space.
    :param cache_dir: the intermetiate files reside in this directory.
    :param cleanup: whether clean up the temporary files
    :return: depth - height x width numpy array
             reflectance  - height x width x 3 numpy array
             mask  - height x width numpy array, indicating whether the pixel is
             valid
    '''
    # Convert to radians
    th = theta * np.pi / 180
    ph = phi * np.pi / 180
    ps = psi * np.pi / 180

    # Compute the camera lookat parameters from Euler angles
    ox = focal_length * np.cos(th) * np.cos(ph)
    oy = focal_length * np.sin(ph)
    oz = - focal_length * np.sin(th) * np.cos(ph)
    origin = np.array([ox, oy, oz])
    target = np.array([0, 0, 0])

    n1 = np.array([-np.sin(ph) * np.cos(th), np.cos(ph),
        np.sin(ph) * np.sin(th)])
    n2 = -np.array([np.sin(th), 0, np.cos(th)])

    up = np.cos(ps) * n1 + np.sin(ps) * n2

    # Generate the scene configuration
    shared_args = dict(
        sample_count=sample_count,
        sensor_scale=sensor_scale,
        height=height,
        width=width,
        origin_str=','.join(map(str, origin)),
        target_str=','.join(map(str, target)),
        up_str=','.join(map(str, up)),
        obj_path=obj_path
    )
    depth_xml = render_template.format(field='distance', undefined='nan',
        pixel_format='luminance', **shared_args)
    refl_xml = render_template.format(field='albedo', undefined='nan,nan,nan',
        pixel_format='rgb', **shared_args)
    norm_xml = render_template.format(field='shNormal', undefined='nan,nan,nan',
        pixel_format='rgb', **shared_args)
    # pos_xml = render_template.format(field='relPosition',
        # undefined='nan,nan,nan', pixel_format='xyz', **shared_args)

    # Save to a file and call the mitsuba renderer
    cache_dir = makedir_if_not_exist(os.path.realpath(os.path.join(cache_dir, uuid.uuid4().hex)))

    depth_xml_path = os.path.join(cache_dir, 'depth.xml')
    refl_xml_path = os.path.join(cache_dir, 'refl.xml')
    # pos_xml_path = os.path.join(cache_dir, 'pos.xml')
    norm_xml_path = os.path.join(cache_dir, 'norm.xml')

    with open(depth_xml_path, 'w') as f:
        f.write(depth_xml)
    with open(refl_xml_path, 'w') as f:
        f.write(refl_xml)
    with open(norm_xml_path, 'w') as f:
        f.write(norm_xml)
    # with open(pos_xml_path, 'w') as f:
        # f.write(pos_xml)

    depth_bin_path = os.path.join(cache_dir, 'depth.npy')
    refl_bin_path = os.path.join(cache_dir, 'refl.npy')
    norm_bin_path = os.path.join(cache_dir, 'norm.npy')
    # pos_bin_path = os.path.join(cache_dir, 'pos.npy')

    env = os.environ.copy()
    MITSUBA_APPEND_PATH = None
    for k, v in MITSUBA_APPEND_PATH.items():
        if env.get(k):
            env[k] += ':' + v
        else:
            env[k] = v

    try:
        owd = os.getcwd()
        os.chdir(cache_dir)
        subprocess.check_output(None + ['depth.xml', '-o',
            'depth.npy'],
            env=env, stderr=subprocess.STDOUT
        )
        subprocess.check_output(None + ['refl.xml', '-o',
            'refl.npy'],
            env=env, stderr=subprocess.STDOUT
        )
        subprocess.check_output(None + ['norm.xml', '-o',
            'norm.npy'],
            env=env, stderr=subprocess.STDOUT
        )
        # subprocess.check_output(config.MITSUBA_COMMAND + ['pos.xml', '-o',
            # 'pos.npy'])
        os.chdir(owd)

        distance = np.load(depth_bin_path)
        refl = np.load(refl_bin_path)
        norm = np.load(norm_bin_path)
        # pos = np.load(pos_bin_path)

        assert distance is not None, depth_bin_path
        assert refl is not None, refl_bin_path
        assert norm is not None, norm_bin_path

        depth = -distance

        # Compute the mask
        umask_depth = np.isnan(depth)
        umask_refl = np.logical_or.reduce(np.isnan(refl), axis=2)
        umask_norm = np.logical_or.reduce(np.isnan(norm), axis=2)

        # umask = np.logical_or(np.logical_or(umask_depth, umask_refl), umask_norm)
        umask = np.logical_or(umask_depth, umask_refl)
        mask = np.logical_not(umask)
        umask_3 = np.stack((umask,) * 3, axis=2)

        depth[umask] = depth[mask].min()

        # Calibrate the depth so that each pixel has size (1, 1)
        depth *= width / 2 / sensor_scale
        depth_min = depth.min()
        depth -= depth.min()
        depth = medfilt2d(depth)

        refl[umask_3] = 0
        norm[umask_3] = 0

        # Compute the norm in camera space
        cam_right = n2
        cam_up = n1
        cam_towards = -origin / focal_length

        world_to_cam = np.stack((cam_right, cam_towards, cam_up))
        norm = np.einsum('ij,rcj->rci', world_to_cam, norm)

        # The axes used in mitsuba are different from our axes
        norm = norm[:, :, [0, 2, 1]]# swap y and z
        norm[:, :, 2] = -norm[:, :, 2] # flip z
        zmask = norm[:, :, 2] < 0
        zmask_3 = np.stack((zmask,) * 3, axis=2)
        norm[zmask_3] = -norm[zmask_3]
        norm = norm.astype(np.float32)


    except subprocess.CalledProcessError as e:
        print(e.output.decode())

    finally:
        if cleanup:
            shutil.rmtree(cache_dir)

    return depth, norm, refl, mask
