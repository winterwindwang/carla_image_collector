#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import math
import random
try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def draw_image_with_compose(surface, image):
    image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    surface.blit(image_surface, (0, 0))

def get_raw_img(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    return array


def get_transform(vehicle_location, angle, d=6.4):
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15))

def zoom_out(default_position, z_distance):
    location = carla.Location(default_position.location.x, default_position.location.y, z_distance)
    return carla.Transform(location, carla.Rotation(pitch=-90))
    # return carla.Transform(location)


def parse_data(data):
    pass

def main(scene_name):
    actor_list = []
    pygame.init()

    height = 608
    width = 608
    display = pygame.display.set_mode(
        (width, height),
        pygame.HWSURFACE | pygame.DOUBLEBUF)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    if scene_name == "Town10":
        world = client.get_world()
    else:
        # double load to avoid the collison of vehicle
        client.load_world(scene_name)
        world = client.load_world(scene_name)

    save_path = f"/datapath/data_{scene_name}"
    save_path_images = "datapath/images_{scene_name}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(save_path_images)

    camera_exit = False
    try:
        m = world.get_map()

        blueprint_library = world.get_blueprint_library()
        
        cam_rgb_bp = blueprint_library.find('sensor.camera.rgb')
        cam_rgb_bp.set_attribute("image_size_x", "{}".format(width))
        cam_rgb_bp.set_attribute("image_size_y", "{}".format(height))

        cam_seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        cam_seg_bp.set_attribute("image_size_x", "{}".format(width))
        cam_seg_bp.set_attribute("image_size_y", "{}".format(height))
        
        
        spawn_points_list = m.get_spawn_points()
        # only collect the aerial image (verticle viewpoint)
        camera_location = [0, 0, 10]
        camera_rotation = [0, 0, -90]
        start_pose = spawn_points_list[0]
        camera_rgb = world.spawn_actor(
            cam_rgb_bp,
            carla.Transform(carla.Location(x=start_pose.location.x, y=start_pose.location.y, z=camera_location[2]),
                            carla.Rotation(pitch=camera_rotation[2]))
        )  # SpringArm  Rigid
        actor_list.append(camera_rgb)
        camera_seg = world.spawn_actor(
            cam_seg_bp,
            carla.Transform(carla.Location(x=start_pose.location.x, y=start_pose.location.y, z=camera_location[2]),
                            carla.Rotation(pitch=camera_rotation[2]))
        )  # SpringArm  Rigid
        actor_list.append(camera_seg)


        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_seg, fps=24) as sync_mode:
            
            # travel every spawn points in each scene
            for spw_idx, spawn_point in enumerate(spawn_points_list[1:]):
                start_pose = spawn_point
                height_list = list(range(10, 101, 10))
                ####################  iteration data collector ###################
                for z_value in height_list:
                    snapshot, image_rgb, image_seg = sync_mode.tick(timeout=1.0)
                    dest_transform = zoom_out(start_pose, z_value)
                    camera_rgb.set_transform(dest_transform)
                    camera_seg.set_transform(dest_transform)

                    veh_trans_data = start_pose
                    cam_trans_data = camera_rgb.get_transform()

                    veh_trans = [[veh_trans_data.location.x, veh_trans_data.location.y, veh_trans_data.location.z],
                                 [veh_trans_data.rotation.pitch, veh_trans_data.rotation.yaw, veh_trans_data.rotation.roll]]
                    cam_trans = [[cam_trans_data.location.x, cam_trans_data.location.y, cam_trans_data.location.z],
                                 [cam_trans_data.rotation.pitch, cam_trans_data.rotation.yaw,
                                  cam_trans_data.rotation.roll]]
                    # veh_location = vehicle.get_location()

                    image_seg.convert(carla.ColorConverter.CityScapesPalette)  # (0, 0, 142) vehicle
                    image_rgb2save = get_raw_img(image_rgb)
                    image_semseg2save = get_raw_img(image_seg)
                    img_np_instance_ID = image_semseg2save[:, :, 0]
                    # get the vehicle's mask 
                    # TODO add the car in the scene
                    img_np_instance_ID = np.where(img_np_instance_ID == 142, 1, 0)
                    # Image.fromarray(img_np_instance_ID).show()

                    np.savez(f'{save_path}/scence_%s_spawn_pos_%d_altitude_%d_%05d.npz' %
                             (scene_name, spw_idx, z_value, snapshot.frame), img=image_rgb2save,
                             img_seg=img_np_instance_ID * 255, cam_trans=cam_trans, veh_trans=veh_trans)
                    Image.fromarray(image_rgb2save[:, :, ::-1]).save(
                        f"{save_path_images}/scence_%s_spawn_pos_%d_altitude_%d_%05d.png" %
                        (scene_name, spw_idx, z_value, snapshot.frame))
                    draw_image(display, image_rgb)
                    pygame.display.flip()
    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print('done.')


if __name__ == '__main__':
    from PIL import Image

    try:
        scene_list = ["Town10", "Town01", "Town05", "Town06", "Town07"]
        for scene_name in scene_list:
            main(scene_name)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')