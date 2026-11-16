# -*- coding: utf-8 -*-
"""Generate multi-vehicle DFD data in CARLA 0.10 (M7, archived issue #8).

Same capture setup as the original get_dataset2.py (Carla-0.10.0-Win64-Shipping/get_dataset2.py):
  Town10HD (whatever map is loaded; get_dataset2 tried Town05, which 0.10 does not have), first usable spawn point,
  sun behind camera (altitude 45 deg), 2048x1536, 40 mm lens (FOV from 36 mm sensor), focus 1500 cm,
  apertures f/1.4 (R) and f/22 (L), shutter 60, ISO 100, exposure_mode histogram, blur_amount 1.0, default gamma,
  camera 4.0 m ahead of the ego car at z = 0.8 m, target placed so its rear bumper is `distance` metres from the camera.
New: an instance-segmentation camera with the same transform/FOV/resolution gives an exact vehicle mask per distance
  -> <set>_mask/T<n>.npy (largest vehicle-class instance = the target directly ahead).
Unlike get_dataset2.py, the vehicle blueprint is chosen explicitly and the run aborts if it does not exist
  (no silent fallback; that is how the old audi.etron / ford.mustang folders ended up containing a nissan.patrol).

Output: <DATA>/multivehicle/<vehicle>_<color>/{Test,Training}_dataset_Focus1500cm_f1.4_f22.0/T<nn>/{R,L}/D<d>_shot<k>.png
        and the matching *_mask folder; a meta.json per set records blueprint, color, distances and actual placement.
Usage (CARLA server running on port 2000):
  ../venv-carla/Scripts/python.exe scripts/gen_multivehicle.py --vehicle vehicle.lincoln.mkz --name sedan_mkz \
      --color red=255,0,0 --set test
"""
import os, sys, json, time, math, queue, argparse, shutil
import numpy as np
import carla

DATA = os.environ.get('CADFDS_DATA', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas')
APERTURES = [1.4, 22.0]
FOCAL_MM, SENSOR_MM, FOCUS_CM = 40.0, 36.0, 1500.0
W, H = 2048, 1536
CAM_X, CAM_Z = 4.0, 0.8
VEHICLE_TAGS = {14, 15, 16, 17, 18, 19}     # CARLA semantic tags: car, truck, bus, train, motorcycle, bicycle
FOV = math.degrees(2 * math.atan(SENSOR_MM / (2 * FOCAL_MM)))


def rgb_cam(bl, fstop):
    b = bl.find('sensor.camera.rgb')
    for k, v in (('image_size_x', W), ('image_size_y', H), ('sensor_tick', 0.0), ('fov', FOV), ('focal_distance', FOCUS_CM),
                 ('fstop', fstop), ('shutter_speed', 60.0), ('iso', 100.0), ('exposure_mode', 'histogram'), ('blur_amount', 1.0)):
        b.set_attribute(k, str(v))
    return b


def seg_cam(bl):
    b = bl.find('sensor.camera.instance_segmentation')
    for k, v in (('image_size_x', W), ('image_size_y', H), ('sensor_tick', 0.0), ('fov', FOV)):
        b.set_attribute(k, str(v))
    return b


def target_mask(img):
    a = np.frombuffer(img.raw_data, np.uint8).reshape(H, W, 4)        # BGRA
    tag = a[..., 2]; inst = a[..., 1].astype(np.int32) + 256 * a[..., 0].astype(np.int32)
    veh = np.isin(tag, list(VEHICLE_TAGS))
    if not veh.any():
        return np.zeros((H, W), np.uint8), 0
    ids, cnt = np.unique(inst[veh], return_counts=True)
    best = ids[np.argmax(cnt)]
    return (veh & (inst == best)).astype(np.uint8), int(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vehicle', required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--color', required=True, help='name=R,G,B')
    ap.add_argument('--set', choices=['test', 'calib'], required=True)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--distances', type=float, nargs='*')
    a = ap.parse_args()
    cname, crgb = a.color.split('=')
    dists = a.distances or ([round(1.2 + 0.5 * i, 1) for i in range(13)] if a.set == 'test' else [round(1.0 + 0.5 * i, 1) for i in range(13)])
    kind = 'Test' if a.set == 'test' else 'Training'
    root = os.path.join(DATA, 'multivehicle', f'{a.name}_{cname}')
    out = os.path.join(root, f'{kind}_dataset_Focus{int(FOCUS_CM)}cm_f{APERTURES[0]}_f{APERTURES[1]}')
    mdir = out + '_mask'
    for d in (out, mdir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    client = carla.Client('localhost', 2000); client.set_timeout(120.0)
    world = client.get_world(); bl = world.get_blueprint_library()
    if not bl.filter(a.vehicle) or bl.filter(a.vehicle)[0].id != a.vehicle:
        sys.exit(f'blueprint {a.vehicle} not found -- refusing to fall back to another vehicle')
    settings = world.get_settings(); settings.synchronous_mode = True; settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    actors = []; npc = None; meta = dict(vehicle=a.vehicle, name=a.name, color=cname, rgb=crgb, set=a.set, map=world.get_map().name,
                                          fov_deg=FOV, focus_cm=FOCUS_CM, apertures=APERTURES, shots=a.shots, distances=[], created=time.strftime('%Y-%m-%d %H:%M'))
    try:
        ego_bp = bl.find('vehicle.firetruck.actors')          # ego only carries the cameras (as in get_dataset2.py)
        ego = None
        for sp in world.get_map().get_spawn_points():
            sp.location.z += 0.5
            ego = world.try_spawn_actor(ego_bp, sp)
            if ego:
                break
        actors.append(ego)
        world.set_weather(carla.WeatherParameters(cloudiness=0.0, precipitation=0.0, sun_azimuth_angle=sp.rotation.yaw + 180.0,
                                                  sun_altitude_angle=45.0, rayleigh_scattering_scale=0.05, fog_density=0.0))
        tf = carla.Transform(carla.Location(x=CAM_X, z=CAM_Z), carla.Rotation(yaw=0))
        cams = []
        for f in APERTURES:
            c = world.spawn_actor(rgb_cam(bl, f), tf, attach_to=ego); q = queue.Queue(); c.listen(q.put); cams.append((q, 'R' if f == APERTURES[0] else 'L')); actors.append(c)
        s = world.spawn_actor(seg_cam(bl), tf, attach_to=ego); sq = queue.Queue(); s.listen(sq.put); actors.append(s)
        npc_bp = bl.find(a.vehicle); npc_bp.set_attribute('role_name', 'npc')
        if npc_bp.has_attribute('color'):
            npc_bp.set_attribute('color', crgb)
        else:
            print('[warn] blueprint has no color attribute; color ignored')
        ego.set_simulate_physics(True)
        for _ in range(20):
            world.tick()
        et = ego.get_transform(); fwd = et.get_forward_vector()
        for i, d in enumerate(dists, 1):
            T = f'T{i:02d}'
            for sub in ('R', 'L'):
                os.makedirs(os.path.join(out, T, sub), exist_ok=True)
            if npc:
                npc.destroy(); npc = None
            loc = et.location + (d + 8.8) * fwd; loc.z += 1.0
            npc = world.try_spawn_actor(npc_bp, carla.Transform(loc, et.rotation))
            if npc is None:
                loc.z += 1.0; npc = world.try_spawn_actor(npc_bp, carla.Transform(loc, et.rotation))
            if npc is None:
                raise RuntimeError(f'cannot spawn target at {d} m')
            npc.set_simulate_physics(True)
            for _ in range(20):
                world.tick()
            npc.set_simulate_physics(False)
            half = npc.bounding_box.extent.x; gz = npc.get_location().z
            tl = et.location + (CAM_X + d + half) * fwd; tl.z = gz
            npc.set_transform(carla.Transform(tl, et.rotation))
            for _ in range(10):
                world.tick()
                for q, _ in cams + [(sq, None)]:
                    while not q.empty():
                        q.get()
            areas = []
            for k in range(1, a.shots + 1):
                world.tick()
                for q, sub in cams:
                    q.get(timeout=10.0).save_to_disk(os.path.join(out, T, sub, f'D{d:.1f}_shot{k:02d}.png'))
                m, iid = target_mask(sq.get(timeout=10.0)); areas.append(int(m.sum()))
                if k == 1:
                    np.save(os.path.join(mdir, f'T{i}.npy'), m)
            meta['distances'].append(dict(T=i, dist_m=d, half_length_m=round(half, 3), mask_px_shot1=areas[0],
                                          mask_px_min=min(areas), mask_px_max=max(areas)))
            print(f'{a.name}_{cname} {a.set} {T} {d:.1f} m  mask {areas[0]} px  (half length {half:.2f} m)', flush=True)
    finally:
        if npc:
            npc.destroy()
        for x in reversed(actors):
            if x is not None and x.is_alive:
                x.destroy()
        st = world.get_settings(); st.synchronous_mode = False; world.apply_settings(st)
        json.dump(meta, open(os.path.join(root, f'meta_{a.set}.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
