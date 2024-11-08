from huri.core.common_import import *
import time
import visualization.panda.world as wd
import modeling.geometric_model as gm
import basis

base = wd.World(cam_pos=[3, 1, 1], lookat_pos=[0, 0, 0.5])
gm.gen_frame().attach_to(base)
yumi_instance = ym.Yumi(enable_cc=True)

# yumi_meshmodel = yumi_instance.gen_meshmodel(toggle_tcpcs=True)
# yumi_instance.fk("rgt_hnd", np.zeros(7))
# yumi_instance.fk("lft_hnd", np.zeros(7))
# yumi_meshmodel.attach_to(base)
# yumi_instance.show_cdprimit()
# base.run()
# ik test

component_name = 'rgt_arm'
tgt_pos = np.array([.3, -.4, .1])
tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], np.pi / 3)
gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
print(repr(yumi_instance.rgt_arm.homeconf))
tic = time.time()
for i in range(50):
    jnt_values = yumi_instance.ik(component_name, tgt_pos, tgt_rotmat, toggle_debug=False)
toc = time.time()
print(toc - tic)
print(jnt_values)
exit(0)
yumi_instance.fk(component_name, jnt_values)
yumi_instance.gen_meshmodel().attach_to(base)
yumi_instance.fk(component_name, np.array([1.17331584, -1.99621953, -1.08811406, -0.18234367, 0.66571608,
                                           1.26591, 0.18141696]))
yumi_instance.gen_meshmodel(rgba=[1, 0, 0, 1]).attach_to(base)
# ik_pos, ik_rot = yumi_instance.get_gl_tcp(component_name)
# print(f"pos value is: {ik_pos}, diff is {(ik_pos-tgt_pos)*1000} mm")

base.run()
exit(0)
yumi_meshmodel = yumi_instance.gen_meshmodel()
yumi_meshmodel.attach_to(base)
yumi_instance.gen_stickmodel().attach_to(base)
tic = time.time()
result = yumi_instance.is_collided()
toc = time.time()
print(result, toc - tic)
base.run()

# hold test
component_name = 'lft_arm'
obj_pos = np.array([-.1, .3, .3])
obj_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
objfile = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
objcm = cm.CollisionModel(objfile, cdprimit_type='cylinder')
# objcm.set_pos(obj_pos)
# objcm.set_rotmat(obj_rotmat)
objcm.attach_to(base)
objcm_copy = objcm.copy()
print("obj at init", objcm_copy.get_homomat())
rel_pos, rel_rotmat = yumi_instance.hold(objcm=objcm_copy, jaw_width=0.03, hnd_name='lft_hnd')

# objcm_copy.is

tgt_pos = np.array([.4, .5, .4])
tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 3)
jnt_values = yumi_instance.ik(component_name, tgt_pos, tgt_rotmat)
yumi_instance.fk(component_name, jnt_values)
# yumi_instance.show_cdprimit()
yumi_meshmodel = yumi_instance.gen_meshmodel()
yumi_meshmodel.attach_to(base)
print(yumi_instance.get_oih_relhomomat(objcm_copy, "lft_hnd"))

print("obj at goal", objcm_copy.get_homomat())
yumi_instance.get_gl_tcp()
base.run()
