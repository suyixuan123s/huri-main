from huri.core.common_import import *
offline_data = fs.load_pickle("ik_offline_db")
armname = "rgt_arm"
offline_data_arm = offline_data.get(armname, None)
uniq_pm_f_KDtree = offline_data_arm["KDTree"]
voxel_size = offline_data_arm["voxel_size"]
group = offline_data_arm["group"]
jnt_samples = offline_data_arm["jnt_samples"]
manipuability = offline_data_arm["manipuability"]

np.save("result/rgt_jnt_samples",jnt_samples)
np.save("result/rgt_manipuability",manipuability)
fs.dump_pickle([uniq_pm_f_KDtree, group, voxel_size], "result/rgt_info")

print("Generated Finished")