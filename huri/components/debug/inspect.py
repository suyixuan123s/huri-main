from modeling.model_collection import ModelCollection

rbt_data = []
ik_error = []
ik_error_pointer = [ik_error]
rbt_collision_error = []
rbt_hnd_error = []
plot_node = [None]


class _Info:
    pass


class IKError(_Info):
    def __init__(self, grasp_pos,
                 grasp_rot,
                 grasp_jaw,
                 hnd_instance,
                 seed_jnt=None,
                 env_obs=None):
        self.grasp_pos = grasp_pos
        self.grasp_rot = grasp_rot
        self.grasp_jaw = grasp_jaw
        self.hnd_instance = hnd_instance
        self.seed_jnt = seed_jnt
        self.env_obs = env_obs


class RbtInfo(_Info):
    def __init__(self,
                 jnts,
                 hand_name,
                 jawwidth,
                 rbt_instance):
        self.jnts = jnts
        self.hand_name = hand_name
        self.jawwidth = jawwidth
        self.rbt_instance = rbt_instance


class RbtCollisionError(RbtInfo):
    def __init__(self, jnts,
                 hand_name,
                 jawwidth,
                 env_obs,
                 rbt_instance):
        super(RbtCollisionError, self).__init__(jnts, hand_name, jawwidth, rbt_instance)
        self.env_obs = env_obs


class RbtHandCollisionError(_Info):
    def __init__(self,
                 hnd,
                 gl_jaw_center_pos,
                 gl_jaw_center_rotmat,
                 jawwidth,
                 env_obs,
                 rgba, ):
        self.hnd = hnd
        self.gl_jaw_center_pos = gl_jaw_center_pos
        self.gl_jaw_center_rotmat = gl_jaw_center_rotmat
        self.jawwidth = jawwidth
        self.env_obs = env_obs
        self.rgba = rgba


def clear_ik_err():
    ik_error_pointer[0] = []


def save_rbt_info(arm_jnts,
                  rbt_instance,
                  hand_name="rgt_hnd",
                  jawwidth=0):
    rbt_data.append(RbtInfo(jnts=arm_jnts,
                            hand_name=hand_name,
                            jawwidth=jawwidth,
                            rbt_instance=rbt_instance))


def save_error_info(error_info: _Info, ):
    if isinstance(error_info, IKError):
        ik_error.append(error_info)
    elif isinstance(error_info, RbtCollisionError):
        rbt_collision_error.append(error_info)
        print("ERROR SAVED")
    elif isinstance(error_info, RbtHandCollisionError):
        rbt_hnd_error.append(error_info)
    else:
        raise Exception("Error Not Defined")


def show_animation(info_type="ik_error"):
    assert info_type in ["ik_error", "cd_error", "rbt_info", 'rbt_hnd_error']
    counter = [0]
    print("inspect")

    data = []
    if info_type == "ik_error":
        data = ik_error
    elif info_type == "cd_error":
        data = rbt_collision_error
    elif info_type == "rbt_info":
        data = rbt_data
    elif info_type == "rbt_hnd_error":
        data = rbt_hnd_error

    print(len(data))

    def update(data,
               counter,
               task):
        if base.inputmgr.keymap["space"]:
            base.inputmgr.keymap["space"] = False
            if counter[0] > len(data) - 1:
                counter[0] = 0
                return task.again
            else:
                if plot_node[0] is not None:
                    plot_node[0].detach()
                plot_node[0] = ModelCollection()
                info = data[counter[0]]
                print(len(data))
                if isinstance(info, RbtCollisionError):
                    print("Test")
                    hand_name = info.hand_name
                    arm_jnts = info.jnts
                    env_obs = info.env_obs
                    jawwidth = info.jawwidth
                    robot_s = info.rbt_instance
                    robot_s.fk(hand_name, arm_jnts)
                    robot_s.jaw_to(hand_name, jawwidth)
                    robot_meshmodel = robot_s.gen_meshmodel()
                    robot_meshmodel.attach_to(plot_node[0])
                    robot_s.show_cdprimit()
                    for _ in env_obs:
                        _.attach_to(plot_node[0])
                        _.show_cdprimit()
                    plot_node[0].attach_to(base)
                elif isinstance(info, IKError):
                    hnd_instance = info.hnd_instance
                    grasp_pos = info.grasp_pos
                    grasp_rot = info.grasp_rot
                    grasp_jaw = info.grasp_jaw
                    env_obs = info.env_obs
                    hand_cm = hnd_instance.gen_meshmodel()
                    hand_cm.attach_to(plot_node[0])
                    hand_cm.show_cdprimit()
                    if env_obs is not None:
                        for _ in env_obs:
                            _.attach_to(plot_node[0])
                            _.show_cdprimit()
                    plot_node[0].attach_to(base)
                elif isinstance(info, RbtInfo):
                    hand_name = info.hand_name
                    arm_jnts = info.jnts
                    jawwidth = info.jawwidth
                    robot_s = info.rbt_instance
                    robot_s.fk(hand_name, arm_jnts)
                    robot_s.jaw_to(hand_name, jawwidth)
                    robot_meshmodel = robot_s.gen_meshmodel()
                    robot_meshmodel.attach_to(plot_node[0])
                    print(arm_jnts)
                    print(robot_s.get_gl_tcp(hand_name))
                    # robot_s.show_cdprimit()
                    plot_node[0].attach_to(base)
                elif isinstance(info, RbtHandCollisionError):
                    hnd_instance = info.hnd
                    hnd_instance.grip_at_with_jcpose(gl_jaw_center_pos=info.gl_jaw_center_pos,
                                                     gl_jaw_center_rotmat=info.gl_jaw_center_rotmat,
                                                     jaw_width=info.jawwidth
                                                     )
                    env_obs = info.env_obs
                    hand_cm = hnd_instance.gen_meshmodel(rgba=info.rgba)
                    hand_cm.attach_to(plot_node[0])
                    hand_cm.show_cdprimit()
                    if env_obs is not None:
                        for _ in env_obs:
                            _.attach_to(plot_node[0])
                            _.show_cdprimit()
                    plot_node[0].attach_to(base)

                counter[0] += 1
                print(counter[0])
        return task.again

    taskMgr.doMethodLater(0.02, update, "update",
                          extraArgs=[data, counter],
                          appendTask=True)
    base.run()
