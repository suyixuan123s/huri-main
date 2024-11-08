import numpy as np
def gen_rand_conf(rack_size, obj_ids, obj_nums):
    rack_len = np.prod(rack_size)
    np.sum(obj_nums)


    obj_num = min(len(goal_slot_idx) + 1, obj_num)
    while True:
        num_random = np_random.randint(min_obj_num, obj_num)
        random_choiced_id = np_random.choice(range(rack_len), size=num_random, replace=False)
        elearray = np.zeros(rack_len)
        goal_selected = goal_slot_idx[np_random.choice(range(goal_slot_len), size=num_random, replace=False)]
        elearray[random_choiced_id] = goalpattern_ravel[goal_selected]
        elearray = elearray.reshape(rack_size).astype(int)
        if not check_feasible(elearray, goalpattern):
            continue
        # all the compo
        # is not done and not repeat
        if not isdone(elearray, goalpattern):
            # if not check_is_repeat(np.array(state_trained), elearray):
            break
    # state_trained.append(elearray)
    return elearray

