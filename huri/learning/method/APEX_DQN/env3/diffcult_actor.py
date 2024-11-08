from huri.learning.method.APEX_DQN.env3.actor import (np,
                                                      mp,
                                                      cv2,
                                                      combine_images,
                                                      torch,
                                                      itertools,
                                                      time,
                                                      Actor,
                                                      Gym_Proto,
                                                      RackState,
                                                      RackStatePlot,
                                                      Tuple)
import traceback
import queue

class DifficultActor(Actor):
    def __init__(self,
                 actor_id: int,
                 net,
                 env: Gym_Proto,
                 max_epsilon,
                 min_epsilon,
                 epsilon_decay,
                 reset_num,
                 target_update_freq,
                 shared_net,
                 shared_state,
                 shared_reanalyzer_mem,
                 difficult_case_buffer: mp.Queue,
                 shared_mem,
                 device,
                 toggle_visual=False):

        super().__init__(actor_id,
                         net,
                         env,
                         max_epsilon,
                         min_epsilon,
                         epsilon_decay,
                         reset_num,
                         target_update_freq,
                         shared_net,
                         shared_state,
                         shared_mem,
                         device,
                         toggle_visual, )

        self.shared_reanalyzer_mem = shared_reanalyzer_mem
        self.difficult_case_buffer = difficult_case_buffer
        if self.shared_reanalyzer_mem is not None:
            self.save_data = True
        else:
            self.save_data = False

    def to_abs_state(self, state, env=None, goal_state=None):
        if env is not None:
            return np.vstack((env.goal_pattern.state, state.abs_state))
        elif goal_state is not None:
            return np.vstack((goal_state.state, state.abs_state))
        else:
            raise Exception

    def abs_to_state(self, abs_state):
        return abs_state[self.env_abs_state_slice, :]

    def select_action(self, state, env=None) -> np.ndarray:
        """Select an action from the input state."""
        if env is None:
            env = self.env
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = env.sample(state)
        else:
            selected_action = self.dqn_select_action(state.feasible_action_set, self.to_abs_state(state, env))
        self.transition = [env.goal_pattern.state, state.state, selected_action]
        # self.her_tmp_transition = [state.state, selected_action]
        return selected_action

    def step(self, action: np.ndarray, env=None, store_in_tmp=False) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        if env is None:
            env = self.env
        next_state, reward, done, _ = env.step(action)  # env step
        # reward clip:
        reward = reward
        self.transition += [reward, next_state.state, done]  #
        # self.her_tmp_transition += [reward, next_state.state]
        if store_in_tmp:
            self.replaybuffer_tmp_store.append(self.transition)

        # self._store_transition(self.transition)
        return next_state, reward, done

    def dqn_select_action(self, feasible_action_set, abs_state, no_repeat=True):
        if no_repeat:
            feasible_action_set_tmp = feasible_action_set
            repeat_acts = []
            state = RackState(abs_state[self.env_abs_state_slice, :])
            for _ in self.env.rack_state_history:
                act = self.env.action_between_states_constraint_free(state, _)
                if act is not None:
                    repeat_acts.append(act)
            feasible_action_set = np.setdiff1d(feasible_action_set, repeat_acts)
            if len(feasible_action_set) == 0:
                feasible_action_set = feasible_action_set_tmp
                # self.early_stop = True
                # # print("Potential ERROR Happens")
                # print("state is", self.env.state)
                # print("goal pattern is", self.env.goal_pattern)

        with torch.no_grad():
            feasible_action_set = torch.as_tensor(feasible_action_set, dtype=torch.int64, device=self.device)
            dqn_action_value = self.dqn(
                torch.as_tensor(abs_state[self.env_abs_state_slice, :], dtype=torch.float32,
                                device=self.device).unsqueeze(
                    0),
                torch.as_tensor(abs_state[self.env_abs_goal_slice, :], dtype=torch.float32,
                                device=self.device).unsqueeze(
                    0), ).detach()
            selected_action = feasible_action_set[dqn_action_value.squeeze()[feasible_action_set].argmax()].item()
        return selected_action

    def _reset_eposilon(self, ):
        if self.state_level != self.shared_state['state_level'] or \
                self.class_level != self.shared_state['class_level']:
            self.epsilon = 1.
            print(f"Actor {self._actor_id}: eposilon reset,{self.epsilon}. "
                  f"State level {self.state_level} "
                  f"Class level {self.class_level}")
            self.scheduler.state_level = self.state_level = self.shared_state['state_level']
            self.scheduler.class_level = self.class_level = self.shared_state['class_level']
            self.difficult_cases_list[:] = []
            return True
        else:
            return False

    def _reset_env_state_class_level(self, state_level=None, class_level=None):
        if state_level is not None:
            self.scheduler.state_level = self.state_level = self.shared_state['state_level']
        if class_level is not None:
            self.scheduler.class_level = self.class_level = self.shared_state['class_level']

    def _store_transition(self, transition):
        self.shared_mem.put(transition)

    def run(self):
        try:
            """Train the agent."""
            # init
            _step = 0
            min_epsilon = self.min_epsilon
            max_epsilon = self.max_epsilon
            env = self.env
            scheduler = env.scheduler
            self.scheduler = scheduler
            reset_num = self.reset_num
            target_update_freq = self.target_update_freq
            toggle_visual = self.toggle_visual
            plot = None

            # dqn load shared net weight
            self.dqn.load_state_dict(self.shared_net.state_dict())

            self.difficult_cases_list = []
            self.epsilon = (self.max_epsilon + self.min_epsilon) / 3
            # start training
            while True:
                if len(self.difficult_cases_list) < 1:
                    self.difficult_cases_list = self.difficult_case_buffer.get(
                        block=True)
                else:
                    try:
                        self.difficult_cases_list = self.difficult_case_buffer.get(
                            block=False)
                    except queue.Empty:
                        pass
                if len(self.difficult_cases_list) < 1:
                    self.env.to_state(np.array([1]))._cache.clear()
                    print("CLEAR")
                    continue

                self.scheduler.state_level = self.state_level = self.shared_state['state_level']
                self.scheduler.class_level = self.class_level = self.shared_state['class_level']
                init_state, goal_pattern = self.difficult_cases_list[
                    np.random.choice(np.arange(len(self.difficult_cases_list)))]
                state = env.reset_state_goal(init_state, goal_pattern)

                self.replaybuffer_tmp_store = []
                # self.her_tmp_store = []
                _reset_cnt = 0
                _score = 0

                # for curriculumn learning
                # if self._reset_eposilon():
                #     continue
                # reset_num = reset_num * self.state_level * self.class_level

                # for plot
                if toggle_visual:
                    rsp = RackStatePlot(env.goal_pattern, )
                    plot = rsp.plot_states([state]).get_img()
                    img_list = [plot]

                start_ep_t = time.time()
                for _ in itertools.count():
                    action = self.select_action(state, env)  # step
                    next_state, reward, done = self.step(action, env, store_in_tmp=True)  # next_state reward done
                    state = next_state  # state = next_state
                    _score += reward  # reward
                    _reset_cnt += 1
                    _step += 1
                    # linearly decrease epsilon
                    # self.epsilon = max(min_epsilon, self.epsilon - (max_epsilon - min_epsilon) * self.epsilon_decay)
                    # plot
                    if toggle_visual:
                        img_list.append(rsp.plot_states([state]).get_img())
                        plot = combine_images(img_list, columns=20)
                        cv2.imshow(f"plot_{self._actor_id}", plot)
                        cv2.waitKey(100)
                    if _step % target_update_freq == 0:
                        # if shared_state.get("dqn_state_dict", None) is not None:
                        if _step % (target_update_freq * 1000) == 0:
                            print(
                                f"Agent {self._actor_id} -> Update 1000 state, Epsilon {self.epsilon:.3f}, State Level {scheduler.state_level}, Class Level {scheduler.class_level}")
                        self.dqn.load_state_dict(self.shared_net.state_dict())
                        # self.dqn.load_state_dict(self.shared_state["dqn_state_dict"])
                    # if episode ends
                    if done:  # done
                        # print(f":: Episode {i_episode}: done with score: {_score}")
                        # print(f"the total length is: {len(env.reward_history)} ", env.reward_history)
                        # if self.epsilon > np.random.random():
                        #     self.gen_bootstrapping_data(self.replaybuffer_tmp_store, a_star_teacher_toggle=False,
                        #                                 her_relabel_toggle=True)
                        # print("----!Find solutions!------")
                        if self.save_data:
                            self.shared_reanalyzer_mem.put(
                                [self.state_level, self.class_level, self.replaybuffer_tmp_store])
                            for _ in self.replaybuffer_tmp_store:
                                self._store_transition(self.transition)
                        break
                    if _reset_cnt % int(np.ceil(reset_num * scheduler.state_level)) == 0:
                        # print(_reset_cnt, self.early_stop)
                        # print(f":: Episode {i_episode}: action cannot find the solution within {reset_num} steps")
                        # print(f"the total length is: {len(env.reward_history)} ", env.reward_history)
                        break
                end_ep_t = time.time()

                if toggle_visual:
                    print(f'Episode steps: {_ + 1:<4}  '
                          f'Return: {_score:<5.1f}')
                    cv2.waitKey(0)
                    del plot
                    plot = None
        except Exception as e:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR >> DIFFICULT ACTOR")
            print(traceback.format_exc())
            print(f"{e}")
