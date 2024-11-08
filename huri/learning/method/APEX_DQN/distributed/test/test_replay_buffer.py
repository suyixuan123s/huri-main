""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230719osaka

"""

if __name__ == '__main__':
    from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer
    from ray.rllib.utils.replay_buffers.replay_buffer import StorageUnit
    from ray.rllib.policy.sample_batch import SampleBatch

    # Store any batch as a whole
    buffer = ReplayBuffer(capacity=10, storage_unit=StorageUnit.FRAGMENTS)
    buffer.add(SampleBatch({"a": [1], "b": [2, 3, 4]}))
    buffer.sample(1)

    # Store only complete episodes
    buffer = ReplayBuffer(capacity=10,
                          storage_unit=StorageUnit.EPISODES)
    buffer.add(SampleBatch({"c": [1, 2, 3, 4],
                            SampleBatch.T: [0, 1, 0, 1],
                            SampleBatch.TERMINATEDS: [False, True, False, True],
                            SampleBatch.EPS_ID: [0, 0, 1, 1]}))
    buffer.sample(1, .6)

    # Store single timesteps
    buffer = ReplayBuffer(capacity=2, storage_unit=StorageUnit.TIMESTEPS)
    buffer.add(SampleBatch({"a": [1, 2], SampleBatch.T: [0, 1]}))
    buffer.sample(1)

    buffer.add(SampleBatch({"a": [3], SampleBatch.T: [2]}))
    print(buffer._eviction_started)
    buffer.sample(1)

    buffer = ReplayBuffer(capacity=10, storage_unit=StorageUnit.SEQUENCES)
    buffer.add(SampleBatch({"c": [1, 2, 3], SampleBatch.SEQ_LENS: [1, 2]}))
    buffer.sample(1)
