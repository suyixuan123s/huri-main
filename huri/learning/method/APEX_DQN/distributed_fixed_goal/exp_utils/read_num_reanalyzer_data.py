""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20240112osaka

"""

if __name__ == '__main__':
    import huri.core.file_sys as fs
    import pandas as pd

    # path_str = r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\run\log'
    path_str = r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp\data_efficiency\A_R_2'
    data_path = fs.Path(path_str)
    data_name = 'reanalyzer*'
    candidate_data = list(data_path.glob(f'**/{data_name}.csv'))
    print(f"Candidate Data Path: {candidate_data}")

    data_field_name_list = ['num_reuse', 'num_need_reuse', 'num_reduced', 'num_total_data_len', 'num_total_data']
    data_field_values = {}
    for data_field_name in data_field_name_list:
        data_field_values[data_field_name] = 0
    for data_path in candidate_data:
        data = pd.read_csv(str(data_path))
        # This assumes 'module_name' is the column name
        for data_field_name in data_field_name_list:
            filtered_data = data[data_field_name].values[-1]
            data_field_values[data_field_name] += filtered_data

    print(data_field_values)
