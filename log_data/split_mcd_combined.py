import pandas as pd

values = pd.read_csv('mcd_combined.csv', sep = ' ', skiprows = 1, index_col = 1, names = ['sys', 'i', 'itr', 'dvfs', 'rapl', 'read_5th', 'read_10th', 'read_50th', 'read_90th', 'read_95th', 'read_99th', 'measure_qps', 'target_qps', 'time', 'joules', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c1', 'c1e', 'c3', 'c6', 'c7', 'num_interrupts'])

# print(values.shape)			# output = (4627, 28)

linux_values = values.loc[values['sys'] == 'linux_tuned']
ebbrt_values = values.loc[values['sys'] == 'ebbrt_tuned']

linux_features = {}
ebbrt_features = {}
linux_data = []
ebbrt_data = []

feature_cols = list(linux_values.keys())
for c in feature_cols:
	linux_features[f'{c}'] = linux_values[c]
	ebbrt_features[f'{c}'] = ebbrt_values[c]
linux_data = pd.DataFrame(linux_features)
ebbrt_data = pd.DataFrame(ebbrt_features)

linux_data.to_csv("linux_features_28.csv", sep = ' ')
ebbrt_data.to_csv("ebbrt_features_28.csv", sep = ' ')

