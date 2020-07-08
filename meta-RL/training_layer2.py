import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch import optim
import csv
import openpyxl
from tqdm import tqdm

PARAMETER_FILE    = 'regdata.csv'
CLASS_PARMETER_FILE_LIST = ['random_parameter_class_0.csv', 'random_parameter_class_1.csv', 'random_parameter_class_2.csv']
NUM_PARAMETER_SET = 82

MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'max-rpe-max-spe', 'min-rpe-min-spe', 'max-rpe-min-spe', 'min-rpe-max-spe']
MODE_LIST_INPUT = [[-1, 0], [1, 0], [0, -1], [0, 1], [1, 1], [-1, -1], [1, -1], [-1, 1]]
ACTION_ONE_HOT = [[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]

class SubjectDataset(Dataset):
	def __init__(self):
		point = int(82*0.7)
		"""open input csv file"""
		total_train_list = []
		total_test_list = []
		for i in range((len(MODE_LIST))):
			with open(PARAMETER_FILE) as f:
			    csv_parser = csv.reader(f)
			    param_list = []
			    for row in csv_parser:
			        param_list.append(torch.FloatTensor(list(map(float, row[:-2] + MODE_LIST_INPUT[i]))))
			total_train_list = total_train_list + param_list[:point+1]    
			total_test_list = total_test_list + param_list[point+1:]
		#label is sequence of one hot vector(length 4)
		train_label_list = []
		test_label_list = []
		for i in range(len(MODE_LIST)):
			"""open output xlsx file"""
			wb = openpyxl.load_workbook(MODE_LIST[i] + ' start index and optimal sequence.xlsx', data_only = True)
			ws = wb['Sheet1']
			all_values = []
			for row in ws.rows:
				encoded_row_value = []
				row_value = []
				for cell in row:
					row_value.append(cell.value)
				row_value = row_value[19:]
				if row_value[0]>3:
					continue
				for action in row_value:
					encoded_row_value = encoded_row_value + ACTION_ONE_HOT[action]
				all_values.append(torch.FloatTensor(encoded_row_value))
			train_label_list = train_label_list + all_values[:point+1]
			test_label_list = test_label_list + all_values[point+1:]	

		self.x_data = total_train_list
		self.y_data = train_label_list
		self.x_test = total_test_list
		self.y_test = test_label_list
		self.len = len(self.x_data)

	def __getitem__(self,index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len




class Net(nn.Module):
	def __init__(self, inner_neurons):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(8, inner_neurons)
		self.f1_bn = nn.BatchNorm1d(inner_neurons)
		self.fc2 = nn.Linear(inner_neurons,inner_neurons)
		self.f2_bn = nn.BatchNorm1d(inner_neurons)
		self.fc3 = nn.Linear(inner_neurons,140)

		nn.init.xavier_normal_(self.fc1.weight)
		nn.init.xavier_normal_(self.fc2.weight)
		nn.init.xavier_normal_(self.fc3.weight)
		nn.init.normal_(self.fc1.bias)
		nn.init.normal_(self.fc2.bias)
		nn.init.normal_(self.fc3.bias)		

	def forward(self, x):
		#x = F.relu(self.fc1(x))
		#x = F.relu(self.fc2(x))
		x = F.relu(self.f1_bn(self.fc1(x)))
		x = F.relu(self.f2_bn(self.fc2(x)))
		x = self.fc3(x)
		return x




if __name__ == '__main__':
	init_rl = 0.01
	net = Net(256)	
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=init_rl)
	dataset = SubjectDataset()
	train_loader = DataLoader(dataset = dataset, batch_size = 32, shuffle=True, num_workers= 0)
	TOTAL_EPISODE = 100
	for epoch in tqdm(range(TOTAL_EPISODE)):
		#adjust learning rate
		lr = init_rl*(0.1**(epoch//(TOTAL_EPISODE//10)))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
		for i, data in enumerate(train_loader):
			# get the inputs
			inputs, labels = data
			# wrap them in Variable
			inputs ,labels = Variable(inputs), Variable(labels)

			# Forward pass: Compute predicted y by passing x to the model
			y_pred = net(inputs)

			# Compute and print loss
			loss = criterion(y_pred, labels)
			#print(epoch, i, loss.data)

			# Zero gradients, perform a backward pass, and update the weights.
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()



	torch.save(net, '/home/ksh/Documents/URP/MB-MF-Study/ControlRL/Net_140output' +str(TOTAL_EPISODE) )

	
	#validation 1 for test set
	total = 0
	total2 = 0
	for num in range(len(dataset.x_test)):
		net.eval()
		test_input = dataset.x_test[num]
		target = dataset.y_test[num]
		test_input , target = Variable(test_input), Variable(target)
		test_output = net(test_input[None, ...])
		test_output = test_output[0]

		decoded_test_output = []
		for i in range(0,len(test_output),4):
			#print(torch.argmax(test_output[i:i+4]))
			decoded_test_output.append(torch.argmax(test_output[i:i+4]))
		decoded_test_output = Variable(torch.FloatTensor(decoded_test_output) )	
		
		decoded_target = []
		for i in range(0,len(target),4):
			#print(torch.argmax(test_output[i:i+4]))
			decoded_target.append(torch.argmax(target[i:i+4]))
		decoded_target = Variable(torch.FloatTensor(decoded_target) )
		#print("#####################COMPARE_%s###############################" % num)		
		#print(decoded_test_output, decoded_target)
		#print(test_input, test_output, target)
		#print("################################################################")
		loss = criterion(test_output, target  )
		total += loss.item()
		loss2 = criterion(decoded_test_output, decoded_target)
		total2 += loss2.item()
	total_avg = total/len(dataset.x_test)
	total2_avg = total2/len(dataset.x_test)
	print("-----------------------VALIDATION #1 for test data-----------------------------")
	print("encoded_total : %s decoded_total2 : %s test_data_len : %s" %(total, total2, len(dataset.x_test)))
	print("avg_encoded_total : %s agv_decoded_total2 : %s" %(total_avg, total2_avg))


	#validation 2 for random set
	total_input_list = []
	for class_num in range(3):
		file_name = CLASS_PARMETER_FILE_LIST[class_num]	
		for mode_num in range((len(MODE_LIST))):
			with open(file_name) as f:
				csv_parser = csv.reader(f)
				param_list = []
				for row in csv_parser:
					param_list.append(torch.FloatTensor(list(map(float, row[:-1] + MODE_LIST_INPUT[mode_num]))))
			total_input_list = total_input_list + param_list[:20]

	total_output_list = []
	for class_num in range(3):
		for mode_num in range(len(MODE_LIST)):
			file_location = 'parameter_extract/class' + str(class_num) + '_20subject/optimal sequence/' +MODE_LIST[mode_num] + ' optimal sequence class' + str(class_num)+'.xlsx'
			wb = openpyxl.load_workbook(file_location, data_only = True)
			ws = wb['Sheet1']
			one_sheet_values = []
			for row in ws.rows:
				row_value = []
				for cell in row:
					row_value.append(cell.value)	
				row_value = row_value[1:]
				encoded_row_value = []
				if row_value[10]>3:
					continue
				for action in row_value:
					encoded_row_value = encoded_row_value + ACTION_ONE_HOT[action]
				one_sheet_values.append(torch.FloatTensor(encoded_row_value))
			total_output_list = total_output_list + one_sheet_values		

	total = 0
	total2 = 0

	for num in range(len(total_input_list)):
		net.eval()
		test_input = total_input_list[num]
		target = total_output_list[num]
		test_input , target = Variable(test_input), Variable(target)
		test_output = net(test_input[None, ...])
		test_output = test_output[0]

		decoded_test_output = []
		for i in range(0,len(test_output),4):
			#print(torch.argmax(test_output[i:i+4]))
			decoded_test_output.append(torch.argmax(test_output[i:i+4]))
		decoded_test_output = Variable(torch.FloatTensor(decoded_test_output) )	
		
		decoded_target = []
		for i in range(0,len(target),4):
			#print(torch.argmax(test_output[i:i+4]))
			decoded_target.append(torch.argmax(target[i:i+4]))
		decoded_target = Variable(torch.FloatTensor(decoded_target) )
		#print("#####################COMPARE_%s###############################" % num)		
		#print(decoded_test_output, decoded_target)
		#print(test_input, test_output, target)
		#print("################################################################")
		loss = criterion(test_output, target  )
		total += loss.item()
		loss2 = criterion(decoded_test_output, decoded_target)
		total2 += loss2.item()
	total_avg = total/len(total_input_list)
	total2_avg = total2/len(total_input_list)
	print("-----------------------VALIDATION #2 for random data-----------------------------")
	print("encoded_total : %s decoded_total2 : %s test_data_len : %s" %(total, total2, len(total_input_list)))
	print("avg_encoded_total : %s agv_decoded_total2 : %s" %(total_avg, total2_avg))

