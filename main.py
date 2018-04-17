from data.handling_data import HandlingData
from models. training_model import TrainingModel

def main():

	hd = HandlingData()
	hd.main()

	tm = TrainingModel(hd.X_prepared, hd.X_train_prepared,hd.X_test_prepared, hd.y_data, hd.y_train, hd.y_test) 
	
	print(tm.non_probabilistic_evaluating_scores())
	print(tm.probabilistic_evaluating_scores())
	print(tm.perfomance_measures())

if __name__ == '__main__':
	main()