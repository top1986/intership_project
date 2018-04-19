from data.handling_data import HandlingData
from models. training_model import BuildingModel

def main():

	hd = HandlingData()
	hd.main()

	tm = BuildingModel(hd.X_train_resampled,hd.X_test_prepared, hd.y_train_resampled, hd.y_test) 
	tm.train_and_eval_models()
	

if __name__ == '__main__':
	main()