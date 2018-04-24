from data.data import HandlingData
from models.train import BuildingModel

def main():

	hd = HandlingData()
	hd.main()

	tm = BuildingModel(hd.X_resampled,hd.X_train_resampled,hd.X_test_prepared,
                   hd.y_resampled, hd.y_train_resampled, hd.y_test)

	tm.main()


if __name__ == '__main__':
	main()
