import Classifiers
import RealFakeData as RFData


if __name__ == '__main__':
    X, Y = RFData.load_real_fake_data_ML_1m()
    #print(type(Y[0]))
    Classifiers.log_reg(X, Y)
