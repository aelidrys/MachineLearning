import matplotlib.pyplot as plt
import numpy as np

def error_VS_degree(train_error, test_error, degree):
    plt.figure(figsize=(15, 13))
    xDegrees = np.arange(1,degree+1)
    plt.plot(xDegrees, train_error, linestyle='-', color='orange', label="Train Error")
    plt.plot(xDegrees, test_error, linestyle='-', color='b', label="Test Error")
    plt.title("Error VS Degree", fontsize=20)
    plt.xlabel("Degree", fontsize=15)
    plt.ylabel("Error", fontsize=15)
    plt.xticks(np.arange(1, degree+1))
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
def FeaurVsError(FsErr, title="", b1_color="green", b2_color="b"):
    
    barWidth = 0.15
    indexs = np.array([0,1,2,3,4,5,6,7,8])
    br2 = [x + barWidth for x in indexs] 
    plt.subplots(figsize=(24, 12))
    plt.bar(indexs, FsErr[:,0], width=barWidth, color=b1_color, label="Train")
    plt.bar(br2, FsErr[:,1], width=barWidth, color=b2_color, label="Test")
    plt.title("Error VS Degree"+title, fontsize=20)
    plt.xlabel("Featue IDs", fontsize=15)
    plt.ylabel("RMSE", fontsize=15)
    plt.xticks([r for r in range(9)],
               ['0','1','2','3','4','5','6','7','8'])

    plt.legend()
    plt.grid(True)
    plt.show()



def alphaVSerror(alphas, errors):
    plt.figure(figsize=(15,13))
    plt.plot(alphas, errors, linestyle="-", label="test_error", color="b")
    plt.title("alphas Vs errors", fontsize=20)
    plt.xlabel("aplhas", fontsize=15)
    plt.ylabel("errors", fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
