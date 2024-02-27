import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def regress_fit(X_train, y_train, X_test):
    
        def divide(x,y):
        if y!= 0:
            return x/y
        else:
            return 0

    X_train = X_train/np.max(X_train,axis=1,keepdims=True)
    X_test = X_test/np.max(X_test,axis=1,keepdims=True)

    p = X_train.shape[0] # Number of features
    N = X_train.shape[1] # Number of sample cases
    
    # Q2) Set value of learning rate e
    e = 0.000198 
    
    w = np.random.uniform(-1/np.sqrt(p), 1/np.sqrt(p), (p+1,1)) # Random initialization of
    # weights
    X = np.ones((p+1,N)) # Adding an extra column of ones to adjust biases
    X[:p,:] = X_train
   
    
    # Q3) Set number of epochs
    num_epochs = 5523
    temp = np.zeros((7,num_epochs))
    for epoch in range(num_epochs): # Loop for iterative process
        J = 0 # Initializing loss
        count = 0 # Initializing count of correct predictions
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range (N):
            z = ((w.T)@X[:,i:i+1])[0,0] # Raw logits (W.T x X)    
            
            y = 1/(1+np.exp(-z)) # Sigmoid activation function
            T = y_train[i] # Ground Truth
           
            J = J-(T*np.log(y)+(1-T)*np.log(1-y)) # Loss function
            # (Note:- The loss function is written after J = J- because we are trying to find
            # the average loss per epoch, so we need to sum it iteratively )
            
            
            k = y-T # Derivative of J w.r.t z (Chain rule, J w.r.t y multiplied by y w.r.t z )
            dJ = k*X[:,i:i+1] # Final Derivative of J w.r.t w (dJ/dz multiplied by dz/dw)
            
           
            w = w - e*dJ # Gradient Descent
            
            if abs(y-T)<0.5:
                count = count+1 # Counting the number of correct predictions
                
           

            if abs(y-T)<0.5 and T==1:
                tp = tp + 1
            if abs(y-T)<0.5 and T==0:
                tn = tn + 1
            if abs(y-T)>=0.5 and T==0:
                fp = fp + 1
            if abs(y-T)>=0.5 and T==1:
                fn = fn + 1
            
            precision = divide(tp,(tp+fp))
            recall = divide(tp,(tp+fn))
            specificity = divide(tn,(tn+fp))
            f1score = divide(2*tp,(2*tp + fp + fn))
            IoU = divide(tp,(tp+fp+fn))
            
        
        train_loss = J/N
        train_accuracy = 100*count/N

        temp[:,epoch]=[train_loss, train_accuracy, precision, recall, specificity, f1score, IoU ]
    
        


        batch_metrics = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} "
        sys.stdout.write('\r' + batch_metrics)
        sys.stdout.flush()
    
   
        
    plt.plot(range(1, num_epochs+1),temp[0,:]) #Loss vs Epoch
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.show()

    plt.plot(range(1, num_epochs+1),temp[1,:]) #Accuracy vs Epoch
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.show()

    plt.plot(range(1, num_epochs+1),temp[2,:]) #Precision vs Epoch
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision vs Epoch')
    plt.show()

    plt.plot(range(1, num_epochs+1),temp[3,:]) #Recall vs Epoch
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall vs Epoch')
    plt.show()

    plt.plot(range(1, num_epochs+1),temp[4,:]) #Specificity vs Epoch
    plt.xlabel('Epochs')
    plt.ylabel('Specificity')
    plt.title('Specificity vs Epoch')
    plt.show()

    plt.plot(range(1, num_epochs+1),temp[5,:]) #F1 Score vs Epoch
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Epoch')
    plt.show()

    plt.plot(range(1, num_epochs+1),temp[6,:]) #IoU vs Epoch
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('IoU vs Epoch')
    plt.show()
    
    
    
    # Testing
    print("\n")
    N2 = X_test.shape[1] # Number of test samples

    X2 = np.ones((p+1,N2)) # adding an additional columns of 1 to adjust biases
    X2[:p,:] = X_test

    z2 = w.T@X2 # test logit matrix
    y_pred = 1/(1+np.exp(-z2)) # Sigmoid activation function to convert into probabilities
    y_pred[y_pred>=0.5] = 1 # Thresholding
    y_pred[y_pred<0.5] = 0

    return y_pred


def regress_fit_sklearn(X_train, y_train, X_test):

    X_train = X_train.T
    X_test = X_test.T

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    poly = PolynomialFeatures(degree=3)
    X_train_scaled = poly.fit_transform(X_train_scaled)
    X_test_scaled = poly.transform(X_test_scaled)

    model = LogisticRegression(max_iter=1000, C=0.5)
    model.fit(X_train_scaled, y_train)
    y_test_pred = model.predict(X_test_scaled)

    return y_test_pred

# Load the dataset
def load_and_fit():

    df = pd.read_csv("diabetes.csv")
    X = df.drop("Outcome", axis=1)
    X2 = np.array(X)
    X2 = X2.T
    y = df["Outcome"]
    X_train = X2[:,:614]
    
    X_test = X2[:,614:]
    y_train = y[:614]
    
    y_test = y[614:]

    # Fit the model
    y_test_pred_sk = regress_fit_sklearn(X_train, y_train, X_test)
    y_test_pred = regress_fit(X_train, y_train, X_test)

    # Evaluate the accuracy
    test_accuracy_sk = accuracy_score(y_test, y_test_pred_sk)
    test_accuracy = accuracy_score(y_test, y_test_pred[0])
    print(f"Test Accuracy using sklearn: {test_accuracy_sk:.5f}")
    print(f"Test Accuracy using your implementation: {test_accuracy:.5f}")
    return round(test_accuracy, 5)

load_and_fit()
###########################################################################################################################################
