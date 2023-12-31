# Trash Classification Using Convolutional Neural Networks
1. To install required libraries: pip install -r requirements.txt
2. To train the model from scratch:
a. Data augmentation:
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,height_shift_range=0.1, rotation_range=45, zoom_range=0.1,
        horizontal_flip=True,vertical_flip=True)
    train_gen.fit(train_X)
b. Building AlexNet model (softmax activation):
    def AlexNetCE(input_shape = (227, 227, 3), classes = 6):
        X_input = tf.keras.Input(input_shape)
        X = X_input
        X = tf.keras.layers.Conv2D(96, (11, 11), strides = (4, 4), activation = "relu", name = 'conv1')(X)
        X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
        X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = tf.keras.layers.Conv2D(256, (5, 5), padding = "same",activation = "relu", name = 'conv2')(X)
        X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
        X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_conv2')(X)
        X = tf.keras.layers.Conv2D(256, (3, 3), padding = "same",activation = "relu", name = 'conv5')(X)
        X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(4096, activation = "relu", name='fc' + str(1))(X)
        X = tf.keras.layers.Dense(4096, activation = "relu", name='fc' + str(2))(X)
        X = tf.keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes))(X)
        model = tf.keras.Model(inputs = X_input, outputs = X, name='ALEXNETCE')
        return model
c. Model training (categorical cross entropy loss):
    model = AlexNetCE()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
d. Inference:
    error = [0,0,0,0,0,0]
    mislabel = np.zeros((6,6))
    prediction = model.predict(test_X_re)
    for i in range(test_X_re.shape[0]):
        if(test_Y_re[np.argmax(prediction,axis=1)[i],i] == 0):
            if(i<101):
                error[0] = error[0]+1
                mislabel[0][np.argmax(prediction,axis=1)[i]] += 1
            elif(i<226):
                error[1] = error[1]+1
                mislabel[1][np.argmax(prediction,axis=1)[i]] += 1
            elif(i<329):
                error[2] = error[2]+1
                mislabel[2][np.argmax(prediction,axis=1)[i]] += 1
            elif(i<478):
                error[3] = error[3]+1
                mislabel[3][np.argmax(prediction,axis=1)[i]] += 1
            elif(i<599):
                error[4] = error[4]+1
                mislabel[4][np.argmax(prediction,axis=1)[i]] += 1
            else:
                error[5] = error[5]+1
                mislabel[5][np.argmax(prediction,axis=1)[i]] += 1
    total = np.array([101,125,103,149,121,34])
    error = np.array(error)
    error = np.divide(error,total)
    print("% of mislabeled cardboard: "+str(error[0]))
    print("Mostly mislabeled as: "+str(np.argmax(mislabel,axis=0)[0]))
    print("% of mislabeled glass: "+str(error[1]))
    print("Mostly mislabeled as: "+str(np.argmax(mislabel,axis=0)[1]))
    print("% of mislabeled metal: "+str(error[2]))
    print("Mostly mislabeled as: "+str(np.argmax(mislabel,axis=0)[2]))
    print("% of mislabeled paper: "+str(error[3]))
    print("Mostly mislabeled as: "+str(np.argmax(mislabel,axis=0)[3]))
    print("% of mislabeled plastic: "+str(error[4]))
    print("Mostly mislabeled as: "+str(np.argmax(mislabel,axis=0)[4]))
    print("% of mislabeled trash: "+str(error[5]))
    print("Mostly mislabeled as: "+str(np.argmax(mislabel,axis=0)[5]))

 The result of model inference is shown as below:
 % of mislabeled cardboard: 0.10891089108910891
 Mostly mislabeled as: 2
 % of mislabeled glass: 0.152
 Mostly mislabeled as: 4
 % of mislabeled metal: 0.46601941747572817
 Mostly mislabeled as: 3
 % of mislabeled paper: 0.12080536912751678
 Mostly mislabeled as: 4
 % of mislabeled plastic: 0.3305785123966942
 Mostly mislabeled as: 1
 % of mislabeled trash: 0.35294117647058826
 Mostly mislabeled as: 2
d. Model evaluation: 
      for i in range(10):
        print("Training Process "+str(len(train_loss)))
        history = model.fit(train_X, train_Y.T,batch_size=32,epochs=1)
        print("Test Result "+str(len(train_loss)))
        preds = model.evaluate(test_X, test_Y.T)
        print ("Loss = " + str(preds[0]))
        print ("Test Accuracy = " + str(preds[1]))
        train_loss.append(history.history["loss"][0])
        train_acc.append(history.history["accuracy"][0])
        test_loss.append(preds[0])
        test_acc.append(preds[1])
    plt.plot(train_loss)
    plt.show()
    plt.plot(test_loss)
    plt.show()

 The evaluation of the model is shown as below:
 ![image](https://github.com/congduytran12/Trash-Classification-/assets/109121562/a5c0d753-1aec-45d2-99e1-6180c4d335b3)
 ![image](https://github.com/congduytran12/Trash-Classification-/assets/109121562/c96faf89-6ea8-4300-932a-c431ee9f687a)
